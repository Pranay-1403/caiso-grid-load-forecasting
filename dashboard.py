import streamlit as st
import requests
import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="CAISO Grid Forecasting & AI Dispatcher", page_icon="⚡", layout="wide")

st.title("⚡ CAISO Grid Load Forecasting & Autonomous Dispatcher")
st.write("Production microservice architecture powered by PyTorch TFT, FastAPI, Open-Meteo, and an autonomous LangChain agent.")

GEMINI_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# Tool 1: Live FastAPI Grid Forecast Microservice on Render
@tool
def fetch_caiso_load_forecast() -> dict:
    """Queries the live CAISO Forecasting FastAPI microservice on Render to retrieve 
    the active status and 24-to-168-hour forecasted grid load parameters."""
    url = "https://caiso-grid-load-forecasting.onrender.com/predict"
    try:
        response = requests.get(url, timeout=45)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Endpoint returned status code {response.status_code}"}
    except Exception as e:
        return {"error": f"Failed to connect to forecast service: {str(e)}"}

# Tool 2: Live California Weather Ingestion via Open-Meteo
@tool
def fetch_california_ambient_weather() -> dict:
    """Queries Open-Meteo to fetch current ambient weather conditions (temperature, 
    relative humidity, apparent temperature, and wind speed) for the Southern California load center."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 34.0522,
        "longitude": -118.2437,
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "wind_speed_10m"],
        "temperature_unit": "fahrenheit",
        "timezone": "America/Los_Angeles"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            return res.json().get("current", {})
        return {"error": f"Open-Meteo returned status {res.status_code}"}
    except Exception as e:
        return {"error": f"Failed to fetch weather: {str(e)}"}

# Agent Executor Builder
def get_agent_executor(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.2
    )
    tools = [fetch_caiso_load_forecast, fetch_california_ambient_weather]
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert Energy Market & Transmission Grid Dispatch Analyst for the California "
            "Independent System Operator (CAISO). Your role is to examine live load forecasts and "
            "cross-examine them against real-time ambient weather (cooling/heating load drivers). "
            "Detect potential reserve margin strains, evaluate net-load ramp risks, and generate concise, "
            "executive-ready grid status briefings using your available tools."
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

# UI Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Inference")
    if st.button("Get 24-Hour Forecast"):
    api_url = "https://caiso-grid-load-forecasting.onrender.com/predict"
    
    with st.spinner("Fetching forecast data..."):
        try:
            response = requests.get(api_url, timeout=45)
            if response.status_code == 200:
                data = response.json()
                st.success("Successful response from FastAPI!")
                
                # --- REPLACE st.json(data) WITH THIS ---
                forecast_list = data.get("forecast", [])
                df = pd.DataFrame(forecast_list)
                
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                    
                    st.subheader("Predicted Grid Load (MW)")
                    st.line_chart(df["forecast_load_mw"])
                    
                    with st.expander("View Detailed Hourly Table"):
                        st.dataframe(df)
                # ---------------------------------------
            else:
                st.error(f"Error {response.status_code}: Unable to retrieve forecast.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

with col2:
    st.subheader("🤖 AI Dispatcher Co-Pilot")
    if st.button("Generate Dispatcher Briefing"):
        if not GEMINI_KEY:
            st.warning("Please configure your GOOGLE_API_KEY in Streamlit Secrets.")
        else:
            with st.spinner("Agent querying grid microservice and live weather telemetry..."):
                try:
                    executor = get_agent_executor(GEMINI_KEY)
                    query = (
                        "Query both the CAISO forecasting microservice and current California ambient weather. "
                        "Synthesize an operational executive briefing evaluating: "
                        "1. Live microservice status and telemetry health "
                        "2. Current ambient weather conditions and their expected impact on cooling/heating load "
                        "3. Critical operational risks (e.g. net-load ramp, peak strain) and recommended dispatcher actions."
                    )
                    raw_result = executor.invoke({"input": query})
                    
                    output = raw_result.get("output", "")
                    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                        clean_text = output[0].get("text", "")
                    elif isinstance(output, str):
                        clean_text = output
                    else:
                        clean_text = str(output)

                    st.markdown(clean_text)
                except Exception as e:
                    err_msg = str(e)
                    if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                        st.warning("⚠️ API Quota Limit Reached: The Gemini free-tier rate limit was hit. Please wait a minute and retry.")
                    else:
                        st.error(f"Agent execution failed: {err_msg}")
