import streamlit as st
import requests
import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="CAISO Grid Forecasting & AI Dispatcher", page_icon="⚡", layout="wide")

st.title("⚡ CAISO Grid Load Forecasting & Autonomous Dispatcher")
st.write("Production microservice architecture powered by PyTorch TFT, FastAPI, and an autonomous LangChain agent.")

# Retrieve the key securely from Streamlit secrets or environment
GEMINI_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# 1. Define the LangChain Tool
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

# 2. Build the Agent
def get_agent_executor(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.2
    )
    tools = [fetch_caiso_load_forecast]
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert Energy Market & Transmission Grid Analyst for the California "
            "Independent System Operator (CAISO). Your role is to examine live load forecasts, "
            "detect potential reserve margin strains, evaluate risks, and generate concise, "
            "executive-ready grid status briefings. Use the tools available to inspect current data."
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

# 3. Interactive Interface Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Inference")
    if st.button("Get 24-Hour Forecast"):
        with st.spinner("Calling FastAPI microservice on Render..."):
            try:
                res = requests.get("https://caiso-grid-load-forecasting.onrender.com/predict", timeout=45)
                if res.status_code == 200:
                    st.success("Successful response from FastAPI!")
                    st.json(res.json())
                else:
                    st.error(f"Error: Microservice returned status {res.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with col2:
    st.subheader("🤖 AI Dispatcher Co-Pilot")
    if st.button("Generate Dispatcher Briefing"):
        if not GEMINI_KEY:
            st.warning("Please configure your GOOGLE_API_KEY in Streamlit Secrets or pass it in the sidebar.")
        else:
            with st.spinner("AI Agent querying microservice & synthesizing grid conditions..."):
                try:
                    executor = get_agent_executor(GEMINI_KEY)
                    query = (
                        "Check our live CAISO forecasting service. Confirm the endpoint status "
                        "and generate a 3-bullet executive morning briefing covering grid readiness, "
                        "operational risks, and recommended dispatcher action."
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
                    st.error(f"Agent execution failed: {e}")
