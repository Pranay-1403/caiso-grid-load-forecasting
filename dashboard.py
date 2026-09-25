import streamlit as st
import requests

st.title("⚡ CAISO Grid Load Forecasting")
st.write("This dashboard connects directly to your active FastAPI microservice.")

if st.button("Get 24-Hour Forecast"):
    # Send a request to the FastAPI backend running on local port 8000
    response = requests.get("https://caiso-grid-load-forecasting.onrender.com/predict")

    if response.status_code == 200:
        st.success("Successful response from FastAPI!")
        st.json(response.json())
    else:
        st.error("Failed to connect to the microservice.")
