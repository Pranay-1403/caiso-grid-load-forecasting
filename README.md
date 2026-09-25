# ⚡ CAISO Grid Load Forecasting & Autonomous AI Dispatcher

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://caiso-grid-load-forecasting-koj7j4lyxtfjtbkzffsw9j.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch_Forecasting-EE4C2C.svg?logo=pytorch)](https://pytorch-forecasting.readthedocs.io/)
[![LangChain](https://img.shields.io/badge/Agent-LangChain-1C3C3C.svg)](https://www.langchain.com/)
[![Render](https://img.shields.io/badge/Deployment-Render-46E3B7.svg?logo=render)](https://render.com/)

🔗 **Live Interactive Demo:** [CAISO Grid Load Forecasting & Dispatcher Dashboard](https://caiso-grid-load-forecasting-koj7j4lyxtfjtbkzffsw9j.streamlit.app/)

---

## 📌 Executive Summary

Modern power grids with high penetrations of solar and wind generation face extreme volatility, commonly reflected in the California Independent System Operator (CAISO) "Duck Curve." Accurately projecting multi-horizon electricity consumption while dynamically reacting to thermal load drivers is critical for preventing grid instability and brownouts.

This project delivers an **end-to-end, production-grade MLOps and Agentic AI solution**:
1. **Deep Learning Forecasting:** Implements a **Temporal Fusion Transformer (TFT)** trained on 5 years of historical CAISO data that models complex diurnal, seasonal, and weather-driven dependencies.
2. **Microservice Serving:** Deploys the predictive inference pipeline as an asynchronous **FastAPI microservice** hosted on Render.
3. **Autonomous AI Dispatcher:** Integrates an agentic co-pilot powered by **LangChain** and **Google Gemini** that pulls real-time grid and ambient meteorological telemetry to generate executive dispatch briefings and risk analyses.
4. **Persistent Operations UI:** Delivers a dual-pane **Streamlit** console featuring persistent session caching and error-resilient API communication.

---

## 🏗️ System Architecture

```text
       +-----------------------------------------------------+
       |               User / Grid Dispatcher                |
       |      (Web Interface: Streamlit Cloud Dashboard)     |
       +--------------------------+--------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v [Click: Get Forecast]                           v [Click: Briefing]
+---------------------------------+             +------------------------------------+
|     Persistent Session State    |             |   LangChain ReAct Agent Executor   |
|   (st.session_state Caching)    |             |      (Powered by Gemini Flash)     |
+----------------+----------------+             +---------+----------------+---------+
                 |                                        |                |
                 | HTTP GET                               | Tool Call 1    | Tool Call 2
                 v                                        v                v
+---------------------------------+             +-------------------+  +-------------------+
|      FastAPI Microservice       | <-----------+ Render API Route  |  |  Open-Meteo API   |
|     (Hosted on Render.com)      |             |   /predict        |  | Real-Time Weather |
| - TFT Multi-Horizon Ingestion   |             +-------------------+  +-------------------+
| - 24-168h Load Generation (MW)  |                       |                      |
+---------------------------------+                       +----------+-----------+
                                                                     |
                                                                     v
                                                +------------------------------------+
                                                |  Thermodynamic & Risk Synthesizer  |
                                                |  - Ramp Risk (Duck Curve / BESS)   |
                                                |  - Intertie Headroom (Path 26/66)  |
                                                +------------------+-----------------+
                                                                   |
                                                                   v
                                                +------------------------------------+
                                                | Live Executive Briefing Displayed  |
                                                +------------------------------------+

```

---

## 🛠️ Tech Stack & Tooling

| Domain | Technologies |
| --- | --- |
| **Deep Learning & Modeling** | PyTorch Forecasting, PyTorch Lightning, XGBoost (Baseline), Pandas, NumPy |
| **Agentic AI & LLMs** | LangChain Core, LangChain Community, Google Gemini (`gemini-3.8-flash`) |
| **Backend & Microservices** | FastAPI, Uvicorn, Requests, Pydantic |
| **Frontend & Monitoring** | Streamlit, Streamlit Session State, Altair |
| **Telemetry & Ingestion** | CAISO GridStatus API, Open-Meteo Weather API |
| **Cloud Hosting** | Render (FastAPI Web Service), Streamlit Community Cloud (Frontend) |

---

## 🤖 Autonomous AI Dispatcher Co-Pilot

Beyond static forecast plots, the platform functions as an active decision-support system for transmission dispatchers. The agent employs dynamic function calling with two custom tools:

* **`fetch_caiso_load_forecast`**: Queries the live FastAPI microservice on Render to extract forward-looking megawatt load intervals and verify data ingestion integrity.
* **`fetch_california_ambient_weather`**: Pulls dry-bulb temperature, heat index, relative humidity, and surface wind velocities for the Southern California load center via Open-Meteo.

### Operational Intelligence Synthesis

* **Thermal HVAC Drivers:** Evaluates apparent heat and humidity retention to determine Cooling Degree Days (CDD) and peak evening air-conditioning persistence.
* **Duck-Curve Net-Load Transition:** Assesses the steepness of solar generation drop-offs against load ramps ($\text{MW}/\Delta t$) to alert operators to rapid ramp stress.
* **Actionable Mitigation Directives:** Formulates operational recommendations including Battery Energy Storage System (BESS) pre-charging schedules, flexible peaker commitments, and Path 26/46/66 intertie transfer headroom checks.

---

## 📂 Project Repository Structure

```text
├── 01_CAISO_Data_Pipeline_and_Baseline.ipynb  # Ingestion pipeline (GridStatus + Open-Meteo) & XGBoost baselines
├── 02caiso_load_forecasting_tft.ipynb         # Temporal Fusion Transformer training & quantile evaluation
├── api.py                                     # FastAPI microservice serving real-time predictions
├── dashboard.py                               # Production Streamlit UI with LangChain agent integration
├── requirements.txt                           # Production environment dependencies
└── README.md                                  # System documentation

```

---

## 🚀 Local Installation & Setup

### 1. Clone Repository & Install Dependencies

```bash
git clone [https://github.com/](https://github.com/Pranay-1403/caiso-grid-load-forecasting.git)
cd caiso-grid-load-forecasting
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Configure Environment Variables

Create a `.env` file in the root directory (or `.streamlit/secrets.toml` for Streamlit):

```env
GOOGLE_API_KEY="your-google-gemini-api-key"

```

### 3. Run FastAPI Backend Microservice

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

```

Test the health endpoint at `http://localhost:8000/` and the forecast endpoint at `http://localhost:8000/predict`.

### 4. Launch Streamlit Operations Console

In a separate terminal window:

```bash
streamlit run dashboard.py

```

---

## 🛡️ Production Engineering & Fault Tolerance

* **Session Persistence (`st.session_state`):** Prevents UI data loss during page re-renders, ensuring predictive time series charts and AI briefings persist alongside one another.
* **API Rate Limit Fallbacks:** Implements targeted handling for Google Gemini free-tier quotas (`429 RESOURCE_EXHAUSTED`) and cloud server capacity spikes (`503 UNAVAILABLE`) with actionable user warnings.
* **Non-Blocking Telemetry Queries:** Custom timeout controls ensure third-party meteorological latency does not hang the dispatcher thread.
