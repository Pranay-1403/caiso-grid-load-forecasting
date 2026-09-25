# CAISO Grid Load Forecasting: End-to-End MLOps Pipeline

## Project Overview
This project implements a Temporal Fusion Transformer (TFT) to forecast 168-hour electricity grid loads for the California Independent System Operator (CAISO). The deep learning model significantly outperforms traditional XGBoost baselines in both RMSE and MAE metrics by capturing complex multi-horizon temporal dependencies. 

The trained model is operationalized as a microservice and connected to an interactive front-end, demonstrating a complete end-to-end data science deployment pipeline.

## Tech Stack
* **Machine Learning:** PyTorch Forecasting, PyTorch Lightning, Pandas, NumPy
* **API Backend:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit
* **Environment:** Google Colab, Google Drive Integration

## Architecture
1. **Model Training:** A TFT neural network trained on 5 years of historical CAISO load data, optimizing for multi-step time series forecasting.
2. **REST API:** The saved model weights (`.ckpt`) are loaded into a FastAPI microservice that exposes a `/predict` endpoint.
3. **Interactive Dashboard:** A Streamlit web application consumes the FastAPI endpoint, allowing users to request and visualize 24-hour to 168-hour forecasts on demand.

## Project Structure
* `CAISO_Forecasting_Model.ipynb`: Complete code for data ingestion, preprocessing, model training, and evaluation.
* `api.py`: FastAPI server script handling model inference.
* `dashboard.py`: Streamlit interface for client interaction.
