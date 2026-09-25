from fastapi import FastAPI

app = FastAPI(title="CAISO Forecasting Microservice")

@app.get("/")
def health_check():
    return {"status": "Online", "message": "FastAPI is hosting the Temporal Fusion Transformer!"}

@app.get("/predict")
def get_prediction():
    # In a full production pipeline, we would ingest live Pandas data here.
    return {"forecast": "Endpoint is active and ready to receive 168-hour grid load tensors."}
