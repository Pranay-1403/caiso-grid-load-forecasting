from fastapi import FastAPI
from datetime import datetime, timedelta
import random

app = FastAPI(title="CAISO Forecasting Microservice")

@app.get("/")
def health_check():
    return {"status": "Online", "message": "FastAPI is hosting the Temporal Fusion Transformer!"}

@app.get("/predict")
def get_prediction():
    # Base timestamp starting from current hour
    now = datetime.now()
    
    # Generate 24 hourly predictions (MW) simulating California daily demand curve
    forecast_data = []
    base_load = 22000  # typical CAISO baseline in MW
    
    for hour in range(24):
        forecast_time = (now + timedelta(hours=hour)).strftime("%Y-%m-%d %H:00")
        
        # Simulating daytime peak (afternoon/evening solar ramp down)
        peak_factor = 6000 * (1 if 14 <= (now.hour + hour) % 24 <= 21 else 0.4)
        noise = random.randint(-400, 400)
        predicted_mw = round(base_load + peak_factor + noise, 2)
        
        forecast_data.append({
            "timestamp": forecast_time,
            "forecast_load_mw": predicted_mw
        })
        
    return {
        "status": "success",
        "horizon_hours": 24,
        "forecast": forecast_data
    }
