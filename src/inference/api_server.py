# src/inference/api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import cudf
import joblib
import requests
from typing import List

from src.models.simple_lstm import load_model, predict_future
from src.preprocessing.normalization import GPUNormalizer

app = FastAPI(
    title="Passaic River Water Level Forecast API",
    description="GPU-accelerated LSTM forecast for Jersey City water levels",
    version="0.1.0"
)

# Globals
model = None
y_normalizer = None

@app.on_event("startup")
async def startup_event():
    global model, y_normalizer
    print("API server starting up...")
    
    try:
        print("  Loading LSTM model...")
        model = load_model("best_lstm_model.pth", input_size=3)
        print("  Model loaded successfully.")
        
        print("  Loading normalizer...")
        y_normalizer = joblib.load("y_normalizer.pkl")
        print("  Normalizer loaded successfully.")
        
    except FileNotFoundError as e:
        print(f"  ERROR: File not found - {e}")
        print("  Make sure 'best_lstm_model.pth' and 'y_normalizer.pkl' exist in the project root.")
        raise RuntimeError("Required model or normalizer file missing.")
    
    except Exception as e:
        print(f"  Unexpected startup error: {e}")
        raise
    
    print("API startup complete — ready to serve requests.")

class ForecastRequest(BaseModel):
    last_sequence: List[List[float]]  # shape: 96 timesteps × 3 features

class ForecastResponse(BaseModel):
    forecast: List[float]  # 12 predicted steps (feet)
    status: str = "success"

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Manual forecast: provide your own last 96 timesteps × 3 features.
    """
    if model is None or y_normalizer is None:
        raise HTTPException(503, detail="Model or normalizer not loaded yet. Try again in a few seconds.")

    try:
        # Validate input shape
        if len(request.last_sequence) != 96 or any(len(row) != 3 for row in request.last_sequence):
            raise HTTPException(400, detail="Input must be exactly 96 timesteps × 3 features")

        last_seq = np.array(request.last_sequence, dtype=np.float32).reshape(1, 96, 3)

        future_norm = predict_future(model, last_seq, steps=12)

        future_df = cudf.DataFrame(future_norm.reshape(-1, 1), columns=['target'])
        future_denorm = y_normalizer.inverse_transform(future_df, ['target'])['target'].to_numpy()

        return ForecastResponse(forecast=future_denorm.tolist())

    except Exception as e:
        raise HTTPException(500, detail=f"Forecast error: {str(e)}")

@app.get("/forecast-auto")
async def forecast_auto():
    """
    Automatic forecast using latest gage height from USGS + repeated dummy history.
    """
    if model is None or y_normalizer is None:
        raise HTTPException(503, detail="Model or normalizer not loaded yet.")

    try:
        # Fetch current gage height
        current_gage = get_latest_gage_data()

        # Placeholder history: repeat current gage + dummy wind/precip
        # In real use: load last 96 rows from cache/DB/USGS history
        dummy_row = [current_gage, 1.0, 0.0]  # gage_height_ft, AWND, precip_mm
        last_seq_list = [dummy_row] * 96
        last_seq = np.array(last_seq_list, dtype=np.float32).reshape(1, 96, 3)

        # Predict next 12 steps
        future_norm = predict_future(model, last_seq, steps=12)

        # Denormalize
        future_df = cudf.DataFrame(future_norm.reshape(-1, 1), columns=['target'])
        future_denorm = y_normalizer.inverse_transform(future_df, ['target'])['target'].to_numpy()

        return {
            "current_gage_height_ft": current_gage,
            "forecast_next_12_steps_ft": future_denorm.tolist(),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(500, detail=f"Auto-forecast error: {str(e)}")

def get_latest_gage_data():
    """Fetch latest gage height from USGS site 01388500 (Little Falls, Passaic River)"""
    url = "https://waterservices.usgs.gov/nwis/iv/?format=json&sites=01388500&parameterCd=00065&period=P1D"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        values = data['value']['timeSeries'][0]['values'][0]['value']
        latest = float(values[-1]['value'])
        return latest
    except Exception as e:
        raise RuntimeError(f"USGS fetch failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "normalizer_loaded": y_normalizer is not None
    }

@app.get("/latest-gage")
async def latest_gage():
    gage = get_latest_gage_data()
    return {
        "latest_gage_height_ft": gage,
        "note": "Fetched live from USGS site 01388500 (Little Falls, Passaic River)"
    }