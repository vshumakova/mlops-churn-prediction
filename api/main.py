from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import mlflow
from typing import List, Dict
import os

app = FastAPI(title="Churn Prediction API", version="1.0")

# Global variables
model = None
model_version = "v1.0"

class PredictionRequest(BaseModel):
    features: List[float]
    user_id: str

class PredictionResponse(BaseModel):
    user_id: str
    churn_probability: float
    prediction: int
    model_version: str

@app.on_event("startup")
async def load_model():
    """Loading the model"""
    global model
    try:
        model = joblib.load('model.pkl') if os.path.exists('model.pkl') else None
        print("Model loaded successfully")
    except Exception as e:
        print(f"Could not load model: {e}")
        model = None

@app.get("/health")
async def health_check():
    """Healthcheck endpoint для Docker"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "service": "churn-prediction-api"
    }

@app.get("/ready")
async def readiness_check():
    """Readiness probe"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Churn prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    features = np.array(request.features).reshape(1, -1)
    probability = model.predict_proba(features)[0][1]
    prediction = 1 if probability > 0.5 else 0
    
    return PredictionResponse(
        user_id=request.user_id,
        churn_probability=float(probability),
        prediction=prediction,
        model_version=model_version
    )

@app.get("/metrics")
async def get_metrics():
    """Metrics for monitoring"""
    return {
        "model_version": model_version,
        "model_loaded": model is not None,
        "endpoints": ["/health", "/ready", "/predict", "/metrics"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
