from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os
import logging
from typing import List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")

model = None
model_version = "1.0.0"

class PredictionRequest(BaseModel):
    features: List[float]
    customer_id: str

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: int
    risk_level: str
    recommendation: str
    model_version: str
    timestamp: str

@app.on_event("startup")
async def load_model():
    global model
    # Пробуем разные пути
    paths = ['api/model.pkl', 'model.pkl', 'models/model.pkl']
    for path in paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"✅ Model loaded from {path}")
                return
            except Exception as e:
                logger.error(f"Failed: {e}")
    logger.warning("⚠️ Model not found")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "service": "churn-prediction-api",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    features = np.array(request.features).reshape(1, -1)
    probability = float(model.predict_proba(features)[0][1])
    prediction = 1 if probability > 0.5 else 0
    
    if probability < 0.3:
        risk_level = "Low"
        recommendation = "Monitor only"
    elif probability < 0.6:
        risk_level = "Medium"
        recommendation = "Send email offer"
    elif probability < 0.8:
        risk_level = "High"
        recommendation = "Call customer"
    else:
        risk_level = "Critical"
        recommendation = "Urgent manager call"
    
    return PredictionResponse(
        customer_id=request.customer_id,
        churn_probability=probability,
        prediction=prediction,
        risk_level=risk_level,
        recommendation=recommendation,
        model_version=model_version,
        timestamp=datetime.now().isoformat()
    )

@app.get("/")
async def root():
    return {"message": "Churn Prediction API", "health": "/health", "predict": "/predict"}