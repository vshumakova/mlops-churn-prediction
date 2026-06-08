from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import numpy as np
import joblib
import os
import logging
from typing import List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для модели
model = None
model_version = "1.0.0"

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of 12 feature values")
    customer_id: str = Field(..., description="Customer identifier")

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: int
    risk_level: str
    recommendation: str
    model_version: str
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: загружаем модель
    global model
    logger.info("Starting up...")
    
    paths = [
        'models/model.pkl',
        'api/models/model.pkl',
        'model.pkl'
    ]
    
    for path in paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"Model loaded from {path}")
                break
            except Exception as e:
                logger.error(f"Failed to load from {path}: {e}")
    
    if model is None:
        logger.warning("Model not found in any location")
    
    yield
    
    # Shutdown: очищаем ресурсы
    logger.info("Shutting down...")
    global model
    model = None

# Создаем приложение с lifespan
app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "service": "churn-prediction-api",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ready")
async def readiness_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Проверка количества признаков
    if len(request.features) != 12:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 12 features, got {len(request.features)}"
        )
    
    try:
        features = np.array(request.features).reshape(1, -1)
        probability = float(model.predict_proba(features)[0][1])
        prediction = 1 if probability > 0.5 else 0
        
        if probability < 0.3:
            risk_level = "Low"
            recommendation = "Monitor only, no action needed"
        elif probability < 0.6:
            risk_level = "Medium"
            recommendation = "Send personalized email offer"
        elif probability < 0.8:
            risk_level = "High"
            recommendation = "Call customer + discount offer"
        else:
            risk_level = "Critical"
            recommendation = "Urgent: Manager call + premium retention offer"
        
        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=probability,
            prediction=prediction,
            risk_level=risk_level,
            recommendation=recommendation,
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    return {
        "model_loaded": model is not None,
        "model_version": model_version,
        "endpoints": ["/health", "/ready", "/predict", "/metrics"],
        "features_count": 12
    }

@app.get("/")
async def root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }
