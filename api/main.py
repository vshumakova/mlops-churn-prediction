from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os
import logging
from typing import List, Dict
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="ML Model for customer churn prediction",
    version="1.0.0"
)

# Global variables
model = None
model_version = os.getenv("MODEL_VERSION", "1.0.0")
model_path = os.getenv("MODEL_PATH", "/app/models/model.pkl")

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of feature values")
    customer_id: str = Field(..., description="Customer identifier")

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: int
    risk_level: str
    recommendation: str
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    service: str
    timestamp: str

@app.on_event("startup")
async def load_model():
    """Load model at startup"""
    global model
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
    else:
        logger.warning(f"Model not found at {model_path}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=model_version,
        service="churn-prediction-api",
        timestamp=datetime.now().isoformat()
    )

@app.get("/ready")
async def readiness_check():
    """Readiness probe"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict churn probability for a customer"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        features = np.array(request.features).reshape(1, -1)
        probability = float(model.predict_proba(features)[0][1])
        prediction = 1 if probability > 0.5 else 0
        
        # Determine risk level and recommendation
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
    """Get model metrics"""
    return {
        "model_loaded": model is not None,
        "model_version": model_version,
        "endpoints": ["/health", "/ready", "/predict", "/metrics"],
        "features_count": 5  # Update based on your model
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }
