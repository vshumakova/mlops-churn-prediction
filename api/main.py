from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import numpy as np
import joblib
import os
import logging
import math
from typing import List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
model_version = "1.0.0"

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of 12 feature values")
    customer_id: str = Field(..., description="Customer identifier")

class CustomerData(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    num_products: int
    has_cr_card: bool
    is_active_member: bool
    estimated_salary: float
    gender: str

@app.post("/predict/v2")
async def predict_v2(customer: CustomerData):
    """Predict churn with readable JSON instead of array"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Convert to features
    gender_val = 0 if customer.gender == "Male" else 1
    
    features = calculate_features(
        credit_score=customer.credit_score,
        age=customer.age,
        tenure=customer.tenure,
        balance=customer.balance,
        num_products=customer.num_products,
        has_cr_card=int(customer.has_cr_card),
        is_active_member=int(customer.is_active_member),
        estimated_salary=customer.estimated_salary,
        gender=gender_val
    )
    
    # Get prediction
    features_array = np.array(features).reshape(1, -1)
    probability = float(model.predict_proba(features_array)[0][1])
    prediction = 1 if probability > 0.5 else 0
    
    # Risk level mapping
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
    
    return {
        "churn_probability": round(probability, 4),
        "churn_risk_percent": f"{probability*100:.1f}%",
        "prediction": "Will churn" if prediction == 1 else "Will stay",
        "risk_level": risk_level,
        "recommendation": recommendation,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: int
    risk_level: str
    recommendation: str
    model_version: str
    timestamp: str

def calculate_features(credit_score, age, tenure, balance, num_products,
                       has_cr_card, is_active_member, estimated_salary, gender):
    """
    Calculate all 12 features from raw data
    """
    return [
        float(credit_score),                    # CreditScore
        float(math.log1p(age)),                 # Age_log
        float(tenure),                          # Tenure
        float(math.log1p(max(balance, 0))),     # Balance_log
        float(num_products),                    # NumOfProducts
        float(has_cr_card),                     # HasCrCard
        float(is_active_member),                # IsActiveMember
        float(math.log1p(estimated_salary)),    # Salary_log
        float(gender),                          # Gender (0=Male, 1=Female)
        float(balance / (estimated_salary + 1)), # BalanceSalaryRatio
        float(tenure / (age + 1)),              # TenureByAge
        float(credit_score / (age + 1))         # CreditScoreGivenAge
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Starting up...")
    
    path = 'models/model.pkl'
    
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load from {path}: {e}")
    
    if model is None:
        logger.warning("Model not found in any location")
    
    yield
    
    logger.info("Shutting down...")
    model = None

app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# statistic monitoring
if os.path.exists('api/static'):
    app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/ui", response_class=HTMLResponse)
async def web_interface():
    with open('api/static/index.html', 'r') as f:
        return HTMLResponse(content=f.read())

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
        
