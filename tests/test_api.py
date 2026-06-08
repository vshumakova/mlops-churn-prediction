import pytest
import sys
import os
import math
import requests

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Принудительно загружаем модель ДО тестов
from api.main import load_model, app
load_model()

# Теперь можно использовать TestClient
from fastapi.testclient import TestClient

client = TestClient(app)

BASE_URL = "http://localhost:8000"

def calculate_features(credit_score, age, tenure, balance, num_products,
                       has_cr_card, is_active_member, estimated_salary, gender):
    """Calculate all 12 features from raw data"""
    return [
        credit_score,
        math.log1p(age),
        tenure,
        math.log1p(balance),
        num_products,
        has_cr_card,
        is_active_member,
        math.log1p(estimated_salary),
        gender,
        balance / (estimated_salary + 1),
        tenure / (age + 1),
        credit_score / (age + 1)
    ]

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    print(f"Health response: {response.status_code} - {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_active_client():
    """Test with an active young client (likely to stay)"""
    features = calculate_features(
        credit_score=720,
        age=32,
        tenure=5,
        balance=50000,
        num_products=2,
        has_cr_card=1,
        is_active_member=1,
        estimated_salary=120000,
        gender=0
    )
    
    payload = {
        "features": features,
        "customer_id": "active_client_001"
    }
    response = client.post("/predict", json=payload)
    print(f"Predict response status: {response.status_code}")
    
    if response.status_code == 503:
        print("❌ Model not loaded! Check if model.pkl exists")
        print(f"Model file exists? {os.path.exists('models/model.pkl')}")
        print(f"API model: {app.model}")
    
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0 <= data["churn_probability"] <= 1

def test_predict_elderly_client():
    """Test with an elderly inactive client (likely to leave)"""
    features = calculate_features(
        credit_score=580,
        age=65,
        tenure=2,
        balance=20000,
        num_products=1,
        has_cr_card=0,
        is_active_member=0,
        estimated_salary=30000,
        gender=1
    )
    
    payload = {
        "features": features,
        "customer_id": "elderly_client_002"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid_features():
    """Test with invalid number of features"""
    payload = {
        "features": [0.5, 0.3],
        "customer_id": "test123"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_metrics():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "endpoints" in response.json()

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
