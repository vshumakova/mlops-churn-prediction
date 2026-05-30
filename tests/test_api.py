import pytest
from fastapi.testclient import TestClient
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

client = TestClient(app)

def calculate_features(credit_score, age, tenure, balance, num_products,
                       has_cr_card, is_active_member, estimated_salary, gender):
    """Calculate all 12 features from raw data"""
    return [
        credit_score,                           # CreditScore
        math.log1p(age),                        # Age_log
        tenure,                                 # Tenure
        math.log1p(balance),                    # Balance_log
        num_products,                           # NumOfProducts
        has_cr_card,                            # HasCrCard
        is_active_member,                       # IsActiveMember
        math.log1p(estimated_salary),           # Salary_log
        gender,                                 # Gender (0=Male, 1=Female)
        balance / (estimated_salary + 1),       # BalanceSalaryRatio
        tenure / (age + 1),                     # TenureByAge
        credit_score / (age + 1)                # CreditScoreGivenAge
    ]

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_active_client():
    """Test with an active young client (likely to stay)"""
    # An active young client with high income
    features = calculate_features(
        credit_score=720,
        age=32,
        tenure=5,
        balance=50000,
        num_products=2,
        has_cr_card=1,
        is_active_member=1,
        estimated_salary=120000,
        gender=0  # Male
    )
    
    payload = {
        "features": features,
        "customer_id": "active_client_001"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    print(f"\nAn active young client")
    print(f"The probability of outflow: {data['churn_probability']:.2%}")
    print(f"The level of risk: {data['risk_level']}")
    print(f"Recommendation: {data['recommendation']}\n")

    
    assert "churn_probability" in data
    assert "prediction" in data
    assert 0 <= data["churn_probability"] <= 1

def test_predict_elderly_client():
    """Test with an elderly inactive client (likely to leave)"""
    # An elderly inactive client with a low income
    features = calculate_features(
        credit_score=580,
        age=65,
        tenure=2,
        balance=20000,
        num_products=1,
        has_cr_card=0,
        is_active_member=0,
        estimated_salary=30000,
        gender=1  # Female
    )
    
    payload = {
        "features": features,
        "customer_id": "elderly_client_002"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    print(f"\nAn elderly inactive client")
    print(f"The probability of outflow: {data['churn_probability']:.2%}")
    print(f"The level of risk: {data['risk_level']}")
    print(f"Recommendation: {data['recommendation']}\n")
    
    assert "churn_probability" in data
    assert "prediction" in data
    assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid_features():
    """Test with invalid number of features"""
    payload = {
        "features": [0.5, 0.3],
        "customer_id": "test123"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

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
