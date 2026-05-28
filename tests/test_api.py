import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    payload = {
        "features": [0.5, 0.3, -0.2, 1.0, -0.5],
        "customer_id": "test123"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert data["customer_id"] == "test123"

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "endpoints" in response.json()

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
