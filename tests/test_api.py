import pytest
import os
import sys
import math
from fastapi.testclient import TestClient

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем приложение
from api.main import app

# Создаем клиент - lifespan события выполнятся автоматически
client = TestClient(app)

def calculate_features(credit_score, age, tenure, balance, num_products,
                       has_cr_card, is_active_member, estimated_salary, gender):
    """Calculate all 12 features from raw data"""
    return [
        float(credit_score),
        float(math.log1p(age)),
        float(tenure),
        float(math.log1p(balance)),
        float(num_products),
        float(has_cr_card),
        float(is_active_member),
        float(math.log1p(estimated_salary)),
        float(gender),
        float(balance / (estimated_salary + 1)),
        float(tenure / (age + 1)),
        float(credit_score / (age + 1))
    ]


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert data["service"] == "churn-prediction-api"


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "predict" in data


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "features_count" in data


def test_predict_with_valid_data():
    """Test prediction with valid data"""
    # Проверяем, загружена ли модель
    health_response = client.get("/health")
    if not health_response.json().get("model_loaded", False):
        pytest.skip("Model not loaded - test requires model file")
    
    features = calculate_features(
        credit_score=720, age=32, tenure=5, balance=50000,
        num_products=2, has_cr_card=1, is_active_member=1,
        estimated_salary=120000, gender=0
    )
    
    payload = {"features": features, "customer_id": "test_001"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "test_001"
    assert 0 <= data["churn_probability"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]


def test_predict_invalid_features_count():
    """Test with wrong number of features"""
    payload = {
        "features": [0.5, 0.3, 0.2],
        "customer_id": "invalid_001"
    }
    response = client.post("/predict", json=payload)
    
    # Если модель не загружена -> 503, если загружена -> 400
    health_response = client.get("/health")
    if health_response.json().get("model_loaded", False):
        assert response.status_code == 400
    else:
        assert response.status_code == 503


def test_predict_missing_customer_id():
    """Test without customer_id"""
    payload = {"features": [0.5] * 12}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_ready_endpoint():
    """Test readiness endpoint"""
    response = client.get("/ready")
    health_response = client.get("/health")
    
    if health_response.json().get("model_loaded", False):
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
    else:
        assert response.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
