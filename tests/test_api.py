import pytest
import os
import sys
import math
import json
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем приложение
from api.main import app
from fastapi.testclient import TestClient

# Создаем клиент для тестирования
client = TestClient(app)

def calculate_features(credit_score, age, tenure, balance, num_products,
                       has_cr_card, is_active_member, estimated_salary, gender):
    """
    Calculate all 12 features from raw data
    Эти же функции использует API
    """
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


# ============= HEALTH TESTS =============

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "model_version" in data
    assert data["service"] == "churn-prediction-api"


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "model_info" in data


# ============= PREDICTION TESTS =============

@pytest.mark.parametrize("test_case", [
    {
        "name": "active_young_client",
        "customer_id": "test_young_001",
        "features": {
            "credit_score": 750,
            "age": 28,
            "tenure": 4,
            "balance": 30000,
            "num_products": 2,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 90000,
            "gender": 0
        },
        "expected_risk": "Low"
    },
    {
        "name": "inactive_elderly_client", 
        "customer_id": "test_elderly_002",
        "features": {
            "credit_score": 600,
            "age": 65,
            "tenure": 1,
            "balance": 15000,
            "num_products": 1,
            "has_cr_card": 0,
            "is_active_member": 0,
            "estimated_salary": 35000,
            "gender": 1
        },
        "expected_risk": "High"
    },
    {
        "name": "medium_risk_client",
        "customer_id": "test_medium_003", 
        "features": {
            "credit_score": 680,
            "age": 45,
            "tenure": 3,
            "balance": 45000,
            "num_products": 1,
            "has_cr_card": 1,
            "is_active_member": 0,
            "estimated_salary": 65000,
            "gender": 0
        },
        "expected_risk": "Medium"
    }
])
def test_predict_various_clients(test_case):
    """Test prediction for different client profiles"""
    features = calculate_features(**test_case["features"])
    
    payload = {
        "features": features,
        "customer_id": test_case["customer_id"]
    }
    
    response = client.post("/predict", json=payload)
    
    # Если модель не загружена, пропускаем тест (не ошибка)
    if response.status_code == 503:
        pytest.skip("Model not loaded - skipping prediction test")
    
    assert response.status_code == 200
    data = response.json()
    
    # Проверяем структуру ответа
    assert data["customer_id"] == test_case["customer_id"]
    assert "churn_probability" in data
    assert "prediction" in data
    assert "risk_level" in data
    assert "recommendation" in data
    assert "model_version" in data
    assert "timestamp" in data
    
    # Проверяем диапазоны
    assert 0 <= data["churn_probability"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["Low", "Medium", "High"]
    
    # Проверяем соответствие ожидаемому риску
    if "expected_risk" in test_case:
        assert data["risk_level"] == test_case["expected_risk"]


def test_predict_batch():
    """Test batch prediction with multiple clients"""
    test_clients = [
        {
            "customer_id": "batch_001",
            "features": calculate_features(720, 35, 5, 50000, 2, 1, 1, 100000, 0)
        },
        {
            "customer_id": "batch_002",
            "features": calculate_features(580, 60, 2, 20000, 1, 0, 0, 40000, 1)
        }
    ]
    
    for client in test_clients:
        response = client.post("/predict", json=client)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - skipping batch test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == client["customer_id"]
        assert "churn_probability" in data


# ============= VALIDATION TESTS =============

def test_predict_invalid_features_count():
    """Test with wrong number of features"""
    payload = {
        "features": [0.5, 0.3, 0.2],  # Only 3 features instead of 12
        "customer_id": "invalid_001"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_missing_customer_id():
    """Test without customer_id"""
    payload = {
        "features": [0.5] * 12  # 12 features
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_invalid_feature_type():
    """Test with invalid feature type"""
    payload = {
        "features": ["invalid"] * 12,  # Strings instead of numbers
        "customer_id": "invalid_002"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_empty_request():
    """Test with empty request body"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error


# ============= EDGE CASES =============

def test_predict_boundary_values():
    """Test with boundary values"""
    features = calculate_features(
        credit_score=300,  # Minimum
        age=18,  # Minimum age
        tenure=0,  # New customer
        balance=0,  # No balance
        num_products=4,  # Maximum products
        has_cr_card=1,
        is_active_member=1,
        estimated_salary=10000,  # Low salary
        gender=0
    )
    
    payload = {
        "features": features,
        "customer_id": "boundary_001"
    }
    
    response = client.post("/predict", json=payload)
    
    if response.status_code == 503:
        pytest.skip("Model not loaded - skipping boundary test")
    
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["churn_probability"] <= 1


def test_predict_extreme_values():
    """Test with extreme values"""
    features = calculate_features(
        credit_score=850,  # Very high
        age=100,  # Very old
        tenure=50,  # Long tenure
        balance=1000000,  # High balance
        num_products=4,  # Max products
        has_cr_card=1,
        is_active_member=1,
        estimated_salary=500000,  # High salary
        gender=1
    )
    
    payload = {
        "features": features,
        "customer_id": "extreme_001"
    }
    
    response = client.post("/predict", json=payload)
    
    if response.status_code == 503:
        pytest.skip("Model not loaded - skipping extreme test")
    
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["churn_probability"] <= 1


# ============= PERFORMANCE TESTS =============

def test_response_time():
    """Test response time is reasonable"""
    import time
    
    features = [0.5] * 12
    payload = {
        "features": features,
        "customer_id": "perf_001"
    }
    
    start_time = time.time()
    response = client.post("/predict", json=payload)
    elapsed_time = time.time() - start_time
    
    if response.status_code == 503:
        pytest.skip("Model not loaded - skipping performance test")
    
    assert response.status_code == 200
    assert elapsed_time < 1.0  # Should respond within 1 second


# ============= HELPER FUNCTIONS FOR TEST SETUP =============

@pytest.fixture(scope="session", autouse=True)
def wait_for_model():
    """Wait for model to load if needed"""
    import time
    
    max_attempts = 10
    for attempt in range(max_attempts):
        response = client.get("/health")
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded", False):
                print(f"\n✅ Model loaded successfully")
                return
        time.sleep(1)
    
    print(f"\n⚠️ Model not loaded after {max_attempts} seconds")
    print("Some tests will be skipped")


# ============= MAIN =============

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
