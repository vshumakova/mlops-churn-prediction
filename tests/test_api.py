import pytest
import os
import sys
import math
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

def load_model_for_tests():
    """Load model once for all tests"""
    import joblib
    import api.main
    
    model_path = 'api/models/model.pkl'
    if os.path.exists(model_path):
        api.main.model = joblib.load(model_path)
        return True
    return False

MODEL_LOADED = load_model_for_tests()
client = TestClient(app)


# helper
def features(credit_score=720, age=32, tenure=5, balance=50000,
             num_products=2, has_cr_card=1, is_active_member=1,
             estimated_salary=120000, gender=0):
    """Generate 12 features from raw data"""
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


# Main tests
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "predict" in response.json()


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "features_count" in response.json()


def test_predict():
    if not MODEL_LOADED:
        pytest.skip("Model not loaded")
    
    response = client.post("/predict", json={
        "features": features(),
        "customer_id": "test_001"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["churn_probability"] <= 1


def test_predict_invalid_features():
    response = client.post("/predict", json={
        "features": [0.5, 0.3],  # wrong count
        "customer_id": "test"
    })
    
    if MODEL_LOADED:
        assert response.status_code == 400
    else:
        assert response.status_code == 503


def test_predict_missing_customer_id():
    response = client.post("/predict", json={"features": [0.5] * 12})
    assert response.status_code == 422  # Pydantic validation


def test_ready():
    response = client.get("/ready")
    
    if MODEL_LOADED:
        assert response.status_code == 200
    else:
        assert response.status_code == 503


# CI/CD tests
def test_workflow_exists():
    workflow = '.github/workflows/retrain.yml'
    assert os.path.exists(workflow), "Workflow not found"
    
    with open(workflow) as f:
        content = f.read()
        assert 'schedule:' in content
        assert 'quality gate' in content


def test_training_script_exists():
    assert os.path.exists('src/train.py'), "train.py not found"


def test_retrain_endpoint():
    """Check if retrain endpoint is documented (optional feature)"""
    response = client.get("/")
    data = response.json()
    if "retrain" in data:
        assert isinstance(data["retrain"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
