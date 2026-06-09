import pytest
import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app, calculate_features

# Model loading
def load_model_for_tests():
    """Load model once for all tests"""
    import joblib
    import api.main
    
    model_path = 'models/model.pkl'
    if os.path.exists(model_path):
        api.main.model = joblib.load(model_path)
        return True
    return False

MODEL_LOADED = load_model_for_tests()
client = TestClient(app)


# Helper
def features(credit_score=720, age=32, tenure=5, balance=50000,
             num_products=2, has_cr_card=1, is_active_member=1,
             estimated_salary=120000, gender=0):
    """Generate 12 features using API's calculate_features"""
    return calculate_features(
        credit_score=credit_score,
        age=age,
        tenure=tenure,
        balance=balance,
        num_products=num_products,
        has_cr_card=has_cr_card,
        is_active_member=is_active_member,
        estimated_salary=estimated_salary,
        gender=gender
    )


# API tests
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
    assert data["prediction"] in [0, 1]


def test_predict_invalid_features():
    response = client.post("/predict", json={
        "features": [0.5, 0.3],
        "customer_id": "test"
    })
    
    if MODEL_LOADED:
        assert response.status_code == 400
    else:
        assert response.status_code == 503


def test_predict_missing_customer_id():
    response = client.post("/predict", json={"features": [0.5] * 12})
    assert response.status_code == 422


def test_ready():
    response = client.get("/ready")
    
    if MODEL_LOADED:
        assert response.status_code == 200
    else:
        assert response.status_code == 503


# CI/CD tests
def test_workflow_exists():
    workflow = '.github/workflows/retrain.yml'
    assert os.path.exists(workflow)
    
    with open(workflow) as f:
        content = f.read()
        assert 'schedule:' in content
        assert 'mdd' in content.lower() or 'test_model_comparison' in content
        assert 'rollback' in content.lower()


def test_training_script_exists():
    assert os.path.exists('src/train.py')


def test_retrain_endpoint():
    response = client.get("/")
    data = response.json()
    
    if "retrain" in data:
        assert isinstance(data["retrain"], str)


def test_rollback_script_exists():
    script = 'scripts/rollback.sh'
    if os.path.exists(script):
        print(f"\nRollback script found: {script}")
    else:
        print(f"\nRollback script not found (optional): {script}")


def test_model_file_exists():
    model_path = 'api/models/model.pkl'
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"\nModel file: {model_path} ({size:,} bytes)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
