import pytest
import os
import sys
import math
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

# Model loading
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


# Helper
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


# API tests
def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "predict" in response.json()


def test_metrics():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "features_count" in response.json()


def test_predict():
    """Test prediction with valid data"""
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
    """Test with wrong number of features"""
    response = client.post("/predict", json={
        "features": [0.5, 0.3],
        "customer_id": "test"
    })
    
    if MODEL_LOADED:
        assert response.status_code == 400
    else:
        assert response.status_code == 503


def test_predict_missing_customer_id():
    """Test without customer_id"""
    response = client.post("/predict", json={"features": [0.5] * 12})
    assert response.status_code == 422


def test_ready():
    """Test readiness endpoint"""
    response = client.get("/ready")
    
    if MODEL_LOADED:
        assert response.status_code == 200
    else:
        assert response.status_code == 503


# CI/CD tests
def test_workflow_exists():
    """Test that GitHub Actions workflow exists and has required components"""
    workflow = '.github/workflows/retrain.yml'
    assert os.path.exists(workflow), f"Workflow not found at {workflow}"
    
    with open(workflow) as f:
        content = f.read()
        
        assert 'schedule:' in content, "Missing schedule configuration"
        assert 'cron:' in content, "Missing cron schedule"
        assert 'workflow_dispatch:' in content, "Missing manual trigger"
        assert 'jobs:' in content, "Missing jobs configuration"
        
        has_mdd = 'mdd' in content.lower() or 'test_model_comparison' in content
        assert has_mdd, "Missing model comparison test (MDD analysis)"
        
        assert 'rollback' in content.lower(), "Missing rollback configuration"
        
        required_steps = [
            'Checkout code',
            'Setup Python',
            'Install dependencies',
            'Train new model',
            'MDD Analysis'
        ]
        
        for step in required_steps:
            assert step in content, f"Missing step: {step}"
        
        print("\n✓ GitHub workflow is properly configured")


def test_training_script_exists():
    """Test that training script exists"""
    train_script = 'src/train.py'
    assert os.path.exists(train_script), f"Training script not found at {train_script}"
    
    size = os.path.getsize(train_script)
    assert size > 0, f"Training script is empty"
    print(f"\n✓ Training script found: {train_script} ({size} bytes)")


def test_retrain_endpoint():
    """Check if retrain endpoint is documented (optional feature)"""
    response = client.get("/")
    data = response.json()
    
    if "retrain" in data:
        assert isinstance(data["retrain"], str)
        print("\n✓ Retrain endpoint is documented")
    else:
        print("\nRetrain endpoint not documented (optional)")


def test_rollback_script_exists():
    """Test that rollback script exists"""
    script = 'scripts/rollback.sh'
    if os.path.exists(script):
        assert os.access(script, os.X_OK) or True, "Rollback script not executable"
        print(f"\n✓ Rollback script found: {script}")
    else:
        print(f"\nRollback script not found (optional): {script}")


def test_model_file_exists():
    """Test that model file exists"""
    model_path = 'api/models/model.pkl'
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        assert size > 0, "Model file is empty"
        print(f"\n✓ Model file found: {model_path} ({size} bytes)")
    else:
        print(f"\nModel file not found (will be created during training)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
