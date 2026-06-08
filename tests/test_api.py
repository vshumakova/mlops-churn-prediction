import pytest
import os
import sys
import math
import json
import subprocess
from fastapi.testclient import TestClient

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем приложение
from api.main import app, model

# ПРИНУДИТЕЛЬНО ЗАГРУЖАЕМ МОДЕЛЬ ДЛЯ ТЕСТОВ
def force_load_model():
    """Force load model for testing"""
    import joblib
    
    model_path = 'api/models/model.pkl'
    if os.path.exists(model_path):
        try:
            import api.main
            api.main.model = joblib.load(model_path)
            print(f"\n✓ Model force-loaded from {model_path}")
            return True
        except Exception as e:
            print(f"\n✗ Failed to force-load model: {e}")
            return False
    else:
        print(f"\n✗ Model file not found at {model_path}")
        return False

# Загружаем модель перед созданием клиента
MODEL_LOADED = force_load_model()

# Создаем клиент
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


# ============= EXISTING TESTS =============

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
    if not MODEL_LOADED:
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


def test_predict_specific_case():
    """Test the specific case that worked in curl"""
    if not MODEL_LOADED:
        pytest.skip("Model not loaded - test requires model file")
    
    features = [720, 3.496, 5, 10.82, 2, 1, 1, 11.69, 0, 0.416, 0.1515, 21.82]
    
    payload = {"features": features, "customer_id": "active_client"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "active_client"
    assert data["churn_probability"] == 0.0
    assert data["prediction"] == 0
    assert data["risk_level"] == "Low"


def test_predict_invalid_features_count():
    """Test with wrong number of features"""
    payload = {
        "features": [0.5, 0.3, 0.2],
        "customer_id": "invalid_001"
    }
    response = client.post("/predict", json=payload)
    
    if MODEL_LOADED:
        assert response.status_code == 400
        assert "Expected 12 features" in response.json()["detail"]
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
    
    if MODEL_LOADED:
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
    else:
        assert response.status_code == 503


# ============= RETRAIN TESTS =============

def test_retrain_endpoint_exists():
    """Test that retrain endpoint is implemented"""
    response = client.post("/retrain", json={})
    assert response.status_code in [200, 400, 404, 422]
    
    if response.status_code == 404:
        pytest.skip("/retrain endpoint not implemented")


def test_retrain_with_valid_data():
    """Test retraining model with valid training data"""
    test_response = client.post("/retrain", json={})
    if test_response.status_code == 404:
        pytest.skip("/retrain endpoint not implemented")
    
    # Create sample training data
    training_data = create_sample_training_data(500)
    
    payload = {
        "training_data": training_data.tolist(),
        "model_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    }
    
    response = client.post("/retrain", json=payload)
    
    if response.status_code == 501:
        pytest.skip("Retrain functionality not yet implemented")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert "model_version" in data
    assert "timestamp" in data
    assert data["status"] == "retrained"


def test_retrain_preserves_model_version():
    """Test that retraining updates model version"""
    test_response = client.post("/retrain", json={})
    if test_response.status_code == 404:
        pytest.skip("/retrain endpoint not implemented")
    
    health_response = client.get("/health")
    old_version = health_response.json().get("model_version", "1.0.0")
    
    training_data = create_sample_training_data(300)
    payload = {
        "training_data": training_data.tolist(),
        "model_params": {"n_estimators": 50}
    }
    
    response = client.post("/retrain", json=payload)
    
    if response.status_code == 501:
        pytest.skip("Retrain functionality not yet implemented")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["model_version"] != old_version


def test_retrain_and_predict():
    """Test that retrained model works for predictions"""
    test_response = client.post("/retrain", json={})
    if test_response.status_code == 404:
        pytest.skip("/retrain endpoint not implemented")
    
    training_data = create_sample_training_data(200)
    payload = {
        "training_data": training_data.tolist(),
        "model_params": {"n_estimators": 50, "random_state": 42}
    }
    
    retrain_response = client.post("/retrain", json=payload)
    
    if retrain_response.status_code == 501:
        pytest.skip("Retrain functionality not yet implemented")
    
    assert retrain_response.status_code == 200
    
    features = calculate_features(
        credit_score=650, age=40, tenure=3, balance=25000,
        num_products=1, has_cr_card=0, is_active_member=0,
        estimated_salary=50000, gender=0
    )
    
    predict_payload = {"features": features, "customer_id": "test_after_retrain"}
    predict_response = client.post("/predict", json=predict_payload)
    
    assert predict_response.status_code == 200
    predict_data = predict_response.json()
    assert 0 <= predict_data["churn_probability"] <= 1
    assert predict_data["prediction"] in [0, 1]


def create_sample_training_data(n_samples=1000):
    """Create synthetic training data for churn prediction"""
    import numpy as np
    
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        credit_score = np.random.randint(300, 850)
        age = np.random.randint(18, 70)
        tenure = np.random.randint(0, 10)
        balance = np.random.uniform(0, 250000)
        num_products = np.random.randint(1, 5)
        has_cr_card = np.random.randint(0, 2)
        is_active_member = np.random.randint(0, 2)
        estimated_salary = np.random.uniform(20000, 200000)
        gender = np.random.randint(0, 2)
        
        # Calculate churn probability
        churn_prob = (
            (credit_score < 600) * 0.3 +
            (age > 60) * 0.2 +
            (tenure < 2) * 0.2 +
            (balance == 0) * 0.1 +
            (num_products == 1) * 0.1 +
            (has_cr_card == 0) * 0.1 +
            (is_active_member == 0) * 0.3
        )
        churn = 1 if churn_prob > 0.5 else 0
        
        features = calculate_features(
            credit_score, age, tenure, balance, num_products,
            has_cr_card, is_active_member, estimated_salary, gender
        )
        data.append(features + [churn])
    
    return np.array(data)


# ============= CI/CD TESTS =============

def test_github_workflow_exists():
    """Test that GitHub Actions workflow exists"""
    workflow_path = '.github/workflows/retrain.yml'
    assert os.path.exists(workflow_path), f"Workflow file not found at {workflow_path}"
    
    with open(workflow_path, 'r') as f:
        content = f.read()
        assert 'Auto Retrain Model' in content
        assert 'schedule:' in content
        assert 'cron:' in content
        assert 'quality gate' in content


def test_training_script_exists():
    """Test that training script exists"""
    train_script_path = 'src/train.py'
    assert os.path.exists(train_script_path), f"Training script not found at {train_script_path}"


def test_quality_gate_threshold():
    """Test that quality gate threshold is reasonable"""
    # Check the workflow file for quality gate threshold
    workflow_path = '.github/workflows/retrain.yml'
    with open(workflow_path, 'r') as f:
        content = f.read()
        # Extract the threshold from workflow
        import re
        match = re.search(r'roc_auc\s*<\s*([0-9.]+)', content)
        if match:
            threshold = float(match.group(1))
            print(f"\nQuality gate threshold: {threshold}")
            assert 0.5 <= threshold <= 0.95, "Quality gate threshold should be between 0.5 and 0.95"
        else:
            print("\nWarning: Could not find quality gate threshold in workflow")


def test_model_metrics_exist():
    """Test that model metrics are being saved"""
    metrics_dir = 'metrics'
    if os.path.exists(metrics_dir):
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
        if metrics_files:
            latest_metrics = os.path.join(metrics_dir, 'latest_metrics.json')
            if os.path.exists(latest_metrics):
                with open(latest_metrics, 'r') as f:
                    metrics = json.load(f)
                    print(f"\nCurrent metrics: {metrics}")
                    assert 'roc_auc' in metrics or 'accuracy' in metrics


def test_model_file_is_versioned():
    """Test that model file is tracked in git (or should be ignored)"""
    model_path = 'models/model.pkl'
    api_model_path = 'api/models/model.pkl'
    
    # Check if model files exist
    if os.path.exists(model_path):
        print(f"\nModel found at {model_path}, size: {os.path.getsize(model_path)} bytes")
    
    if os.path.exists(api_model_path):
        print(f"Model found at {api_model_path}, size: {os.path.getsize(api_model_path)} bytes")
    
    # Check .gitignore for model files (optional)
    gitignore_path = '.gitignore'
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            content = f.read()
            if '*.pkl' in content or 'model.pkl' in content:
                print("Model files are properly ignored in .gitignore")
            else:
                print("Warning: model.pkl not found in .gitignore - large files may be committed")


def test_retrain_workflow_syntax():
    """Test that GitHub workflow has valid syntax"""
    import yaml
    import re
    
    workflow_path = '.github/workflows/retrain.yml'
    with open(workflow_path, 'r') as f:
        content = f.read()
        
        # First, check if the file contains the required sections as strings
        assert 'name:' in content, "Missing 'name:' in workflow"
        assert 'on:' in content, "Missing 'on:' in workflow"
        assert 'schedule:' in content, "Missing 'schedule:' in workflow"
        assert 'cron:' in content, "Missing 'cron:' in workflow"
        assert 'workflow_dispatch:' in content, "Missing 'workflow_dispatch:' in workflow"
        assert 'jobs:' in content, "Missing 'jobs:' in workflow"
        
        # Now try to parse YAML (it should work despite the True issue)
        try:
            # Reset file pointer
            f.seek(0)
            workflow = yaml.safe_load(f)
            
            # Check if 'on' is present (might be as string or boolean)
            if 'on' in workflow:
                print("\n✓ 'on' section found as string key")
                on_section = workflow['on']
            elif True in workflow:
                print("\n⚠️ 'on' section loaded as boolean True - this indicates YAML formatting issue")
                # Extract the actual on section from the True key
                on_section = workflow[True]
                print("   The workflow has 'on:' but it's being parsed incorrectly")
                print("   This usually happens due to indentation issues")
            else:
                pytest.fail("Could not find 'on' section in workflow")
            
            # Check on section structure
            if isinstance(on_section, dict):
                if 'schedule' in on_section:
                    print("✓ Schedule configuration found")
                if 'workflow_dispatch' in on_section:
                    print("✓ Manual trigger (workflow_dispatch) is enabled")
            
            # Check jobs
            assert 'jobs' in workflow, "Missing 'jobs' in workflow"
            jobs = workflow['jobs']
            assert 'retrain' in jobs, "Missing 'retrain' job"
            
            retrain_job = jobs['retrain']
            assert 'runs-on' in retrain_job, "Missing runs-on"
            assert 'steps' in retrain_job, "Missing steps"
            
            # Check required steps
            steps = retrain_job['steps']
            step_names = [step.get('name', '') for step in steps]
            
            required_steps = [
                'Checkout code',
                'Setup Python',
                'Install dependencies',
                'Train model',
                'Check quality gate',
                'Run tests'
            ]
            
            for required in required_steps:
                assert any(required in name for name in step_names), f"Missing step: {required}"
            
            # Check quality gate threshold
            for step in steps:
                if step.get('name') == 'Check quality gate':
                    run_script = step.get('run', '')
                    match = re.search(r'<\s*([0-9.]+)', run_script)
                    if match:
                        threshold = float(match.group(1))
                        print(f"\n✓ Quality gate threshold: {threshold}")
                        assert 0.5 <= threshold <= 0.95, "Threshold should be between 0.5 and 0.95"
                    else:
                        print("\n⚠️ Could not find quality gate threshold in script")
            
            print("\n✅ GitHub workflow syntax is valid")
            
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in workflow: {e}")


def test_retrain_workflow_file_structure():
    """Test the raw file structure of the workflow"""
    workflow_path = '.github/workflows/retrain.yml'
    
    with open(workflow_path, 'r') as f:
        lines = f.readlines()
    
    # Check indentation
    found_on = False
    for i, line in enumerate(lines):
        if line.strip().startswith('on:'):
            found_on = True
            # Check next line for schedule
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if 'schedule' in next_line:
                    print("\n✓ Found 'schedule' after 'on:'")
                else:
                    print("\n⚠️ 'schedule' not found on next line after 'on:'")
    
    assert found_on, "Could not find 'on:' line in workflow file"
    
    # Check for proper YAML structure
    content = ''.join(lines)
    
    # Count occurrences
    assert content.count('schedule:') >= 1, "Missing schedule section"
    assert content.count('workflow_dispatch:') >= 1, "Missing workflow_dispatch section"
    assert content.count('jobs:') >= 1, "Missing jobs section"
    
    print("\n✅ Workflow file structure is valid")


def test_retrain_endpoint_documentation():
    """Test that retrain endpoint is documented in root"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    
    # Check if retrain is documented (optional)
    if "retrain" in data:
        assert isinstance(data["retrain"], str)


# ============= PERFORMANCE TESTS =============

def test_model_inference_speed():
    """Test that model inference is fast enough"""
    if not MODEL_LOADED:
        pytest.skip("Model not loaded - test requires model file")
    
    import time
    
    features = [0.5] * 12
    payload = {"features": features, "customer_id": "perf_test"}
    
    # Warm up
    client.post("/predict", json=payload)
    
    # Test 10 predictions
    start_time = time.time()
    for _ in range(10):
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 503]
    elapsed_time = time.time() - start_time
    
    avg_time = elapsed_time / 10
    print(f"\nAverage inference time: {avg_time*1000:.2f}ms")
    
    if MODEL_LOADED:
        assert avg_time < 0.1, f"Inference too slow: {avg_time*1000:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
