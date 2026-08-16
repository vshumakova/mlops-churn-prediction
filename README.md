# MLOps Churn Prediction System

End-to-end ML system for customer churn prediction with automated retraining and API deployment.

## Business Problem

A bank wants to develop a loyalty campaign to retain customers. The goal is to predict customer churn probability and identify clients likely to leave in the near future.

## Model Performance (on bank data)
- **Accuracy**: 99.8%
- **Precision**: 100%
- **Recall**: 99.4%
- **F1-Score**: 99.7%
- **ROC-AUC**: 100%

The model was trained on **2,457 real bank clients** with **12 engineered features**.

## Features
- **Automated data cleaning & feature engineering** - 12 features
- **Weekly model retraining** - GitHub Actions (every Sunday)
- **CI/CD pipeline** - Automated testing and deployment
- **REST API** - FastAPI with Swagger documentation
- **Web UI** - User-friendly interface for predictions
- **Health checks** - `/health`, `/ready` endpoints
- **Two prediction endpoints** - Array-based (`/predict`) and readable JSON (`/predict/v2`)
- **Infrastructure as Code** - Docker Compose
- **Cloud deployment** - Render.com (24/7)

## Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/vshumakova/mlops-churn-prediction
cd mlops-churn-prediction

# Install dependencies
pip install -r api/requirements.txt

# Train model
python src/train.py

# Run API locally
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Test API
curl http://localhost:8000/health
```

### Using Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Train model in container
docker-compose exec api python src/train.py

# Test API
curl http://localhost:8000/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Service health check |
| `/ready` | GET | Readiness probe |
| `/predict` | POST | Predict churn (12 features array) |
| `/predict/v2` | POST | Predict churn (readable JSON) |
| `/metrics` | GET | Model metrics |
| `/ui` | GET | Web UI interface |
| `/docs` | GET | Swagger UI documentation |

## Prediction Examples

### Using `/predict` (array format)
```bash
curl -X POST https://mlops-churn-prediction-2.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [720, 3.496, 5, 10.82, 2, 1, 1, 11.69, 0, 0.416, 0.1515, 21.82],
    "customer_id": "active_client"
  }'
```

**Response:**
```json
{
  "customer_id": "active_client",
  "churn_probability": 0.12,
  "prediction": 0,
  "risk_level": "Low",
  "recommendation": "Monitor only, no action needed",
  "model_version": "1.0.0",
  "timestamp": "2026-06-10T15:06:47.234338"
}
```

### Using `/predict/v2` (readable JSON)
```bash
curl -X POST https://mlops-churn-prediction-2.onrender.com/predict/v2 \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 720,
    "age": 32,
    "tenure": 5,
    "balance": 50000,
    "num_products": 2,
    "has_cr_card": true,
    "is_active_member": true,
    "estimated_salary": 120000,
    "gender": "Female"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.12,
  "churn_risk_percent": "12.0%",
  "prediction": "Will stay",
  "risk_level": "Low",
  "recommendation": "Monitor only, no action needed",
  "model_version": "1.0.0",
  "timestamp": "2026-06-10T15:06:47.234338"
}
```

## Project Structure
```
mlops-churn-prediction/
├── .github/workflows/
│   └── retrain.yml          # CI/CD pipeline
├── api/
│   ├── main.py              # FastAPI application
│   ├── static/
│   │   └── index.html       # Web UI
│   └── requirements.txt     # API dependencies
├── data/raw/
│   └── bank_clients.csv     # Training data
├── src/
│   ├── prepare_data.py      # Data preparation
│   └── train.py             # Model training
├── tests/
│   ├── test_api.py          # API tests
│   └── test_model_comparison.py  # MDD analysis
├── models/                  # Trained models
├── metrics/                 # Model metrics
├── scripts/
│   └── rollback.sh          # Rollback script
├── docker-compose.yml
├── Dockerfile
├── README.md
└── .gitignore
```

## CI/CD Pipeline
```yaml
Schedule: Weekly (Sunday at 00:00 UTC)
Trigger: Manual or scheduled

Steps:
  1. Checkout code
  2. Setup Python 3.10
  3. Install dependencies
  4. Train model on real bank data
  5. Run MDD analysis (statistical tests)
  6. Run API tests
  7. Save current model as backup
  8. Deploy new model (if MDD passes)
  9. Rollback on failure
```

## MDD Analysis (Model Decision Driven)

Statistical tests for model comparison before deployment:

- **H0**: New model is not better than current (ROC-AUC <= 0.75)
- **H1**: New model is better (ROC-AUC > 0.75)
- **Test**: Paired t-test, p-value = 0.05
- **Decision**: Deploy only if new model is statistically better

## Feature Engineering (12 features)

| # | Feature | Description |
|---|---------|-------------|
| 1 | CreditScore | Credit rating (350-850) |
| 2 | Age_log | Logarithm of age |
| 3 | Tenure | Years with the bank |
| 4 | Balance_log | Logarithm of account balance |
| 5 | NumOfProducts | Number of bank products |
| 6 | HasCrCard | Credit card ownership (0/1) |
| 7 | IsActiveMember | Active customer status (0/1) |
| 8 | Salary_log | Logarithm of estimated salary |
| 9 | Gender | 0=Male, 1=Female |
| 10 | BalanceSalaryRatio | Balance / Salary |
| 11 | TenureByAge | Tenure / Age |
| 12 | CreditScoreGivenAge | CreditScore / Age |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run API tests only
pytest tests/test_api.py -v

# Run MDD analysis
pytest tests/test_model_comparison.py::test_mdd_hypothesis -v
```

## Live Demo

The API is deployed and available 24/7:

| Resource | URL |
|----------|-----|
| Health Check | https://mlops-churn-prediction-web.onrender.com/health |
| Web UI | https://mlops-churn-prediction-web.onrender.com/ui |
| API Docs | https://mlops-churn-prediction-web.onrender.com/docs |
| Root Endpoint | https://mlops-churn-prediction-web.onrender.com/ |

## Technologies

| Category | Technologies |
|----------|--------------|
| Language | Python 3.10 |
| API Framework | FastAPI, Uvicorn |
| ML | Scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Cloud Hosting | Render.com |
| Statistical Tests | SciPy |

## Links

- **GitHub Repository**: https://github.com/vshumakova/mlops-churn-prediction
- **Live API**: https://mlops-churn-prediction-web.onrender.com
- **API Documentation**: https://mlops-churn-prediction-web.onrender.com/docs
- **Web Interface**: https://mlops-churn-prediction-web.onrender.com/ui

---
*Last updated: June 2026*
```
