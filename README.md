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
- **Health checks** - `/health`, `/ready` endpoints
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

### API Endpoints
- /health	- GET	(Service health check)
- /ready - GET	(Readiness probe)
- /predict - POST	(Predict churn probability)
- /metrics - GET	(Model metrics)
- /docs - GET	(Swagger UI documentation)

### Predict Example

Request:
curl -X POST https://mlops-churn-prediction-2.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [720, 3.496, 5, 10.82, 2, 1, 1, 11.69, 0, 0.416, 0.1515, 21.82],
    "customer_id": "active_client"
  }'

### Project Structure
```bash
mlops-churn-prediction/
├── .github/workflows/
│   └── retrain.yml          # CI/CD
├── api/
│   ├── main.py              # API
│   └── requirements.txt     # API dependencies
├── data/raw/
│   └── bank_clients.csv     # Data
├── docs/
│   ├── adr/
│   │   └── 001-ml-pipeline.md  # ADR
│   └── sli-slo.md           # SLI/SLO
├── src/
│   ├── prepare_data.py      # Data preparation
│   └── train.py             # Trainig
├── tests/
│   └── test_api.py          # Tests
├── models/                  # Folder for models
├── metrics/                 # Folder for metrics
├── .gitignore               
├── MANIFEST.md              # Manifest
├── README.md                
├── requirements.txt         
└── runtime.txt              # Fixing Python 3.10
```

### CI/CD Pipeline
```bash
Schedule: Weekly (Sunday at 00:00 UTC)
Trigger: Manual or scheduled
Steps:
  1. Checkout code
  2. Setup Python 3.10
  3. Install dependencies
  4. Train model on real bank data
  5. Run API tests
  6. Deploy to Render (if tests pass)
```

### Feature Engineering
1. CreditScore - Credit rating (350-850)
2. Age_log - Logarithm of age
3. Tenure - Years with the bank
4. Balance_log - Logarithm of account balance
5. NumOfProducts - Number of bank products
6. HasCrCard - Credit card ownership (0/1)
7. IsActiveMember - Active customer status (0/1)
8. Salary_log - Logarithm of estimated salary
9. Gender - 0=Male, 1=Female
10. BalanceSalaryRatio - Balance / Salary
11. TenureByAge - Tenure / Age
12. CreditScoreGivenAge - CreditScore / Age

### Testing
```bash
# Run unit tests
pytest tests/test_api.py -v
```

### Live Demo

The API is deployed and available 24/7:
- Health Check: https://mlops-churn-prediction-2.onrender.com/health
- API Documentation: https://mlops-churn-prediction-2.onrender.com/docs
- Predict Endpoint: https://mlops-churn-prediction-2.onrender.com/predict

### Technologies
- Python 3.10 - Core language
- FastAPI - REST API framework
- Scikit-learn - ML algorithms
- Pandas/NumPy - Data processing
- Joblib - Model serialization
- Docker - Containerization
- GitHub Actions - CI/CD
- Render - Cloud hosting

### Links
- GitHub Repository: https://github.com/vshumakova/mlops-churn-prediction
- Live API: https://mlops-churn-prediction-2.onrender.com/health
- API Docs: https://mlops-churn-prediction-2.onrender.com/docs
