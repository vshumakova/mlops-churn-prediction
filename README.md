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

Endpoint	Method	Description
/health	GET	Service health check
/ready	GET	Readiness probe
/predict	POST	Predict churn probability
/metrics	GET	Model metrics
/docs	GET	Swagger UI documentation

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
├── api/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # API dependencies
│   └── models/              # Trained models (.pkl)
│       └── model.pkl
├── src/
│   ├── __init__.py 
│   ├── train.py             # Model training script
│   └── prepare_data.py      # Data preparation
├── tests/
│   └── test_api.py          # API unit tests
├── data/
│   └── raw/                 # Raw data files
│       └── bank_clients.csv
├── metrics/                 # Performance metrics (JSON)
├── docs/
│   └── adr/                 # Architecture Decision Records
├── .github/workflows/
│   └── retrain.yml          # Auto-retrain CI/CD
├── docker-compose.yml       # Infrastructure as Code
├── MANIFEST.md              # ML System Manifest
├── METRICS.md               # Model performance dashboard
└── README.md                # This file
```

