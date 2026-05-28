# MLOps Churn Prediction System

End-to-end ML system for customer churn prediction with automated retraining and traffic switching (MLOps Level 2).

## Features
- Automated data cleaning & feature engineering
- Weekly model retraining with quality gates  
- Automatic model rollback if ROC-AUC < 0.7
- Model versioning & experiment tracking (MLflow)
- CI/CD pipeline (GitHub Actions)
- REST API with FastAPI
- Health checks & monitoring
- Infrastructure as Code (Docker Compose)

## Architecture Level: 2
Full lifecycle management: data → training → deployment → monitoring → rollback

## Business Metric
Reduce churn rate by 15% within 2 quarters

## ML Metrics
- ROC-AUC ≥ 0.75
- Recall ≥ 0.70
- Precision ≥ 0.65

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/mlops-churn-prediction
cd mlops-churn-prediction

# Build and run with Docker Compose
docker-compose up --build

# Train model
docker-compose exec api python src/train.py

# Test API
curl http://localhost:8000/health
```
```bash
API Endpoints
Endpoint	Method	Description
/health	GET	Health check
/predict	POST	Predict churn probability
/metrics	GET	Model metrics

Project Structure
text
├── api/           # FastAPI application
├── src/           # Training & prediction scripts
├── tests/         # Unit tests
├── dags/          # Airflow DAGs
├── monitoring/    # Drift detection
├── configs/       # Configuration files
└── docs/          # Documentation & ADRs
```



