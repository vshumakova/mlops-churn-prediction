import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import json
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_data import prepare_data

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# Data preparing
try:
    X, y = prepare_data()
    print(f"\nData prepared successfully")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
except Exception as e:
    print(f"Error loading data: {e}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
    'timestamp': datetime.now().isoformat()
}
print(f"Metrics: {metrics}")

# with mlflow.start_run():
#     mlflow.log_params(model.get_params())
#     mlflow.log_metrics(metrics)
#     mlflow.sklearn.log_model(model, "churn_model")
#     mlflow.log_param("features", available_features)
#     print("\nLogged to MLflow")

# Saving model
joblib.dump(model, 'models/model.pkl')
print("Model saved")

# Saving metrics
with open('metrics/latest_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to metrics/latest_metrics.json")

# Saving metrics in MARKDOWN for GitHub
with open('METRICS.md', 'w') as f:
    f.write("# Model Performance Metrics\n\n")
    f.write("| Metric | Value |\n")
    f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
    f.write(f"| Precision | {metrics['precision']:.4f} |\n")
    f.write(f"| Recall | {metrics['recall']:.4f} |\n")
    f.write(f"| F1-Score | {metrics['f1']:.4f} |\n")
    f.write(f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n\n")
    f.write(f"_Last updated: {metrics['timestamp']}_\n")

print("\nTraining completed successfully")
