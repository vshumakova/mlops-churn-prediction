import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
import os

def generate_sample_data(n_samples=1000):
    """Generate sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def train_model():
    """Train and save model"""
    print("=" * 50)
    print("Starting model training...")
    print("=" * 50)
    
    # Generate or load data
    X, y = generate_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("\nModel Performance:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    print("\n💾 Model saved to models/model.pkl")
    
    # Log to MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "churn_model")
    
    print("Logged to MLflow")
    return model, metrics

if __name__ == "__main__":
    train_model()
