import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

os.makedirs('models', exist_ok=True)

np.random.seed(42)
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}
print(f"Metrics: {metrics}")

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "churn_model")
    mlflow.log_param("features", available_features)
    print("\nLogged to MLflow")

joblib.dump(model, 'models/model.pkl')
print("Model saved")

with open('metrics/latest_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to metrics/latest_metrics.json")
print("\nTraining completed successfully!")
