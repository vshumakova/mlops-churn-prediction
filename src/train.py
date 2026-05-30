import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs('models', exist_ok=True)

np.random.seed(42)
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

joblib.dump(model, 'models/model.pkl')
print("Model saved")
print(f"Model accuracy: {model.score(X, y):.3f}")
