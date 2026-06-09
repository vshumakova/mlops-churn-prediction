"""
MDD Analysis - Model Decision Driven tests
H0: New model is not better than current (ROC-AUC <= 0.75)
H1: New model is better (ROC-AUC > 0.75)
"""

import pytest
import numpy as np
import joblib
import os
import json
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import calculate_features

ROC_AUC_THRESHOLD = 0.75
SIGNIFICANCE_LEVEL = 0.05
CV_FOLDS = 5
RANDOM_SEED = 42


def load_and_preprocess_test_data():
    """
    Load raw CSV and preprocess using API's calculate_features
    """
    test_data_path = 'data/test_data.csv'
    
    if os.path.exists(test_data_path):
        df = pd.read_csv(test_data_path)
        print(f"\nRaw data loaded from {test_data_path}")
        print(f"Shape: {df.shape}")
            
        # Preprocess each row using API function
        features_list = []
        for idx, row in df.iterrows():
            try:
                # Convert Gender: Male -> 0, Female -> 1
                gender = 0 if row['Gender'] == 'Male' else 1
                    
                features = calculate_features(
                        credit_score=row['CreditScore'],
                        age=row['Age'],
                        tenure=row['Tenure'],
                        balance=row['Balance'],
                        num_products=row['NumOfProducts'],
                        has_cr_card=row['HasCrCard'],
                        is_active_member=row['IsActiveMember'],
                        estimated_salary=row['EstimatedSalary'],
                        gender=gender
                )
                features_list.append(features)
            except Exception as e:
                print(f"Error preprocessing row {idx}: {e}")
                continue
            
        X = np.array(features_list)
        y = df['Exited'].values[:len(X)]
                
        print(f"Preprocessed {len(X)} samples, {X.shape[1]} features")
        print(f"Target distribution: {y.mean():.2%} churn")
                
        return X, y
    
    # Generate synthetic data if no file exists
    print("\nNo test data found, generating synthetic data")
    np.random.seed(RANDOM_SEED)
    X = np.random.randn(500, 12)
    y = (np.random.rand(500) < 0.2).astype(int)
    return X, y


def load_model(model_path):
    """Load model from path"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def evaluate_model(model, X, y):
    """Evaluate model and return ROC-AUC"""
    try:
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
        return roc_auc_score(y, y_pred)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.5


def bootstrap_roc_auc(model, X, y, n_bootstrap=100):
    """Bootstrap ROC-AUC to get confidence intervals"""
    np.random.seed(RANDOM_SEED)
    scores = []
    n = len(y)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        score = evaluate_model(model, X_boot, y_boot)
        scores.append(score)
    
    return np.array(scores)


def save_decision(result, decision, metrics):
    """Save MDD decision to file"""
    from datetime import datetime
    
    output = {
        'test': 'MDD_Hypothesis_Test',
        'result': result,
        'decision': decision,
        'metrics': {
            'current_roc_auc': float(metrics['current_roc_auc']),
            'candidate_roc_auc': float(metrics['candidate_roc_auc']),
            'p_value': float(metrics['p_value']) if not np.isnan(metrics['p_value']) else None,
            'is_significant': bool(metrics['is_significant']),
            'has_improvement': bool(metrics['has_improvement']),
            'threshold': float(metrics['threshold'])
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('mdd_decision.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDecision saved to mdd_decision.json")


# MAIN MDD TEST

def test_mdd_hypothesis():
    """
    MDD Analysis: H0 vs H1 test
    H0: New model is not better than current (ROC-AUC <= 0.75)
    H1: New model is better (ROC-AUC > 0.75)
    """
    print("\nMDD ANALYSIS: Model Comparison\n")
    
    # Load and preprocess test data
    X, y = load_and_preprocess_test_data()
    
    print(f"\nTest data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.mean():.2%} churn")
    
    # Load models
    current_model = load_model('models/model.pkl')
    candidate_model = load_model('models/candidate_model.pkl')
    
    if candidate_model is None:
        pytest.skip("No candidate model found for comparison")
    
    assert current_model is not None, "Current model not found"
    
    # Evaluate both models
    current_roc_auc = evaluate_model(current_model, X, y)
    candidate_roc_auc = evaluate_model(candidate_model, X, y)
    
    print(f"\nCurrent model ROC-AUC: {current_roc_auc:.4f}")
    print(f"Candidate model ROC-AUC: {candidate_roc_auc:.4f}")
    print(f"Improvement: {(candidate_roc_auc - current_roc_auc)*100:.2f}%")
    
    # Bootstrap for confidence intervals
    current_scores = bootstrap_roc_auc(current_model, X, y)
    candidate_scores = bootstrap_roc_auc(candidate_model, X, y)
    
    current_ci = np.percentile(current_scores, [2.5, 97.5])
    candidate_ci = np.percentile(candidate_scores, [2.5, 97.5])
    
    print(f"\nCurrent model 95% CI: [{current_ci[0]:.4f}, {current_ci[1]:.4f}]")
    print(f"Candidate model 95% CI: [{candidate_ci[0]:.4f}, {candidate_ci[1]:.4f}]")
    
    # Paired t-test
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    current_cv_scores = []
    candidate_cv_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        current_cv_scores.append(evaluate_model(current_model, X_test, y_test))
        candidate_cv_scores.append(evaluate_model(candidate_model, X_test, y_test))
    
    if np.std(current_cv_scores) == 0 and np.std(candidate_cv_scores) == 0:
        p_value = 1.0
        print("\nAll CV scores are identical")
    else:
        _, p_value = stats.ttest_rel(candidate_cv_scores, current_cv_scores)
    
    print(f"\nPaired t-test p-value: {p_value:.4f}")
    
    # Decision logic
    is_better = candidate_roc_auc > ROC_AUC_THRESHOLD
    is_significant = p_value < SIGNIFICANCE_LEVEL
    has_improvement = candidate_roc_auc > current_roc_auc
    
    print(f"\nDecision Criteria:")
    print(f"   • Candidate > {ROC_AUC_THRESHOLD}: {is_better} ({candidate_roc_auc:.4f})")
    print(f"   • p-value < {SIGNIFICANCE_LEVEL}: {is_significant} ({p_value:.4f})")
    print(f"   • Improvement over current: {has_improvement}")
    
    if is_better and is_significant and has_improvement:
        print("\nDECISION: REJECT H0 - Deploy new model")
        decision = "DEPLOY"
        result = "REJECT_H0"
    else:
        print("\nDECISION: FAIL TO REJECT H0 - Keep current model")
        decision = "KEEP_CURRENT"
        result = "FAIL_TO_REJECT_H0"
    
    save_decision(result, decision, {
        'current_roc_auc': current_roc_auc,
        'candidate_roc_auc': candidate_roc_auc,
        'p_value': p_value,
        'is_significant': is_significant,
        'has_improvement': has_improvement,
        'threshold': ROC_AUC_THRESHOLD
    })
    
    assert is_better and is_significant and has_improvement, \
        f"Model not good enough for deployment"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
