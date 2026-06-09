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
import math
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import pandas as pd

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
    test_data_paths = 'data/test_data.csv'

    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\nRaw data loaded from {path}")
        print(f"  Shape: {df.shape}")
            
        # Preprocess each row using API function
        features_list = []
        for idx, row in df.iterrows():
            try:
                # Convert Gender to 0/1
                gender = 0 if row['Gender'] == 'Female' else 1
                    
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
                print(f"  Error preprocessing row {idx}: {e}")
                continue
            
        X = np.array(features_list)
        y = df['Exited'].values[:len(X)]  # Target is 'Exited'
            
        print(f"Preprocessed {len(X)} samples, {X.shape[1]} features")
        print(f"Target distribution: {y.mean():.2%} churn")
            
        return X, y, df
    
    # Generate synthetic data if no file exists
    print("\nNo test data found, generating synthetic data")
    return generate_synthetic_data()


def generate_synthetic_data(n_samples=500):
    """Generate synthetic preprocessed data"""
    np.random.seed(RANDOM_SEED)
    X = np.random.randn(n_samples, 12)
    y = (np.random.rand(n_samples) < 0.2).astype(int)
    # Add some correlation
    y = (y + 0.1 * X[:, 0] + 0.05 * X[:, 1] > 0.3).astype(int)
    return X, y, None


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


def bootstrap_roc_auc(model, X, y, n_bootstrap=100, random_seed=42):
    """Bootstrap ROC-AUC to get confidence intervals"""
    np.random.seed(random_seed)
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


#  MAIN MDD TEST

def test_mdd_hypothesis():
    """
    MDD Analysis: H0 vs H1 test
    H0: New model is not better than current (ROC-AUC <= 0.75)
    H1: New model is better (ROC-AUC > 0.75)
    """
    print("\nMDD ANALYSIS: Model Comparison\n")
    
    # Load and preprocess test data using API function
    X, y, _ = load_and_preprocess_test_data()
    
    print(f"\nTest data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.mean():.2%} churn")
    
    # Load models
    current_model = load_model('api/models/model.pkl')
    candidate_model = load_model('models/candidate_model.pkl')
    
    # If no candidate model, test will be skipped (not failed)
    if candidate_model is None:
        pytest.skip("No candidate model found for comparison")
    
    assert current_model is not None, "Current model not found"
    
    # Evaluate both models
    current_roc_auc = evaluate_model(current_model, X, y)
    candidate_roc_auc = evaluate_model(candidate_model, X, y)
    
    print(f"\nCurrent model ROC-AUC: {current_roc_auc:.4f}")
    print(f"Candidate model ROC-AUC: {candidate_roc_auc:.4f}")
    improvement = (candidate_roc_auc - current_roc_auc) * 100
    print(f"Improvement: {improvement:.2f}%")
    
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
    
    # Handle case when all scores are identical
    if np.std(current_cv_scores) == 0 and np.std(candidate_cv_scores) == 0:
        t_stat = 0.0
        p_value = 1.0
        print("\nAll CV scores are identical - cannot perform t-test")
    else:
        t_stat, p_value = stats.ttest_rel(candidate_cv_scores, current_cv_scores)
    
    print(f"\nPaired t-test results:")
    if not np.isnan(t_stat):
        print(f"   t-statistic: {t_stat:.4f}")
    if not np.isnan(p_value):
        print(f"   p-value: {p_value:.4f}")
    print(f"   Significance level: {SIGNIFICANCE_LEVEL}")
    
    # Decision logic
    is_better = candidate_roc_auc > ROC_AUC_THRESHOLD
    is_significant = p_value < SIGNIFICANCE_LEVEL if not np.isnan(p_value) else False
    has_improvement = candidate_roc_auc > current_roc_auc
    
    print(f"\nDecision Criteria:")
    print(f"   • Candidate > threshold ({ROC_AUC_THRESHOLD}): {is_better} ({candidate_roc_auc:.4f} > {ROC_AUC_THRESHOLD})")
    print(f"   • Statistically significant (p < {SIGNIFICANCE_LEVEL}): {is_significant}")
    print(f"   • Improvement over current: {has_improvement} ({candidate_roc_auc:.4f} > {current_roc_auc:.4f})")
    
    # Final decision
    if is_better and is_significant and has_improvement:
        print("\nDECISION: REJECT H0 - Deploy new model")
        decision = "DEPLOY"
        result = "REJECT_H0"
    else:
        print("\nDECISION: FAIL TO REJECT H0 - Keep current model")
        decision = "KEEP_CURRENT"
        result = "FAIL_TO_REJECT_H0"
    
    print(f"\nFinal Decision: {decision}")
    
    # Save decision for CI/CD
    save_decision(result, decision, {
        'current_roc_auc': current_roc_auc,
        'candidate_roc_auc': candidate_roc_auc,
        'p_value': p_value,
        'is_significant': is_significant,
        'has_improvement': has_improvement,
        'threshold': ROC_AUC_THRESHOLD
    })
    
    # Assert for CI/CD pipeline
    assert is_better and is_significant and has_improvement, \
        f"Model not good enough for deployment. ROC-AUC: {candidate_roc_auc:.4f}, p-value: {p_value:.4f}"


# ADDITIONAL TESTS

def test_bootstrap_confidence_intervals():
    """Test that confidence intervals are correctly calculated"""
    print("\nBOOTSTRAP CONFIDENCE INTERVALS\n")
    
    current_model = load_model('api/models/model.pkl')
    if current_model is None:
        pytest.skip("No current model found")
    
    X, y, _ = load_and_preprocess_test_data()
    
    scores = bootstrap_roc_auc(current_model, X, y)
    ci = np.percentile(scores, [2.5, 97.5])
    
    print(f"\nBootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Mean ROC-AUC: {np.mean(scores):.4f}")
    
    ci_width = ci[1] - ci[0]
    assert ci_width < 0.3, f"Confidence interval too wide: {ci_width:.4f}"
    
    print("\nConfidence intervals test passed")


def test_rollback_condition():
    """Test rollback conditions (informational)"""
    print("\nROLLBACK CONDITIONS TEST\n")
    
    script_path = 'scripts/rollback.sh'
    if os.path.exists(script_path):
        print(f"\n✓ Rollback script found: {script_path}")
    else:
        print(f"\nRollback script not found: {script_path}")
    
    print("\nRecommended rollback conditions:")
    conditions = [
        "ROC-AUC drops below 0.75",
        "p-value > 0.05 (not statistically significant)",
        "Inference latency increases > 20%",
        "Error rate > 1%"
    ]
    for condition in conditions:
        print(f"   • {condition}")
    
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
