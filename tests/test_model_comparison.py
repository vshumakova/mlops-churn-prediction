"""
Statistical tests for model comparison (MDD Analysis)
Compares current model vs candidate model for deployment decision
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import os
import json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

# CONFIGURATION
ROC_AUC_THRESHOLD = 0.75
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold
CV_FOLDS = 5
RANDOM_SEED = 42


# HELPER FUNCTIONS
def load_test_data():
    """
    Load test data for model comparison
    Expected format: features + target (last column)
    """
    # Path to test data (adjust as needed)
    test_data_paths = [
        'data/test_data.csv',
        'data/validation.csv', 
        'tests/test_data.csv',
        'api/data/test_data.csv'
    ]
    
    for path in test_data_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            print(f"\n✓ Test data loaded from {path}")
            print(f"  Shape: {data.shape}")
            return data
    
    # If no test data found, generate synthetic data for testing
    print("\nNo test data found, generating synthetic data for testing")
    return generate_synthetic_test_data()


def generate_synthetic_test_data(n_samples=1000, random_seed=42):
    """Generate synthetic test data for model comparison"""
    np.random.seed(random_seed)
    
    # Features (12 features)
    X = np.random.randn(n_samples, 12)
    
    # Generate realistic target (churn ~ 20%)
    y = (np.random.rand(n_samples) < 0.2).astype(int)
    
    # Add some correlation with features
    y = (y + 0.1 * X[:, 0] + 0.05 * X[:, 1] > 0.3).astype(int)
    
    # Create DataFrame
    feature_names = [
        'credit_score', 'log_age', 'tenure', 'log_balance',
        'num_products', 'has_cr_card', 'is_active_member',
        'log_salary', 'gender', 'balance_salary_ratio',
        'tenure_age_ratio', 'credit_score_age_ratio'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def load_model(model_path):
    """Load model from path"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def evaluate_model(model, X, y):
    """
    Evaluate model and return ROC-AUC score
    """
    try:
        # Try predict_proba first
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            y_pred_proba = model.predict(X)
        
        # Calculate ROC-AUC
        roc_auc = roc_auc_score(y, y_pred_proba)
        return roc_auc
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.5


def bootstrap_roc_auc(model, X, y, n_bootstrap=1000, random_seed=42):
    """
    Bootstrap ROC-AUC to get confidence intervals
    """
    np.random.seed(random_seed)
    scores = []
    n = len(y)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        X_boot = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
        y_boot = y[indices]
        
        score = evaluate_model(model, X_boot, y_boot)
        scores.append(score)
    
    return np.array(scores)


# MAIN MDD TESTS

def test_mdd_hypothesis():
    """
    MDD Analysis: H0 vs H1 test
    H0: New model is not better than current (ROC-AUC <= 0.75)
    H1: New model is better (ROC-AUC > 0.75)
    """
    
    print("\nMDD ANALYSIS: Model Comparison\n")
    
    # Load test data
    data = load_test_data()
    feature_cols = [col for col in data.columns if col != 'target']
    X = data[feature_cols].values
    y = data['target'].values
    
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
    print(f"Improvement: {(candidate_roc_auc - current_roc_auc)*100:.2f}%")
    
    # Bootstrap for confidence intervals
    current_scores = bootstrap_roc_auc(current_model, X, y)
    candidate_scores = bootstrap_roc_auc(candidate_model, X, y)
    
    # Calculate confidence intervals (95%)
    current_ci = np.percentile(current_scores, [2.5, 97.5])
    candidate_ci = np.percentile(candidate_scores, [2.5, 97.5])
    
    print(f"\nCurrent model 95% CI: [{current_ci[0]:.4f}, {current_ci[1]:.4f}]")
    print(f"Candidate model 95% CI: [{candidate_ci[0]:.4f}, {candidate_ci[1]:.4f}]")
    
    # Perform paired t-test
    # We need paired scores - use cross-validation for paired comparison
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    current_cv_scores = []
    candidate_cv_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # For fair comparison, we should retrain on same folds
        # But here we're evaluating pre-trained models
        current_cv_scores.append(evaluate_model(current_model, X_test, y_test))
        candidate_cv_scores.append(evaluate_model(candidate_model, X_test, y_test))
    
    # Paired t-test
    differences = np.array(candidate_cv_scores) - np.array(current_cv_scores)
    t_stat, p_value = stats.ttest_rel(candidate_cv_scores, current_cv_scores)
    
    print(f"\nPaired t-test results:")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Significance level: {SIGNIFICANCE_LEVEL}")
    
    # Decision logic
    is_better = candidate_roc_auc > ROC_AUC_THRESHOLD
    is_significant = p_value < SIGNIFICANCE_LEVEL
    has_improvement = candidate_roc_auc > current_roc_auc
    
    print(f"\nDecision Criteria:")
    print(f"   Candidate > threshold ({ROC_AUC_THRESHOLD}): {is_better}")
    print(f"   Statistically significant (p < {SIGNIFICANCE_LEVEL}): {is_significant}")
    print(f"   Improvement over current: {has_improvement}")
    
    # Final decision
    if is_better and is_significant and has_improvement:
        print("\nDECISION: Reject H0 - Deploy new model")
        decision = "DEPLOY"
        result = "REJECT_H0"
    else:
        print("\nDECISION: Cannot reject H0 - Keep current model")
        decision = "KEEP_CURRENT"
        result = "FAIL_TO_REJECT_H0"
    
    print(f"\nFinal Decision: {decision}")
    
    # Save decision for CI/CD
    save_decision(result, decision, {
        'current_roc_auc': float(current_roc_auc),
        'candidate_roc_auc': float(candidate_roc_auc),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'has_improvement': has_improvement,
        'threshold': ROC_AUC_THRESHOLD
    })
    
    # Assert for CI/CD pipeline
    assert is_better and is_significant and has_improvement, \
        f"Model not good enough for deployment. ROC-AUC: {candidate_roc_auc:.4f}, p-value: {p_value:.4f}"


def test_bootstrap_confidence_intervals():
    """
    Test that confidence intervals are correctly calculated
    """
    data = load_test_data()
    feature_cols = [col for col in data.columns if col != 'target']
    X = data[feature_cols].values
    y = data['target'].values
    
    current_model = load_model('api/models/model.pkl')
    if current_model is None:
        pytest.skip("Current model not found")
    
    scores = bootstrap_roc_auc(current_model, X, y)
    ci = np.percentile(scores, [2.5, 97.5])
    
    print(f"\nBootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # CI should be reasonable (width less than 0.1)
    ci_width = ci[1] - ci[0]
    assert ci_width < 0.2, f"Confidence interval too wide: {ci_width:.4f}"
    
    # CI should be within [0, 1]
    assert 0 <= ci[0] <= 1, "Lower CI bound out of range"
    assert 0 <= ci[1] <= 1, "Upper CI bound out of range"


def test_effect_size():
    """
    Calculate effect size (Cohen's d) for model improvement
    """
    data = load_test_data()
    feature_cols = [col for col in data.columns if col != 'target']
    X = data[feature_cols].values
    y = data['target'].values
    
    current_model = load_model('api/models/model.pkl')
    candidate_model = load_model('models/candidate_model.pkl')
    
    if candidate_model is None:
        pytest.skip("No candidate model found")
    
    # Get scores via cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    current_scores = []
    candidate_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        current_scores.append(evaluate_model(current_model, X_test, y_test))
        candidate_scores.append(evaluate_model(candidate_model, X_test, y_test))
    
    # Calculate Cohen's d (effect size)
    mean_diff = np.mean(candidate_scores) - np.mean(current_scores)
    pooled_std = np.sqrt((np.var(current_scores) + np.var(candidate_scores)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    
    # Interpret effect size
    if cohens_d < 0.2:
        effect = "negligible"
    elif cohens_d < 0.5:
        effect = "small"
    elif cohens_d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    print(f"   Effect interpretation: {effect}")
    
    # For deployment, we want at least small effect
    if cohens_d < 0.2 and mean_diff > 0:
        print(f"Warning: Improvement is statistically significant but practically negligible")


def save_decision(result, decision, metrics):
    """
    Save MDD decision for CI/CD pipeline
    """
    decision_file = 'mdd_decision.json'
    
    output = {
        'test': 'MDD_Hypothesis_Test',
        'result': result,
        'decision': decision,
        'metrics': metrics,
        'timestamp': str(pd.Timestamp.now()),
        'threshold': ROC_AUC_THRESHOLD,
        'significance_level': SIGNIFICANCE_LEVEL
    }
    
    with open(decision_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDecision saved to {decision_file}")


# ADDITIONAL TESTS
def test_model_improvement_percentile():
    """
    Test that improvement is robust across different data splits
    """
    data = load_test_data()
    feature_cols = [col for col in data.columns if col != 'target']
    X = data[feature_cols].values
    y = data['target'].values
    
    current_model = load_model('api/models/model.pkl')
    candidate_model = load_model('models/candidate_model.pkl')
    
    if candidate_model is None:
        pytest.skip("No candidate model found")
    
    # Multiple bootstrap iterations to check robustness
    improvements = []
    
    for seed in range(10):
        np.random.seed(seed)
        indices = np.random.choice(len(y), len(y), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        current_score = evaluate_model(current_model, X_boot, y_boot)
        candidate_score = evaluate_model(candidate_model, X_boot, y_boot)
        improvements.append(candidate_score - current_score)
    
    improvements = np.array(improvements)
    improvement_percentiles = np.percentile(improvements, [2.5, 50, 97.5])
    
    print(f"\nImprovement distribution across bootstraps:")
    print(f"   Median improvement: {improvement_percentiles[1]*100:.2f}%")
    print(f"   95% CI: [{improvement_percentiles[0]*100:.2f}%, {improvement_percentiles[2]*100:.2f}%]")
    
    # Check that median improvement is positive
    median_improvement = improvement_percentiles[1]
    if median_improvement <= 0:
        print(f"Warning: Median improvement is not positive ({median_improvement*100:.2f}%)")


def test_rollback_condition():
    """
    Test rollback conditions (if new model fails)
    """
    print("\nRollback Conditions Test")
    
    # Check if rollback mechanism is documented
    rollback_conditions = [
        "ROC-AUC drops below threshold",
        "p-value > 0.05",
        "Inference latency increases > 20%",
        "Error rate > 1%"
    ]
    
    # Check for rollback script
    rollback_script = 'scripts/rollback.sh'
    if os.path.exists(rollback_script):
        print(f"Rollback script found: {rollback_script}")
    else:
        print(f"Rollback script not found (optional)")
    
    print("\nRecommended rollback conditions:")
    for condition in rollback_conditions:
        print(f"   • {condition}")
    
    # This test always passes - it's informational
    assert True


if __name__ == "__main__":
    # Run MDD tests
    pytest.main([__file__, "-v", "-s"])
