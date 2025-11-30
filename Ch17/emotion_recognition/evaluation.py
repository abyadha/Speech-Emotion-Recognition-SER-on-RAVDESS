"""
Evaluation functions: metrics, statistical tests, model selection.
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
from typing import Dict, Tuple, Optional
import joblib


def select_best_model(all_results: Dict, criterion: str = 'test_f1') -> Tuple:
    """
    Select best model from all results.
    
    Args:
        all_results: Dictionary of all model results
        criterion: Selection criterion ('val_f1' or 'test_f1')
        
    Returns:
        best_model_name: Name of best model
        best_model_results: Results dictionary for best model
    """
    sorted_models = sorted(all_results.items(), 
                          key=lambda x: x[1][criterion], 
                          reverse=True)
    
    print("\n" + "="*70)
    print(f"TOP 5 MODELS (by {criterion.upper()}):")
    print("="*70)
    for i, (name, results) in enumerate(sorted_models[:5], 1):
        print(f"{i}. {name:30s} | Val F1: {results['val_f1']:.3f} | Test F1: {results['test_f1']:.3f}")
    
    best_model_name, best_model_results = sorted_models[0]
    
    print(f"\n" + "="*70)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print("="*70)
    print(f"Validation F1: {best_model_results['val_f1']:.3f}")
    print(f"Test F1:       {best_model_results['test_f1']:.3f}")
    
    return best_model_name, best_model_results


def load_svm_predictions(linear_path: str, rbf_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    Load SVM benchmark predictions.
    
    Returns:
        y_pred_linear: Linear SVM predictions or None
        y_pred_rbf: RBF SVM predictions or None
        svm_loaded: Whether predictions were loaded successfully
    """
    print("\n" + "="*70)
    print("LOADING SVM BENCHMARKS")
    print("="*70 + "\n")
    
    try:
        y_pred_linear = np.load(linear_path)
        y_pred_rbf = np.load(rbf_path)
        print("SVM benchmark predictions loaded successfully")
        print(f"  Linear SVM predictions: {len(y_pred_linear)} samples")
        print(f"  RBF SVM predictions: {len(y_pred_rbf)} samples")
        return y_pred_linear, y_pred_rbf, True
    except FileNotFoundError:
        print("SVM benchmark predictions not found")
        print("Statistical tests will be skipped")
        return None, None, False


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple:
    """
    Perform McNemar's test.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        
    Returns:
        statistic: Chi-square statistic
        p_value: P-value
    """
    # Create contingency table
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n_00 = np.sum(~correct1 & ~correct2)  # Both wrong
    n_01 = np.sum(~correct1 & correct2)   # Model 1 wrong, Model 2 correct
    n_10 = np.sum(correct1 & ~correct2)   # Model 1 correct, Model 2 wrong
    n_11 = np.sum(correct1 & correct2)    # Both correct
    
    table = np.array([[n_00, n_01], [n_10, n_11]])
    result = mcnemar(table, exact=False, correction=True)
    
    return result.statistic, result.pvalue


def paired_t_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray, 
                 classes: list) -> Tuple:
    """
    Perform paired t-test on per-class F1 scores.
    
    Returns:
        t_statistic: T-statistic
        p_value: P-value
        mean_diff: Mean difference in F1 scores
    """
    # Compute per-class F1 for each model
    f1_model1 = []
    f1_model2 = []
    
    for cls_idx in range(len(classes)):
        # Binary mask for this class
        y_true_binary = (y_true == cls_idx).astype(int)
        y_pred1_binary = (y_pred1 == cls_idx).astype(int)
        y_pred2_binary = (y_pred2 == cls_idx).astype(int)
        
        f1_1 = f1_score(y_true_binary, y_pred1_binary, average='binary', zero_division=0)
        f1_2 = f1_score(y_true_binary, y_pred2_binary, average='binary', zero_division=0)
        
        f1_model1.append(f1_1)
        f1_model2.append(f1_2)
    
    # Paired t-test
    differences = np.array(f1_model1) - np.array(f1_model2)
    t_stat, p_val = stats.ttest_rel(f1_model1, f1_model2)
    
    return t_stat, p_val, np.mean(differences)


def run_statistical_tests(y_test: np.ndarray, y_pred_lda: np.ndarray,
                          y_pred_linear: np.ndarray, y_pred_rbf: np.ndarray,
                          classes: list) -> Dict:
    """
    Run all statistical tests comparing LDA to SVM benchmarks.
    
    Returns:
        Dictionary with all test results
    """
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70 + "\n")
    
    results = {}
    
    # McNemar's Test
    print("McNemar's Test:")
    stat_lin, p_lin = mcnemar_test(y_test, y_pred_lda, y_pred_linear)
    stat_rbf, p_rbf = mcnemar_test(y_test, y_pred_lda, y_pred_rbf)
    
    print(f"  LDA vs Linear SVM: chi2={stat_lin:.3f}, p={p_lin:.4f}")
    print(f"  LDA vs RBF SVM:    chi2={stat_rbf:.3f}, p={p_rbf:.4f}")
    
    results['mcnemar'] = {
        'vs_linear_svm': {'chi_square': float(stat_lin), 'p_value': float(p_lin)},
        'vs_rbf_svm': {'chi_square': float(stat_rbf), 'p_value': float(p_rbf)}
    }
    
    # Paired t-test
    print("\nPaired t-test (per-class F1):")
    t_stat_lin, p_val_lin, diff_lin = paired_t_test(y_test, y_pred_lda, y_pred_linear, classes)
    t_stat_rbf, p_val_rbf, diff_rbf = paired_t_test(y_test, y_pred_lda, y_pred_rbf, classes)
    
    print(f"  LDA vs Linear SVM: t={t_stat_lin:.3f}, p={p_val_lin:.4f}, mean_diff={diff_lin:.3f}")
    print(f"  LDA vs RBF SVM:    t={t_stat_rbf:.3f}, p={p_val_rbf:.4f}, mean_diff={diff_rbf:.3f}")
    
    results['paired_t_test'] = {
        'vs_linear_svm': {'t_statistic': float(t_stat_lin), 'p_value': float(p_val_lin), 'mean_difference': float(diff_lin)},
        'vs_rbf_svm': {'t_statistic': float(t_stat_rbf), 'p_value': float(p_val_rbf), 'mean_difference': float(diff_rbf)}
    }
    
    return results


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> np.ndarray:
    """
    Compute F1 score for each class.
    
    Returns:
        Array of per-class F1 scores
    """
    f1_per_class = []
    
    for cls_idx in range(len(classes)):
        y_true_binary = (y_true == cls_idx).astype(int)
        y_pred_binary = (y_pred == cls_idx).astype(int)
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        f1_per_class.append(f1)
    
    return np.array(f1_per_class)


def save_best_model(model, filepath: str):
    """Save best model to file."""
    joblib.dump(model, filepath)
    print(f"\nBest model saved to: {filepath}")
