"""
Model training functions for emotion recognition.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple


def train_baseline_lda(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Train baseline LDA model.
    
    Returns:
        Dictionary with model, predictions, and scores
    """
    print("\n" + "="*70)
    print("TRAINING BASELINE LDA")
    print("="*70 + "\n")
    
    baseline_lda = LinearDiscriminantAnalysis()
    baseline_lda.fit(X_train, y_train)
    
    y_val_pred = baseline_lda.predict(X_val)
    y_test_pred = baseline_lda.predict(X_test)
    
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print(f"Baseline LDA:")
    print(f"  Validation F1: {val_f1:.3f}")
    print(f"  Test F1:       {test_f1:.3f}")
    
    return {
        'model': baseline_lda,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred
    }


def train_selectkbest_lda(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          k_values: List[int],
                          selection_criterion: str = 'val_f1') -> Tuple[Dict, Dict]:
    """
    Train LDA with SelectKBest feature selection.
    
    Args:
        selection_criterion: 'val_f1' or 'test_f1' (should always be 'val_f1')
    
    Returns:
        results_kbest: Dictionary of all results
        best_config: Best configuration based on selection criterion
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION - SelectKBest + LDA")
    print("="*70 + "\n")
    
    results_kbest = {}
    
    for k in k_values:
        print(f"Testing k={k}...")
        
        pipeline = Pipeline([
            ('selector', SelectKBest(f_classif, k=k)),
            ('lda', LinearDiscriminantAnalysis())
        ])
        
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        
        results_kbest[f'SelectKBest_k{k}'] = {
            'model': pipeline,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'y_test_pred': y_test_pred,
            'k': k
        }
        
        print(f"  k={k}: Val F1 = {val_f1:.3f}, Test F1 = {test_f1:.3f}")
    
    # Select best based on criterion
    best_config = max(results_kbest.items(), key=lambda x: x[1][selection_criterion])
    print(f"\nBest SelectKBest configuration (by {selection_criterion}):")
    print(f"  {best_config[0]}: Val F1 = {best_config[1]['val_f1']:.3f}, Test F1 = {best_config[1]['test_f1']:.3f}")
    
    return results_kbest, best_config


def train_pca_lda(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  n_components_list: List[int],
                  seed: int,
                  selection_criterion: str = 'val_f1') -> Tuple[Dict, Dict]:
    """
    Train LDA with PCA dimensionality reduction.
    
    Returns:
        results_pca: Dictionary of all results
        best_config: Best configuration based on selection criterion
    """
    print("\n" + "="*70)
    print("DIMENSIONALITY REDUCTION - PCA + LDA")
    print("="*70 + "\n")
    
    results_pca = {}
    
    for n in n_components_list:
        print(f"Testing n_components={n}...")
        
        pipeline = Pipeline([
            ('pca', PCA(n_components=n, random_state=seed)),
            ('lda', LinearDiscriminantAnalysis())
        ])
        
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        
        results_pca[f'PCA_n{n}'] = {
            'model': pipeline,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'y_test_pred': y_test_pred,
            'n': n
        }
        
        print(f"  n={n}: Val F1 = {val_f1:.3f}, Test F1 = {test_f1:.3f}")
    
    # Select best
    best_config = max(results_pca.items(), key=lambda x: x[1][selection_criterion])
    print(f"\nBest PCA configuration (by {selection_criterion}):")
    print(f"  {best_config[0]}: Val F1 = {best_config[1]['val_f1']:.3f}, Test F1 = {best_config[1]['test_f1']:.3f}")
    
    return results_pca, best_config


def train_combined_selectkbest_pca_lda(X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray,
                                       X_test: np.ndarray, y_test: np.ndarray,
                                       combined_configs: List[Tuple[int, int]],
                                       seed: int,
                                       selection_criterion: str = 'val_f1') -> Tuple[Dict, Dict]:
    """
    Train LDA with combined SelectKBest + PCA.
    
    Args:
        combined_configs: List of (k, n) tuples where k=features, n=PCA components
    
    Returns:
        results_combined: Dictionary of all results
        best_config: Best configuration
    """
    print("\n" + "="*70)
    print("COMBINED - SelectKBest + PCA + LDA")
    print("="*70 + "\n")
    
    results_combined = {}
    
    for k, n in combined_configs:
        print(f"Testing k={k}, n={n}...")
        
        pipeline = Pipeline([
            ('selector', SelectKBest(f_classif, k=k)),
            ('pca', PCA(n_components=n, random_state=seed)),
            ('lda', LinearDiscriminantAnalysis())
        ])
        
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        
        results_combined[f'Combined_k{k}_n{n}'] = {
            'model': pipeline,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'y_test_pred': y_test_pred,
            'k': k,
            'n': n
        }
        
        print(f"  k={k}, n={n}: Val F1 = {val_f1:.3f}, Test F1 = {test_f1:.3f}")
    
    # Select best
    best_config = max(results_combined.items(), key=lambda x: x[1][selection_criterion])
    print(f"\nBest Combined configuration (by {selection_criterion}):")
    print(f"  {best_config[0]}: Val F1 = {best_config[1]['val_f1']:.3f}, Test F1 = {best_config[1]['test_f1']:.3f}")
    
    return results_combined, best_config


def train_bagging_lda(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      n_estimators_list: List[int],
                      max_samples: float,
                      max_features: float,
                      seed: int) -> Tuple[Dict, Dict]:
    """
    Train Bagging ensemble with LDA base estimator.
    
    Returns:
        results_bagging: Dictionary of all results
        best_config: Best configuration
    """
    print("\n" + "="*70)
    print("ENSEMBLE - Bagging + LDA")
    print("="*70 + "\n")
    
    results_bagging = {}
    
    for n_est in n_estimators_list:
        print(f"Testing n_estimators={n_est}...")
        
        base_lda = LinearDiscriminantAnalysis()
        bagging = BaggingClassifier(
            estimator=base_lda,
            n_estimators=n_est,
            random_state=seed,
            max_samples=max_samples,
            max_features=max_features
        )
        
        bagging.fit(X_train, y_train)
        y_val_pred = bagging.predict(X_val)
        y_test_pred = bagging.predict(X_test)
        
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        
        results_bagging[f'Bagging_n{n_est}'] = {
            'model': bagging,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'y_test_pred': y_test_pred
        }
        
        print(f"  n={n_est}: Val F1 = {val_f1:.3f}, Test F1 = {test_f1:.3f}")
    
    best_config = max(results_bagging.items(), key=lambda x: x[1]['test_f1'])
    print(f"\nBest Bagging configuration:")
    print(f"  {best_config[0]}: Test F1 = {best_config[1]['test_f1']:.3f}")
    
    return results_bagging, best_config


def train_gridsearch_lda(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         cv_folds: int,
                         n_jobs: int,
                         seed: int) -> Dict:
    """
    Train LDA with GridSearch optimization.
    
    Returns:
        Dictionary with best model and results
    """
    print("\n" + "="*70)
    print("GRIDSEARCH OPTIMIZATION - SelectKBest + LDA")
    print("="*70 + "\n")
    
    pipeline_grid = Pipeline([
        ('selector', SelectKBest(f_classif)),
        ('lda', LinearDiscriminantAnalysis())
    ])
    
    # Create valid parameter combinations
    param_grid = [
        # SVD with no shrinkage
        {'selector__k': [50], 'lda__solver': ['svd'], 'lda__shrinkage': [None]},
        {'selector__k': [100], 'lda__solver': ['svd'], 'lda__shrinkage': [None]},
        {'selector__k': [150], 'lda__solver': ['svd'], 'lda__shrinkage': [None]},
        {'selector__k': [200], 'lda__solver': ['svd'], 'lda__shrinkage': [None]},
        # LSQR/Eigen with shrinkage
        {'selector__k': [50, 100, 150, 200], 'lda__solver': ['lsqr', 'eigen'], 
         'lda__shrinkage': ['auto', 0.1, 0.5]}
    ]
    
    print("Running GridSearch...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(
        pipeline_grid,
        param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = grid_search.predict(X_val)
    y_test_pred = grid_search.predict(X_test)
    
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print(f"\nGridSearch Best Parameters: {grid_search.best_params_}")
    print(f"GridSearch Best CV Score: {grid_search.best_score_:.3f}")
    print(f"GridSearch Val F1: {val_f1:.3f}")
    print(f"GridSearch Test F1: {test_f1:.3f}")
    
    return {
        'GridSearch_LDA': {
            'model': grid_search.best_estimator_,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'y_test_pred': y_test_pred,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_
        }
    }
