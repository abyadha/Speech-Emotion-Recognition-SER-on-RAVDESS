"""
Visualization functions for emotion recognition results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from typing import Dict, List


def setup_plotting_style(style: str = 'whitegrid', dpi: int = 100):
    """Configure matplotlib plotting style."""
    sns.set_style(style)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'


def plot_selectkbest_validation_curve(results_kbest: Dict, k_values: List[int], 
                                       figures_dir: str):
    """
    Plot k vs F1 score for SelectKBest feature selection.
    
    Args:
        results_kbest: Dictionary of SelectKBest results
        k_values: List of k values tested
        figures_dir: Directory to save figure
    """
    print("\n" + "="*70)
    print("VISUALIZING SELECTKBEST RESULTS")
    print("="*70 + "\n")
    
    val_f1_scores = [results_kbest[f'SelectKBest_k{k}']['val_f1'] for k in k_values]
    test_f1_scores = [results_kbest[f'SelectKBest_k{k}']['test_f1'] for k in k_values]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, val_f1_scores, 'o-', linewidth=2, markersize=10, 
            label='Validation F1', color='blue')
    ax.plot(k_values, test_f1_scores, 's--', linewidth=2, markersize=10, 
            label='Test F1', color='red', alpha=0.6)
    
    # Mark best k
    best_k_idx = np.argmax(val_f1_scores)
    best_k = k_values[best_k_idx]
    ax.axvline(best_k, color='green', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Best k={best_k} (Val F1={val_f1_scores[best_k_idx]:.3f})')
    
    # Add value labels
    for i, (k, val_f1, test_f1) in enumerate(zip(k_values, val_f1_scores, test_f1_scores)):
        ax.text(k, val_f1 + 0.01, f'{val_f1:.3f}', ha='center', va='bottom', 
                fontsize=9, color='blue')
        ax.text(k, test_f1 - 0.01, f'{test_f1:.3f}', ha='center', va='top', 
                fontsize=9, color='red', alpha=0.7)
    
    ax.set_xlabel('Number of Features (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('SelectKBest Feature Selection: k vs F1 Score\n(Selection based on Validation F1)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    ax.set_ylim([min(min(val_f1_scores), min(test_f1_scores)) - 0.05, 
                 max(max(val_f1_scores), max(test_f1_scores)) + 0.05])
    
    plt.tight_layout()
    filepath = f'{figures_dir}/01_selectkbest_k_vs_f1.png'
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.show()
    
    print(f"Best k selected: {best_k} (Val F1: {val_f1_scores[best_k_idx]:.3f})")


def plot_pca_analysis(results_pca: Dict, n_components_list: List[int],
                     X_train_scaled: np.ndarray, seed: int, figures_dir: str):
    """
    Plot PCA scree plot and n_components vs F1 score.
    
    Args:
        results_pca: Dictionary of PCA results
        n_components_list: List of n_components tested
        X_train_scaled: Scaled training data for PCA fitting
        seed: Random seed
        figures_dir: Directory to save figure
    """
    print("\n" + "="*70)
    print("VISUALIZING PCA RESULTS")
    print("="*70 + "\n")
    
    # Fit PCA with all components
    pca_full = PCA(n_components=min(X_train_scaled.shape), random_state=seed)
    pca_full.fit(X_train_scaled)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # SUBPLOT 1: Scree plot
    ax1.plot(range(1, len(cumsum_var)+1), cumsum_var, 'o-', linewidth=2, markersize=4, color='navy')
    ax1.axhline(0.95, color='red', linestyle='--', linewidth=2, label='95% variance', alpha=0.7)
    ax1.axhline(0.90, color='orange', linestyle='--', linewidth=2, label='90% variance', alpha=0.7)
    
    # Mark tested values
    for n in n_components_list:
        ax1.axvline(n, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        variance_at_n = cumsum_var[n-1]
        ax1.text(n, variance_at_n, f'n={n}\n{variance_at_n:.2%}', 
                 rotation=0, ha='left', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('Number of Components', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax1.set_title('PCA Scree Plot - Cumulative Variance Explained', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, min(200, len(cumsum_var))])
    ax1.set_ylim([0, 1.05])
    
    # SUBPLOT 2: n vs F1
    val_f1_pca = [results_pca[f'PCA_n{n}']['val_f1'] for n in n_components_list]
    test_f1_pca = [results_pca[f'PCA_n{n}']['test_f1'] for n in n_components_list]
    
    ax2.plot(n_components_list, val_f1_pca, 'o-', linewidth=2, markersize=10,
             label='Validation F1', color='blue')
    ax2.plot(n_components_list, test_f1_pca, 's--', linewidth=2, markersize=10,
             label='Test F1', color='red', alpha=0.6)
    
    # Mark best n
    best_n_idx = np.argmax(val_f1_pca)
    best_n = n_components_list[best_n_idx]
    ax2.axvline(best_n, color='green', linestyle=':', linewidth=2, alpha=0.7,
                label=f'Best n={best_n} (Val F1={val_f1_pca[best_n_idx]:.3f})')
    
    # Add value labels
    for i, (n, val_f1, test_f1) in enumerate(zip(n_components_list, val_f1_pca, test_f1_pca)):
        ax2.text(n, val_f1 + 0.01, f'{val_f1:.3f}', ha='center', va='bottom', 
                fontsize=9, color='blue')
        ax2.text(n, test_f1 - 0.01, f'{test_f1:.3f}', ha='center', va='top', 
                fontsize=9, color='red', alpha=0.7)
    
    ax2.set_xlabel('Number of PCA Components (n)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax2.set_title('PCA + LDA: n_components vs F1 Score\n(Selection based on Validation F1)',
                  fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([min(min(val_f1_pca), min(test_f1_pca)) - 0.05, 
                  max(max(val_f1_pca), max(test_f1_pca)) + 0.05])
    
    plt.tight_layout()
    filepath = f'{figures_dir}/02_pca_analysis.png'
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.show()
    
    print(f"Best n selected: {best_n} (Val F1: {val_f1_pca[best_n_idx]:.3f})")
    print(f"Variance explained with n={best_n}: {cumsum_var[best_n-1]:.2%}")


def plot_combined_grid_search(results_combined: Dict, combined_configs: List, figures_dir: str):
    """
    Plot grid search heatmap for combined SelectKBest + PCA.
    
    Args:
        results_combined: Dictionary of combined results
        combined_configs: List of (k, n) tuples
        figures_dir: Directory to save figure
    """
    print("\n" + "="*70)
    print("VISUALIZING COMBINED GRID SEARCH")
    print("="*70 + "\n")
    
    # Extract unique values
    k_unique = sorted(list(set([k for k, n in combined_configs])))
    n_unique = sorted(list(set([n for k, n in combined_configs])))
    
    # Create matrices
    val_f1_matrix = np.full((len(k_unique), len(n_unique)), np.nan)
    test_f1_matrix = np.full((len(k_unique), len(n_unique)), np.nan)
    
    for i, k in enumerate(k_unique):
        for j, n in enumerate(n_unique):
            key = f'Combined_k{k}_n{n}'
            if key in results_combined:
                val_f1_matrix[i, j] = results_combined[key]['val_f1']
                test_f1_matrix[i, j] = results_combined[key]['test_f1']
    
    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 ,5))
    
    # Validation F1
    im1 = ax1.imshow(val_f1_matrix, cmap='YlGnBu', aspect='auto', vmin=0.3, vmax=0.7)
    ax1.set_xticks(range(len(n_unique)))
    ax1.set_yticks(range(len(k_unique)))
    ax1.set_xticklabels(n_unique, fontsize=11)
    ax1.set_yticklabels(k_unique, fontsize=11)
    ax1.set_xlabel('PCA Components (n)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Selected Features (k)', fontsize=13, fontweight='bold')
    ax1.set_title('Validation F1 - Grid Search\n(k in {' + ','.join(map(str, k_unique)) + '}, n < k)', 
                 fontsize=15, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='F1 Score')
    
    for i in range(len(k_unique)):
        for j in range(len(n_unique)):
            if not np.isnan(val_f1_matrix[i, j]):
                ax1.text(j, i, f'{val_f1_matrix[i, j]:.3f}',
                        ha="center", va="center", color="black", 
                        fontsize=11, fontweight='bold')
    
    # Test F1
    im2 = ax2.imshow(test_f1_matrix, cmap='YlGnBu', aspect='auto', vmin=0.3, vmax=0.7)
    ax2.set_xticks(range(len(n_unique)))
    ax2.set_yticks(range(len(k_unique)))
    ax2.set_xticklabels(n_unique, fontsize=11)
    ax2.set_yticklabels(k_unique, fontsize=11)
    ax2.set_xlabel('PCA Components (n)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Selected Features (k)', fontsize=13, fontweight='bold')
    ax2.set_title('Test F1 - Grid Search\n(k in {' + ','.join(map(str, k_unique)) + '}, n < k)', 
                 fontsize=15, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='F1 Score')
    
    for i in range(len(k_unique)):
        for j in range(len(n_unique)):
            if not np.isnan(test_f1_matrix[i, j]):
                ax2.text(j, i, f'{test_f1_matrix[i, j]:.3f}',
                        ha="center", va="center", color="black", 
                        fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filepath = f'{figures_dir}/03_combined_grid_search.png'
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.show()


def plot_top_models_comparison(all_results: Dict, figures_dir: str, top_n: int = 10):
    """
    Plot bar chart comparing top models by test F1.
    
    Args:
        all_results: Dictionary of all model results
        figures_dir: Directory to save figure
        top_n: Number of top models to display
    """
    print("\n" + "="*70)
    print("VISUALIZING TOP MODELS COMPARISON")
    print("="*70 + "\n")
    
    # Sort by test F1
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
    top_models = sorted_models[:top_n]
    
    names = [name for name, _ in top_models]
    test_f1 = [results['test_f1'] for _, results in top_models]
    val_f1 = [results['val_f1'] for _, results in top_models]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_f1, width, label='Validation F1', color='skyblue', alpha=0.9)
    bars2 = ax.bar(x + width/2, test_f1, width, label='Test F1', color='lightcoral', alpha=0.9)
    
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {top_n} Models by Test F1 Score', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/04_top_models_comparison.png'
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.show()


def plot_benchmark_comparison(best_model_name: str, test_f1_lda: float,
                              benchmark_linear: float, benchmark_rbf: float,
                              figures_dir: str):
    """
    Plot bar chart comparing LDA to SVM benchmarks.
    
    Args:
        best_model_name: Name of best LDA model
        test_f1_lda: Test F1 score of best LDA
        benchmark_linear: Linear SVM benchmark F1
        benchmark_rbf: RBF SVM benchmark F1
        figures_dir: Directory to save figure
    """
    print("\n" + "="*70)
    print("VISUALIZING BENCHMARK COMPARISON")
    print("="*70 + "\n")
    
    models = ['Linear SVM\n(Benchmark)', 'RBF SVM\n(Benchmark)', f'Best LDA\n({best_model_name})']
    f1_scores = [benchmark_linear, benchmark_rbf, test_f1_lda]
    colors = ['lightgray', 'lightgray', 'lightgreen']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, f1_scores, color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Test F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('LDA vs SVM Benchmarks - Test F1 Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{f1:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add difference from LDA
        if i < 2:
            diff = test_f1_lda - f1
            color = 'green' if diff > 0 else 'red'
            ax.text(bar.get_x() + bar.get_width()/2., height - 0.1,
                   f'({diff:+.3f})', ha='center', va='top', fontsize=10, color=color, fontweight='bold')
    
    plt.tight_layout()
    filepath = f'{figures_dir}/05_benchmark_comparison.png'
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.show()


def plot_per_emotion_comparison(y_test: np.ndarray, y_pred_lda: np.ndarray,
                                y_pred_linear: np.ndarray, y_pred_rbf: np.ndarray,
                                classes: List[str], best_model_name: str, figures_dir: str):
    """
    Plot per-emotion F1 comparison: LDA vs SVM benchmarks.
    
    Args:
        y_test: True test labels
        y_pred_lda: LDA predictions
        y_pred_linear: Linear SVM predictions
        y_pred_rbf: RBF SVM predictions
        classes: List of class names
        best_model_name: Name of best model
        figures_dir: Directory to save figure
    """
    from sklearn.metrics import f1_score
    
    print("\n" + "="*70)
    print("VISUALIZING PER-EMOTION COMPARISON")
    print("="*70 + "\n")
    
    # Compute per-class F1
    f1_lda = []
    f1_linear = []
    f1_rbf = []
    
    for cls_idx in range(len(classes)):
        y_true_binary = (y_test == cls_idx).astype(int)
        y_pred_lda_binary = (y_pred_lda == cls_idx).astype(int)
        y_pred_linear_binary = (y_pred_linear == cls_idx).astype(int)
        y_pred_rbf_binary = (y_pred_rbf == cls_idx).astype(int)
        
        f1_lda.append(f1_score(y_true_binary, y_pred_lda_binary, average='binary', zero_division=0))
        f1_linear.append(f1_score(y_true_binary, y_pred_linear_binary, average='binary', zero_division=0))
        f1_rbf.append(f1_score(y_true_binary, y_pred_rbf_binary, average='binary', zero_division=0))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax.bar(x - width, f1_lda, width, label=f'Best LDA ({best_model_name})', 
                   color='lightblue', alpha=0.9)
    bars2 = ax.bar(x, f1_linear, width, label='Linear SVM', color='lightcoral', alpha=0.9)
    bars3 = ax.bar(x + width, f1_rbf, width, label='RBF SVM', color='lightgreen', alpha=0.9)
    
    ax.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Emotion', fontsize=13, fontweight='bold')
    ax.set_title(f'Per-Emotion F1 Comparison: {best_model_name} vs SVM Benchmarks',
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filepath = f'{figures_dir}/06_per_emotion_comparison.png'
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.show()


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, classes: List[str],
                         model_name: str, test_f1: float, figures_dir: str):
    """
    Plot confusion matrix (counts and normalized).
    
    Args:
        y_test: True test labels
        y_pred: Predicted labels
        classes: List of class names
        model_name: Model name for title
        test_f1: Test F1 score for title
        figures_dir: Directory to save figure
    """
    print("\n" + "="*70)
    print("VISUALIZING CONFUSION MATRIX")
    print("="*70 + "\n")
    
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 ,5))
    
    # Counts
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title(f'{model_name} - Counts\nTest F1 = {test_f1:.3f}',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax1.text(j, i, f'{cm[i, j]}',
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=11, fontweight='bold')
    
    ax1.set_xticks(range(len(classes)))
    ax1.set_yticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.set_yticklabels(classes)
    ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    # Normalized
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.set_title(f'{model_name} - Normalized\nTest F1 = {test_f1:.3f}',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax2.text(j, i, f'{cm_norm[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=11, fontweight='bold')
    
    ax2.set_xticks(range(len(classes)))
    ax2.set_yticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_yticklabels(classes)
    ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filepath = f'{figures_dir}/07_confusion_matrix_{model_name.replace(" ", "_")}.png'
    plt.savefig(filepath, dpi=300)
    print(f"Saved: {filepath}")
    plt.show()
