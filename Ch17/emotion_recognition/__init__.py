"""
Ch17 SER Project.

Linear Discriminant Analysis (LDA) with various feature selection and
dimensionality reduction techniques.

Modules:
    config: Configuration parameters
    utils: Utility functions
    data_loading: Data loading and splitting
    features: Audio feature extraction
    preprocessing: Data augmentation and scaling
    models: Model training functions
    evaluation: Metrics and statistical tests
    visualization: Plotting functions
"""

__version__ = "1.0.0"
__author__ = "Ahmed Abyadh"

# Import key functions
from .config import (
    BASE_DIR,
    DATA_DIR,
    FIGURES_DIR,
    TARGET_EMOTIONS,
    SELECTKBEST_K_VALUES,
    PCA_N_COMPONENTS_LIST,
    COMBINED_CONFIGS,
    BENCHMARK_LINEAR_SVM,
    BENCHMARK_RBF_SVM,
    MODEL_SELECTION_CRITERION,
    print_configuration
)

from .utils import (
    set_random_seed,
    save_results,
    print_section_header,
    print_benchmark_comparison,
    meets_benchmark_targets
)

from .data_loading import (
    load_metadata,
    create_train_val_test_splits
)

from .features import (
    extract_basic_features,
    build_or_load_cache
)

from .preprocessing import (
    augment_data,
    standardize_features
)

from .models import (
    train_baseline_lda,
    train_selectkbest_lda,
    train_pca_lda,
    train_combined_selectkbest_pca_lda,
    train_bagging_lda,
    train_gridsearch_lda
)

from .evaluation import (
    select_best_model,
    load_svm_predictions,
    run_statistical_tests,
    compute_per_class_f1,
    save_best_model
)

from .visualization import (
    setup_plotting_style,
    plot_selectkbest_validation_curve,
    plot_pca_analysis,
    plot_combined_grid_search,
    plot_top_models_comparison,
    plot_benchmark_comparison,
    plot_per_emotion_comparison,
    plot_confusion_matrix
)

__all__ = [
    # Config
    'print_configuration',
    # Utils
    'set_random_seed',
    'save_results',
    'print_section_header',
    'print_benchmark_comparison',
    'meets_benchmark_targets',
    # Data loading
    'load_metadata',
    'create_train_val_test_splits',
    # Features
    'extract_basic_features',
    'build_or_load_cache',
    # Preprocessing
    'augment_data',
    'standardize_features',
    # Models
    'train_baseline_lda',
    'train_selectkbest_lda',
    'train_pca_lda',
    'train_combined_selectkbest_pca_lda',
    'train_bagging_lda',
    'train_gridsearch_lda',
    # Evaluation
    'select_best_model',
    'load_svm_predictions',
    'run_statistical_tests',
    'compute_per_class_f1',
    'save_best_model',
    # Visualization
    'setup_plotting_style',
    'plot_selectkbest_validation_curve',
    'plot_pca_analysis',
    'plot_combined_grid_search',
    'plot_top_models_comparison',
    'plot_benchmark_comparison',
    'plot_per_emotion_comparison',
    'plot_confusion_matrix',
]
