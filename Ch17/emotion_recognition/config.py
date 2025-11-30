"""
Ch17 project SER.
"""

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = '/content/drive/MyDrive/Fall25/EE502/FinalProjects/Ch17'
DATA_DIR = f'{BASE_DIR}/data'
FIGURES_DIR = f'{BASE_DIR}/figures'
MANIFEST_SPLIT = f'{DATA_DIR}/manifest_ser_5emotions_split.csv'
CACHE_PATH = f'{DATA_DIR}/_cache_ser_features_BASIC.npz'

# ============================================================================
# AUDIO SETTINGS
# ============================================================================
SR = 16000  # Sample rate
N_MFCC = 13  # Number of MFCC coefficients

# ============================================================================
# AUGMENTATION SETTINGS
# ============================================================================
USE_AUGMENTATION = True
N_AUGMENTATIONS = 1
AUGMENTATION_NOISE_LEVEL = 0.05

# ============================================================================
# RANDOM SEED
# ============================================================================
CV_SEED = 42

# ============================================================================
# TARGET EMOTIONS
# ============================================================================
TARGET_EMOTIONS = {'happy', 'sad', 'angry', 'fearful', 'neutral'}

# ============================================================================
# FEATURE SELECTION PARAMETERS (UPDATED - k between 10-14)
# ============================================================================
SELECTKBEST_K_VALUES = [10, 11, 12, 13, 14]

# ============================================================================
# PCA PARAMETERS
# ============================================================================
PCA_N_COMPONENTS_LIST = [20, 50, 100, 150]

# ============================================================================
# COMBINED FEATURE SELECTION + PCA
# (k, n) pairs where n < k to avoid dimensionality errors
# ============================================================================
COMBINED_CONFIGS = [
    (11, 5),
    (11, 8),
    (12, 5),
    (12, 8),
    (13, 5),
    (13, 10),
    (14, 5),
    (14, 10)
]

# ============================================================================
# BAGGING ENSEMBLE PARAMETERS
# ============================================================================
BAGGING_N_ESTIMATORS_LIST = [5, 10, 15, 20]
BAGGING_MAX_SAMPLES = 0.8
BAGGING_MAX_FEATURES = 0.8

# ============================================================================
# GRIDSEARCH PARAMETERS
# ============================================================================
GRIDSEARCH_CV_FOLDS = 3
GRIDSEARCH_N_JOBS = -1

# ============================================================================
# BENCHMARK VALUES
# ============================================================================
BENCHMARK_LINEAR_SVM = 0.581
BENCHMARK_RBF_SVM = 0.633
BENCHMARK_MINIMUM = 0.60
BENCHMARK_COMPETITIVE = 0.633
BENCHMARK_EXCELLENT = 0.653

# ============================================================================
# MODEL SELECTION CRITERION
# ============================================================================

MODEL_SELECTION_CRITERION = 'val_f1'

# ============================================================================
# PLOTTING SETTINGS
# ============================================================================
PLOT_DPI = 300
PLOT_STYLE = 'whitegrid'
FIGURE_SIZE_SINGLE = (10, 6)
FIGURE_SIZE_DOUBLE = (16, 6)

# ============================================================================
# PATHS FOR SAVED FILES
# ============================================================================
BEST_MODEL_PATH = f'{DATA_DIR}/best_lda_final.pkl'
RESULTS_JSON_PATH = f'{DATA_DIR}/final_lda_results.json'
SVM_LINEAR_MODEL_PATH = f'{DATA_DIR}/benchmark_linear_svm.pkl'
SVM_RBF_MODEL_PATH = f'{DATA_DIR}/benchmark_rbf_svm.pkl'
SVM_LINEAR_PREDICTIONS_PATH = f'{DATA_DIR}/svm_linear_predictions.npy'
SVM_RBF_PREDICTIONS_PATH = f'{DATA_DIR}/svm_rbf_predictions.npy'


def print_configuration():
    """Print current configuration."""
    print("=" * 70)
    print("EMOTION RECOGNITION - CONFIGURATION")
    print("=" * 70)
    print(f"Dataset: RAVDESS (5 emotions)")
    print(f"Target Emotions: {sorted(TARGET_EMOTIONS)}")
    print(f"Feature Selection: k in {SELECTKBEST_K_VALUES}")
    print(f"PCA Components: n in {PCA_N_COMPONENTS_LIST}")
    print(f"Combined Configs: {len(COMBINED_CONFIGS)} combinations")
    print(f"Bagging Estimators: {BAGGING_N_ESTIMATORS_LIST}")
    print(f"Benchmarks: Linear SVM = {BENCHMARK_LINEAR_SVM}, RBF SVM = {BENCHMARK_RBF_SVM}")
    print(f"Model Selection: Based on {MODEL_SELECTION_CRITERION.upper()}")
    print(f"Random Seed: {CV_SEED}")
    print("=" * 70)
    print()
