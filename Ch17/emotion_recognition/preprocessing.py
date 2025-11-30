"""
Data preprocessing functions: augmentation and scaling.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def augment_data(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                use_augmentation: bool, n_augmentations: int, 
                noise_level: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment training data with feature-space noise.
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Group identifiers
        use_augmentation: Whether to apply augmentation
        n_augmentations: Number of augmented copies per sample
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        X_aug: Augmented feature matrix
        y_aug: Augmented labels
        g_aug: Augmented groups
    """
    if not use_augmentation:
        return X, y, groups

    np.random.seed(seed)
    X_aug_list = [X]
    y_aug_list = [y]
    g_aug_list = [groups]

    for _ in range(n_augmentations):
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug = X + noise
        X_aug_list.append(X_aug)
        y_aug_list.append(y)
        g_aug_list.append(groups)

    X_out = np.vstack(X_aug_list)
    y_out = np.hstack(y_aug_list)
    g_out = np.hstack(g_aug_list)

    print(f"Training data augmented: {len(X)} -> {len(X_out)} samples")
    return X_out, y_out, g_out


def standardize_features(X_train: np.ndarray, X_val: np.ndarray, 
                        X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features (fit on training, apply to all).
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        
    Returns:
        X_train_scaled: Scaled training features
        X_val_scaled: Scaled validation features
        X_test_scaled: Scaled test features
        scaler: Fitted StandardScaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nFeatures standardized:")
    print(f"  Train: {X_train_scaled.shape}")
    print(f"  Val:   {X_val_scaled.shape}")
    print(f"  Test:  {X_test_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
