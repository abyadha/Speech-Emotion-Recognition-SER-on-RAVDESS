"""
Data loading and splitting functions for emotion recognition.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple


def load_metadata(manifest_path: str, target_emotions: set) -> Tuple[pd.DataFrame, list, dict]:
    """
    Load and filter metadata from manifest file.
    
    Args:
        manifest_path: Path to manifest CSV file
        target_emotions: Set of emotion labels to keep
        
    Returns:
        df_all: Filtered dataframe with target emotions
        classes: Sorted list of class labels
        class_to_idx: Dictionary mapping class labels to indices
    """
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"
    print("Manifest file found")
    
    # Load and filter
    df_all = pd.read_csv(manifest_path)
    df_all = df_all[df_all['emotion'].isin(target_emotions)].reset_index(drop=True)
    
    # Create class mapping
    classes = sorted(target_emotions)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    print(f"Classes: {classes}")
    print(f"Total samples: {len(df_all)}")
    print("\nClass distribution:")
    print(df_all['emotion'].value_counts().sort_index())
    
    return df_all, classes, class_to_idx


def get_split_data(df_all: pd.DataFrame, split: str, X_all: np.ndarray, 
                   y_all: np.ndarray, paths_all: np.ndarray) -> Tuple:
    """
    Get data for a specific split (train/val/test).
    
    Args:
        df_all: Complete dataframe with all samples
        split: Split name ('train', 'val', or 'test')
        X_all: All feature vectors
        y_all: All labels
        paths_all: All file paths
        
    Returns:
        df_split: Dataframe for this split
        X: Feature vectors for this split
        y: Labels for this split
        groups: Actor groups for this split
    """
    # Create index mapping
    idx_map = pd.DataFrame({'filepath': paths_all, 'idx': np.arange(len(paths_all))})
    
    # Get data for this split
    df_split = df_all[df_all['split'] == split].merge(idx_map, on='filepath').reset_index(drop=True)
    X = X_all[df_split['idx'].values]
    y = y_all[df_split['idx'].values]
    groups = df_split['actor'].values
    
    return df_split, X, y, groups


def create_train_val_test_splits(df_all: pd.DataFrame, X_all: np.ndarray, 
                                  y_all: np.ndarray, paths_all: np.ndarray) -> dict:
    """
    Create train, validation, and test splits.
    
    Args:
        df_all: Complete dataframe
        X_all: All features
        y_all: All labels
        paths_all: All file paths
        
    Returns:
        Dictionary containing all splits with keys:
            'train': (df_train, X_train, y_train, g_train)
            'val': (df_val, X_val, y_val, g_val)
            'test': (df_test, X_test, y_test, g_test)
    """
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        df_split, X, y, groups = get_split_data(df_all, split_name, X_all, y_all, paths_all)
        splits[split_name] = (df_split, X, y, groups)
    
    # Print split sizes
    print(f"\nData splits created:")
    print(f"  Train: {len(splits['train'][1])} samples")
    print(f"  Val:   {len(splits['val'][1])} samples")
    print(f"  Test:  {len(splits['test'][1])} samples")
    
    return splits
