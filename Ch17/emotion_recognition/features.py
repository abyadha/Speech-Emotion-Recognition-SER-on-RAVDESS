"""
Audio feature extraction functions for SER
"""

import numpy as np
import librosa
from tqdm import tqdm
import os
from typing import Optional


def extract_basic_features(path: str, sr: int, n_mfcc: int) -> Optional[np.ndarray]:
    """
    Extract basic audio features (MFCC + deltas + spectral features).
    
    Args:
        path: Path to audio file
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Feature vector (1D array) or None if extraction fails
    """
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if y is None or len(y) == 0:
            return None
    except:
        return None

    def _stats(mat):
        """Compute statistics (mean, std, min, max) for each row."""
        return np.hstack([mat.mean(axis=1), mat.std(axis=1),
                         mat.min(axis=1), mat.max(axis=1)])

    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    stack = np.vstack([mfcc, d1, d2])
    feats = [_stats(stack)]

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spec_contr = librosa.feature.spectral_contrast(y=y, sr=sr)

    for F in [spec_cent, spec_bw, spec_roll, zcr, rms]:
        feats.append(_stats(F))
    feats.append(_stats(spec_contr))

    return np.hstack(feats).astype(np.float32)


def build_or_load_cache(df_all, cache_path: str, sr: int, n_mfcc: int, class_to_idx: dict):
    """
    Build feature cache or load existing cache.
    
    Args:
        df_all: Dataframe with file paths and emotions
        cache_path: Path to cache file
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        class_to_idx: Dictionary mapping emotions to indices
        
    Returns:
        X_all: Feature matrix (n_samples, n_features)
        y_all: Label vector (n_samples,)
        paths_all: Array of file paths (n_samples,)
    """
    if os.path.exists(cache_path):
        print("\nLoading cached features...")
        data = np.load(cache_path, allow_pickle=True)
        X_all = data['X_all']
        y_all = data['y_all']
        paths_all = data['paths_all']
        print(f"Loaded features: {X_all.shape[1]} dimensions, {X_all.shape[0]} samples")
        return X_all, y_all, paths_all

    print("\nExtracting features (this may take a while)...")
    X_list, y_list, p_list = [], [], []

    for pth, emo in tqdm(zip(df_all['filepath'], df_all['emotion']),
                         total=len(df_all), desc="Extracting"):
        x = extract_basic_features(pth, sr, n_mfcc)
        if x is not None:
            X_list.append(x)
            y_list.append(class_to_idx[emo])
            p_list.append(pth)

    X_all = np.vstack(X_list)
    y_all = np.array(y_list, dtype=np.int64)
    paths_all = np.array(p_list)

    # Save cache
    np.savez(cache_path, X_all=X_all, y_all=y_all, paths_all=paths_all)
    print(f"Cache saved to: {cache_path}")
    print(f"Features: {X_all.shape[1]} dimensions, {X_all.shape[0]} samples")
    
    return X_all, y_all, paths_all
