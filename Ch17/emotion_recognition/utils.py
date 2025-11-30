"""
Utility functions for SER project.
"""

import numpy as np
import json
import random
from typing import Any, Dict


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to: {seed}")


def make_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types and model objects to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif hasattr(obj, 'predict'):
        # This is a model object
        return f"<Model: {obj.__class__.__name__}>"
    else:
        return obj


def save_results(results: Dict, filepath: str):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save JSON file
    """
    serializable_results = make_json_serializable(results)
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to: {filepath}")


def print_section_header(title: str, width: int = 70):
    """Print a formatted section header."""
    print()
    print("=" * width)
    print(title)
    print("=" * width)
    print()


def print_subsection_header(title: str, width: int = 70):
    """Print a formatted subsection header."""
    print()
    print("-" * width)
    print(title)
    print("-" * width)


def format_f1_comparison(name: str, val_f1: float, test_f1: float, width: int = 40):
    """Format F1 scores for display."""
    return f"{name:<{width}} | Val F1: {val_f1:.3f} | Test F1: {test_f1:.3f}"


def print_benchmark_comparison(test_f1: float, benchmark_linear: float, benchmark_rbf: float):
    """Print comparison with benchmarks."""
    print("\nBENCHMARK COMPARISON:")
    print(f"  Calc Model Test F1:  {test_f1:.3f}")
    print(f"  Linear SVM (paper):  {benchmark_linear:.3f}  |  Difference: {test_f1 - benchmark_linear:+.3f}")
    print(f"  RBF SVM (paper):     {benchmark_rbf:.3f}  |  Difference: {test_f1 - benchmark_rbf:+.3f}")


def meets_benchmark_targets(test_f1: float, minimum: float, competitive: float, excellent: float):
    """Check if model meets benchmark targets."""
    print("\nBENCHMARK TARGETS:")
    print(f"  Minimum ({minimum:.2f}):      {'ACHIEVED' if test_f1 >= minimum else 'NOT MET'} [{test_f1:.3f}]")
    print(f"  Competitive ({competitive:.2f}): {'ACHIEVED' if test_f1 >= competitive else 'NOT MET'} [{test_f1:.3f}]")
    print(f"  Excellent ({excellent:.2f}+):  {'ACHIEVED' if test_f1 >= excellent else 'NOT MET'} [{test_f1:.3f}]")
    
    if test_f1 >= excellent:
        print("\n" + "=" * 70)
        print("OUTSTANDING! MODEL EXCEEDS EXCELLENT TARGET!")
        print("=" * 70)
    elif test_f1 >= competitive:
        print("\n" + "=" * 70)
        print("SUCCESS! MODEL MATCHES/BEATS RBF SVM BENCHMARK!")
        print("=" * 70)
    elif test_f1 >= minimum:
        print("\n" + "=" * 70)
        print("GOOD! MODEL MEETS MINIMUM TARGET!")
        print("=" * 70)
    else:
        gap = minimum - test_f1
        print(f"\nModel is {gap:.3f} F1 points away from minimum target.")
