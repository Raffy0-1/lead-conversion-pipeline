"""
General utilities for Lead Conversion Prediction.

Random seed management and model serialisation helpers.
"""

import os
import random

import joblib
import numpy as np


def set_random_seed(seed: int = 42) -> None:
    """Set global random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, filepath: str) -> None:
    """Persist a trained model (or pipeline) to disk via joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved → {filepath}")


def load_model(filepath: str):
    """Load a previously saved model from disk."""
    return joblib.load(filepath)
