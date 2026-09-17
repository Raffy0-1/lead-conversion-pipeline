"""
Model evaluation utilities for Lead Conversion Prediction.

Provides functions to compute classification metrics, plot confusion
matrices, ROC curves, and build model comparison tables.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    model, X_test, y_test, model_name: str = "Model"
) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """Calculate standard classification metrics on a hold-out set.

    Returns
    -------
    metrics : dict
        Keys: Model, Precision, Recall, F1, ROC-AUC, Accuracy.
    y_pred : np.ndarray
    y_prob : np.ndarray or None
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "Model": model_name,
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": (
            round(roc_auc_score(y_test, y_prob), 4)
            if y_prob is not None
            else None
        ),
        "Accuracy": round((y_pred == y_test).mean(), 4),
    }
    return metrics, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, model_name: str = "Model", ax=None):
    """Plot a confusion-matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Not Converted", "Converted"],
        yticklabels=["Not Converted", "Converted"],
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    return ax


def plot_roc_curve(y_test, y_probs_dict: dict, ax=None):
    """Plot ROC curves for one or more models on the same axes.

    Parameters
    ----------
    y_probs_dict : dict[str, np.ndarray]
        ``{model_name: predicted_probabilities}``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for model_name, y_prob in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return ax


def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """Create a comparison DataFrame from a list of metric dicts."""
    return pd.DataFrame(metrics_list).set_index("Model")
