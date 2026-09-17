"""
Feature engineering utilities for Lead Conversion Prediction.

Provides functions to create derived features and handle outliers
in numerical columns.
"""

import numpy as np
import pandas as pd


# ── Derived features ─────────────────────────────────────────────────────────


def create_activity_channel_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many marketing/activity channels a lead interacted with.

    Sums the binary Yes/No activity columns into a single integer feature.
    This reduces dimensionality while preserving the overall engagement signal.
    """
    activity_cols = [
        "Search",
        "Newspaper Article",
        "X Education Forums",
        "Newspaper",
        "Digital Advertisement",
        "Through Recommendations",
    ]
    df = df.copy()
    existing = [c for c in activity_cols if c in df.columns]
    if existing:
        df["Activity_Channel_Count"] = df[existing].apply(
            lambda row: (row == "Yes").sum(), axis=1
        )
    return df


# ── Outlier handling ─────────────────────────────────────────────────────────


def cap_outliers(
    series: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
    train_series: pd.Series | None = None,
) -> pd.Series:
    """Cap outliers using percentile thresholds.

    Parameters
    ----------
    series : pd.Series
        The series to cap.
    lower_pct, upper_pct : float
        Percentile boundaries.
    train_series : pd.Series, optional
        If provided, thresholds are learned from *this* series instead of
        ``series`` itself.  Use this to prevent data leakage when capping
        the test set.

    Returns
    -------
    pd.Series
        Capped series.
    """
    reference = train_series if train_series is not None else series
    lower = reference.quantile(lower_pct)
    upper = reference.quantile(upper_pct)
    return series.clip(lower=lower, upper=upper)
