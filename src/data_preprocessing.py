"""
Data preprocessing utilities for Lead Conversion Prediction.

This module provides functions to load, clean, and prepare the Leads dataset
for machine learning. It handles:
  - Loading raw CSV data
  - Treating placeholder values ('Select') as missing
  - Dropping uninformative columns (constant, ID, high-missing)
  - Identifying numerical vs categorical feature groups
"""

import pandas as pd
import numpy as np


# ── Column groups identified during data inspection ──────────────────────────

CONSTANT_COLUMNS = [
    "Magazine",
    "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content",
    "Get updates on DM Content",
    "I agree to pay the amount through cheque",
]

ID_COLUMNS = ["Prospect ID", "Lead Number"]

HIGH_MISSING_COLUMNS = [
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
    "Lead Quality",
]

LOW_SIGNAL_COLUMNS = [
    "What matters most to you in choosing a course",
    "How did you hear about X Education",
    "Country",
]

SELECT_COLUMNS = ["Specialization", "Lead Profile", "City"]


# ── Functions ────────────────────────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
    """Load the leads dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the Leads.csv file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(filepath)


def treat_select_as_missing(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Replace 'Select' placeholder values with ``NaN``.

    Many categorical columns in the Leads dataset use the string 'Select'
    to represent a non-answer.  Treating these as missing allows proper
    imputation downstream.
    """
    if columns is None:
        columns = SELECT_COLUMNS
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace("Select", np.nan)
    return df


def drop_uninformative_columns(
    df: pd.DataFrame, columns_to_drop: list[str] | None = None
) -> pd.DataFrame:
    """Drop columns that carry no predictive signal.

    Default list includes constant-value columns, ID columns,
    columns with > 45 % missing values, and near-zero-variance columns
    identified during exploratory analysis.
    """
    if columns_to_drop is None:
        columns_to_drop = (
            CONSTANT_COLUMNS
            + ID_COLUMNS
            + HIGH_MISSING_COLUMNS
            + LOW_SIGNAL_COLUMNS
        )
    existing = [c for c in columns_to_drop if c in df.columns]
    return df.drop(columns=existing)


def get_column_groups(
    df: pd.DataFrame, target_col: str = "Converted"
) -> tuple[list[str], list[str]]:
    """Identify numerical and categorical feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        The (cleaned) dataframe.
    target_col : str
        Name of the target column to exclude from features.

    Returns
    -------
    numerical_cols : list[str]
    categorical_cols : list[str]
    """
    feature_cols = [c for c in df.columns if c != target_col]
    numerical_cols = (
        df[feature_cols]
        .select_dtypes(include=["int64", "float64"])
        .columns.tolist()
    )
    categorical_cols = (
        df[feature_cols]
        .select_dtypes(include=["object", "string", "category"])
        .columns.tolist()
    )
    return numerical_cols, categorical_cols
