"""
Model training utilities for Lead Conversion Prediction.

Provides helper functions to build scikit-learn preprocessing pipelines,
combine them with classifiers, and run hyperparameter tuning via
GridSearchCV.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Build a ``ColumnTransformer`` that handles numerical and categorical
    features separately.

    Numerical path:   median imputation → standard scaling.
    Categorical path:  most-frequent imputation → one-hot encoding.
    """
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Combine a preprocessor and a classifier into a single pipeline."""
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def tune_model(
    pipeline: Pipeline,
    param_grid: dict,
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "f1",
) -> GridSearchCV:
    """Run ``GridSearchCV`` on a pipeline.

    Returns the fitted ``GridSearchCV`` object so callers can inspect
    ``best_params_``, ``best_score_``, etc.
    """
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search
