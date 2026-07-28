"""Train-only imputer, scaler and PCA."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pca_pipeline(n_components: int = 4) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )


def fit_transform_pca(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline, np.ndarray]:
    """Fit pipeline on train features only; transform all partitions."""
    pipe = build_pca_pipeline(n_components=n_components)
    X_train = pipe.fit_transform(train[feature_cols])
    X_val = pipe.transform(validation[feature_cols])
    X_test = pipe.transform(test[feature_cols])
    explained = pipe.named_steps["pca"].explained_variance_ratio_
    return X_train, X_val, X_test, pipe, explained


def save_pipeline(pipe: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load_pipeline(path: str | Path) -> Pipeline:
    return joblib.load(path)
