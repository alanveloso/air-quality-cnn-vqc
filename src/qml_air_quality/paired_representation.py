"""Paired full-8D and PCA-2 + angular representations (train-only fit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qml_air_quality.preprocessing.angular_scaling import (
    AngularScaler,
    make_angular_scaler,
)


@dataclass
class FittedTransform:
    """Tracks that transformers were fit only on the train partition."""

    name: str
    fit_partition: str = "train"
    transformer: Any = None


@dataclass
class PairedRepresentation:
    """Full 8D and PCA-2 embeddings for classical/quantum fair comparison."""

    feature_cols: list[str]
    X_train_full: np.ndarray
    X_val_full: np.ndarray
    X_test_full: np.ndarray
    X_train_pca: np.ndarray
    X_val_pca: np.ndarray
    X_test_pca: np.ndarray
    full_pipeline: Pipeline
    pca_pipeline: Pipeline
    fit_records: list[FittedTransform] = field(default_factory=list)
    explained_variance_ratio: np.ndarray | None = None

    def angular(
        self,
        scaler_name: str,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, AngularScaler, FittedTransform]:
        ang = make_angular_scaler(scaler_name, seed=seed)
        X_tr = ang.fit_transform(self.X_train_pca)
        X_va = ang.transform(self.X_val_pca)
        X_te = ang.transform(self.X_test_pca)
        record = FittedTransform(name=f"angular_{scaler_name}", fit_partition="train", transformer=ang)
        return X_tr, X_va, X_te, ang, record


def build_full_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_pca_pipeline(n_components: int = 2, seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
        ]
    )


def fit_paired_representation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    pca_components: int = 2,
    seed: int = 42,
) -> PairedRepresentation:
    """Fit imputer/scaler/PCA on train only; transform all partitions."""
    full_pipe = build_full_pipeline()
    pca_pipe = build_pca_pipeline(n_components=pca_components, seed=seed)

    X_train_full = full_pipe.fit_transform(train[feature_cols])
    X_val_full = full_pipe.transform(validation[feature_cols])
    X_test_full = full_pipe.transform(test[feature_cols])

    X_train_pca = pca_pipe.fit_transform(train[feature_cols])
    X_val_pca = pca_pipe.transform(validation[feature_cols])
    X_test_pca = pca_pipe.transform(test[feature_cols])

    records = [
        FittedTransform("full_imputer_scaler", "train", full_pipe),
        FittedTransform("pca_pipeline", "train", pca_pipe),
    ]
    explained = pca_pipe.named_steps["pca"].explained_variance_ratio_
    return PairedRepresentation(
        feature_cols=list(feature_cols),
        X_train_full=np.asarray(X_train_full, dtype=float),
        X_val_full=np.asarray(X_val_full, dtype=float),
        X_test_full=np.asarray(X_test_full, dtype=float),
        X_train_pca=np.asarray(X_train_pca, dtype=float),
        X_val_pca=np.asarray(X_val_pca, dtype=float),
        X_test_pca=np.asarray(X_test_pca, dtype=float),
        full_pipeline=full_pipe,
        pca_pipeline=pca_pipe,
        fit_records=records,
        explained_variance_ratio=explained,
    )


def save_preprocessors(repr_: PairedRepresentation, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(repr_.full_pipeline, directory / "full_pipeline.joblib")
    joblib.dump(repr_.pca_pipeline, directory / "pca_pipeline.joblib")
