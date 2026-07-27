"""Temporal split and stratified subsample for quantum comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split; rows with NaN future target are dropped first."""
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("split fractions must sum to 1")

    data = df.dropna(subset=["future_pm25"]).sort_values("timestamp").reset_index(drop=True)
    n = len(data)
    i_train = int(n * train_fraction)
    i_val = i_train + int(n * validation_fraction)

    train = data.iloc[:i_train].copy()
    validation = data.iloc[i_train:i_val].copy()
    test = data.iloc[i_val:].copy()

    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise ValueError("empty partition after temporal split")

    if not (train["timestamp"].max() < validation["timestamp"].min()):
        raise ValueError("train/validation temporal overlap")
    if not (validation["timestamp"].max() < test["timestamp"].min()):
        raise ValueError("validation/test temporal overlap")

    return train, validation, test


def stratified_subsample(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_size: int = 500,
    validation_size: int = 150,
    test_size: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified subsample by target; returns subsets + index table."""

    def _sample(part: pd.DataFrame, size: int, name: str) -> pd.DataFrame:
        size = min(size, len(part))
        if size < len(part):
            idx, _ = train_test_split(
                part.index,
                train_size=size,
                stratify=part["target"],
                random_state=seed,
            )
            out = part.loc[idx].sort_values("timestamp").copy()
        else:
            out = part.copy()
        out = out.reset_index(names="original_index")
        out["partition"] = name
        return out

    tr = _sample(train, train_size, "train")
    va = _sample(validation, validation_size, "validation")
    te = _sample(test, test_size, "test")

    index_table = pd.concat(
        [
            tr[["original_index", "timestamp", "target", "partition"]],
            va[["original_index", "timestamp", "target", "partition"]],
            te[["original_index", "timestamp", "target", "partition"]],
        ],
        ignore_index=True,
    )
    return tr, va, te, index_table


def save_sample_indices(index_table: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index_table.to_csv(path, index=False)
