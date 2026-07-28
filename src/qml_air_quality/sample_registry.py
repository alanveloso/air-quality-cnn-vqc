"""Fixed validation/test indices; train subsample varies by seed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def _stratified_indices(
    part: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.Index:
    size = min(size, len(part))
    if size >= len(part):
        return part.index
    idx, _ = train_test_split(
        part.index,
        train_size=size,
        stratify=part["target"],
        random_state=seed,
    )
    return pd.Index(idx)


def _rows_from_indices(part: pd.DataFrame, indices: pd.Index, partition: str) -> pd.DataFrame:
    out = part.loc[indices].sort_values("timestamp").copy()
    out = out.reset_index(names="original_index")
    out["partition"] = partition
    return out


def build_fixed_evaluation_sets(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    validation_size: int = 200,
    test_size: int = 300,
    evaluation_seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select validation and test once; reuse across all training seeds."""
    val_idx = _stratified_indices(validation, validation_size, evaluation_seed)
    test_idx = _stratified_indices(test, test_size, evaluation_seed + 1)

    va = _rows_from_indices(validation, val_idx, "validation")
    te = _rows_from_indices(test, test_idx, "test")
    meta = {
        "evaluation_seed": evaluation_seed,
        "validation_size": len(va),
        "test_size": len(te),
        "validation_indices": va["original_index"].tolist(),
        "test_indices": te["original_index"].tolist(),
        "fixed_validation": True,
        "fixed_test": True,
    }
    return va, te, meta


def sample_train_for_seed(
    train: pd.DataFrame,
    train_size: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified training subsample that varies with seed."""
    idx = _stratified_indices(train, train_size, seed)
    return _rows_from_indices(train, idx, "train")


def save_partition_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["original_index", "timestamp", "target", "partition"] if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df[cols + extra].to_csv(path, index=False)


def load_partition_indices(path: str | Path) -> list[Any]:
    df = pd.read_csv(path)
    return df["original_index"].tolist()


class SampleRegistry:
    """Persist fixed val/test and per-seed train indices under artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_fixed(
        self,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        save_partition_csv(validation, self.root / "validation_fixed.csv")
        save_partition_csv(test, self.root / "test_fixed.csv")

    def save_train(self, train: pd.DataFrame, seed: int, train_size: int | None = None) -> None:
        suffix = f"train_seed_{seed}"
        if train_size is not None:
            suffix = f"train_seed_{seed}_n{train_size}"
        save_partition_csv(train, self.root / f"{suffix}.csv")

    def load_fixed_indices(self) -> tuple[list[Any], list[Any]]:
        val = load_partition_indices(self.root / "validation_fixed.csv")
        test = load_partition_indices(self.root / "test_fixed.csv")
        return val, test
