"""Fixed validation/test across seeds."""

from __future__ import annotations

import pandas as pd

from qml_air_quality.sample_registry import (
    SampleRegistry,
    build_fixed_evaluation_sets,
    sample_train_for_seed,
)


def _partition(n: int, start: str, pos_rate: float = 0.2) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="h")
    target = [1 if i < int(n * pos_rate) else 0 for i in range(n)]
    return pd.DataFrame({"timestamp": ts, "target": target, "x": range(n)})


def test_fixed_evaluation_sets():
    validation = _partition(400, "2015-06-01")
    test = _partition(500, "2015-09-01")
    va_a, te_a, _ = build_fixed_evaluation_sets(
        validation, test, validation_size=200, test_size=300, evaluation_seed=2026
    )
    va_b, te_b, _ = build_fixed_evaluation_sets(
        validation, test, validation_size=200, test_size=300, evaluation_seed=2026
    )
    assert va_a["original_index"].tolist() == va_b["original_index"].tolist()
    assert te_a["original_index"].tolist() == te_b["original_index"].tolist()


def test_fixed_test_across_train_seeds(tmp_path):
    train = _partition(800, "2015-01-01")
    validation = _partition(400, "2015-06-01")
    test = _partition(500, "2015-09-01")
    va, te, _ = build_fixed_evaluation_sets(
        validation, test, validation_size=200, test_size=300, evaluation_seed=2026
    )
    registry = SampleRegistry(tmp_path)
    registry.save_fixed(va, te)

    tr7 = sample_train_for_seed(train, train_size=200, seed=7)
    tr42 = sample_train_for_seed(train, train_size=200, seed=42)
    registry.save_train(tr7, seed=7)
    registry.save_train(tr42, seed=42)

    val_idx_7, test_idx_7 = registry.load_fixed_indices()
    val_idx_42, test_idx_42 = registry.load_fixed_indices()
    assert test_idx_7 == test_idx_42
    assert val_idx_7 == val_idx_42
    assert tr7["original_index"].tolist() != tr42["original_index"].tolist()
