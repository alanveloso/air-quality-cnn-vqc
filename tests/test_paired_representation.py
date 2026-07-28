"""Paired representation: identical inputs for classical and quantum arms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qml_air_quality.paired_representation import fit_paired_representation


def _frames(n: int = 80) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(0)
    cols = [
        "pm25_min",
        "pm25_max",
        "pm25_median",
        "pm25_variance",
        "temperature_min",
        "temperature_max",
        "temperature_median",
        "temperature_variance",
    ]

    def _one(start: str, size: int) -> pd.DataFrame:
        data = {c: rng.normal(size=size) for c in cols}
        data["timestamp"] = pd.date_range(start, periods=size, freq="h")
        data["target"] = rng.integers(0, 2, size=size)
        return pd.DataFrame(data)

    return _one("2015-01-01", n), _one("2015-02-01", 40), _one("2015-03-01", 40), cols


def test_same_paired_inputs():
    train, val, test, cols = _frames()
    repr_ = fit_paired_representation(train, val, test, cols, pca_components=2, seed=7)
    X_tr_q, X_va_q, X_te_q, _, _ = repr_.angular("minmax_0_1", seed=7)
    X_tr_c, X_va_c, X_te_c, _, _ = repr_.angular("minmax_0_1", seed=7)
    assert np.array_equal(X_te_q, X_te_c)
    assert np.array_equal(X_tr_q, X_tr_c)
    assert np.array_equal(X_va_q, X_va_c)
    assert repr_.X_train_full.shape[1] == 8
    assert repr_.X_train_pca.shape[1] == 2


def test_pca_explained_variance_present():
    train, val, test, cols = _frames()
    repr_ = fit_paired_representation(train, val, test, cols, pca_components=2)
    assert repr_.explained_variance_ratio is not None
    assert len(repr_.explained_variance_ratio) == 2
