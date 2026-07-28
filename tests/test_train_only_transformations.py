"""Transformers must be fit only on the training partition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qml_air_quality.paired_representation import fit_paired_representation


def test_transformers_fit_only_on_train():
    rng = np.random.default_rng(1)
    cols = [f"f{i}" for i in range(8)]

    def _part(n: int, start: str) -> pd.DataFrame:
        data = {c: rng.normal(size=n) for c in cols}
        data["timestamp"] = pd.date_range(start, periods=n, freq="h")
        data["target"] = rng.integers(0, 2, size=n)
        return pd.DataFrame(data)

    train, val, test = _part(60, "2015-01-01"), _part(30, "2015-02-01"), _part(30, "2015-03-01")
    repr_ = fit_paired_representation(train, val, test, cols, pca_components=2, seed=0)
    for record in repr_.fit_records:
        assert record.fit_partition == "train"

    _, _, _, ang, ang_record = repr_.angular("minmax_0_pi", seed=0)
    assert ang_record.fit_partition == "train"
    # Angular scaler parameters come from train PCA only
    X_tr = repr_.X_train_pca
    assert hasattr(ang, "n_features_in_")
    assert ang.n_features_in_ == X_tr.shape[1]
