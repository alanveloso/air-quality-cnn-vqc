"""Temporal purge and target-timestamp leakage checks."""

from __future__ import annotations

import pandas as pd

from qml_air_quality.features import add_future_target, make_binary_target
from qml_air_quality.split import apply_temporal_purge, temporal_split


def _synthetic_hourly(n: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2015-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "PM2.5": [10.0 + (i % 50) for i in range(n)],
            "TEMP": [20.0] * n,
        }
    )


def test_feature_and_target_timestamps():
    df = add_future_target(_synthetic_hourly(48), horizon_hours=24)
    assert "feature_timestamp" in df.columns
    assert (df["feature_timestamp"] == df["timestamp"]).all()
    assert (df["target_timestamp"] == df["timestamp"] + pd.Timedelta(hours=24)).all()


def test_temporal_purge():
    df = add_future_target(_synthetic_hourly(300), horizon_hours=24)
    df = df.dropna(subset=["future_pm25"]).reset_index(drop=True)
    train, validation, test = temporal_split(df, 0.6, 0.2, 0.2)
    train_p, val_p, test_p, meta = apply_temporal_purge(train, validation, test, purge_hours=24)

    assert train_p["target_timestamp"].max() < validation["timestamp"].min()
    assert val_p["target_timestamp"].max() < test["timestamp"].min()
    assert meta["train_rows_removed"] >= 1
    assert meta["validation_rows_removed"] >= 1
    assert len(test_p) == len(test)


def test_target_threshold_from_purged_train():
    df = add_future_target(_synthetic_hourly(300), horizon_hours=24)
    df = df.dropna(subset=["future_pm25"]).reset_index(drop=True)
    train, validation, test = temporal_split(df, 0.6, 0.2, 0.2)
    train, validation, test, _ = apply_temporal_purge(train, validation, test)
    train, validation, test, meta = make_binary_target(
        train,
        validation,
        test,
        percentile=0.90,
        source="purged_training_partition",
    )
    assert meta["source"] == "purged_training_partition"
    expected = float(train["future_pm25"].quantile(0.90))
    assert meta["threshold"] == expected
    assert set(train["target"].unique()) <= {0, 1}
