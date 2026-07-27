"""Feature engineering and target creation (no future leakage)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour/month encodings from timestamp."""
    out = df.copy()
    hour = out["timestamp"].dt.hour
    month = out["timestamp"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def add_lag_and_rolling(
    df: pd.DataFrame,
    columns: list[str],
    lag_hours: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Causal lags and rolling stats (past-only windows)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(col)
        for lag in lag_hours:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for w in rolling_windows:
            # shift(1) so window ends at t-1 … actually plan says value at t is allowed
            # rolling mean of last w hours including current: rolling(w) is causal
            out[f"{col}_roll_mean_{w}"] = out[col].rolling(window=w, min_periods=1).mean()
            out[f"{col}_roll_std_{w}"] = out[col].rolling(window=w, min_periods=1).std()
    return out


def add_future_target(
    df: pd.DataFrame,
    target_column: str = "PM2.5",
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Add future_pm25 = target shifted -horizon (label only; not a feature)."""
    out = df.copy()
    out["future_pm25"] = out[target_column].shift(-horizon_hours)
    out["target_timestamp"] = out["timestamp"] + pd.Timedelta(hours=horizon_hours)
    return out


def apply_causal_ffill(
    df: pd.DataFrame,
    columns: list[str],
    limit_hours: int = 3,
) -> pd.DataFrame:
    """Forward-fill limited to limit_hours (causal)."""
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = out[c].ffill(limit=limit_hours)
    return out


def impute_with_train_medians(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, ...]:
    """Fill remaining NaNs using medians computed only on train."""
    medians = {c: train[c].median() for c in columns if c in train.columns}
    result = []
    for part in (train, *others):
        out = part.copy()
        for c, med in medians.items():
            if c in out.columns:
                out[c] = out[c].fillna(med)
        result.append(out)
    return tuple(result)


def make_binary_target(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    percentile: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Threshold from train future_pm25 only; attach target 0/1."""
    threshold = float(train["future_pm25"].quantile(percentile))
    meta = {
        "target": "PM2.5",
        "horizon_hours": 24,
        "percentile": percentile,
        "threshold": threshold,
        "threshold_source": "training_partition",
    }

    def _apply(part: pd.DataFrame) -> pd.DataFrame:
        out = part.copy()
        out["target"] = (out["future_pm25"] >= threshold).astype(int)
        return out

    return _apply(train), _apply(validation), _apply(test), meta


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns excluding target/future/id leakage names."""
    banned_substrings = ("future", "target", "t_plus_24")
    ban_exact = {
        "timestamp",
        "target_timestamp",
        "year",
        "month",
        "day",
        "hour",
        "station",
        "No",
        "wd",
        "future_pm25",
        "target",
    }
    cols = []
    for c in df.columns:
        if c in ban_exact:
            continue
        cl = c.lower()
        if any(b in cl for b in banned_substrings):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def save_target_metadata(meta: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
