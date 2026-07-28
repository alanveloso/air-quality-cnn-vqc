"""Temporal block bootstrap for paired model comparisons."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, recall_score


def _block_starts(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample contiguous blocks with replacement covering ~n observations."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(0, n - block_size)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])
    return idx[:n]


def block_bootstrap_deltas(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    pred_a: np.ndarray | None = None,
    pred_b: np.ndarray | None = None,
    timestamps: np.ndarray | pd.Series | None = None,
    n_boot: int = 1000,
    block_size_hours: int = 24,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Block-bootstrap differences (A − B) for ranking and operational metrics.

    Assumes hourly rows so block_size_hours maps 1:1 to consecutive indices
    when timestamps are sorted (caller should pass chronologically ordered data).
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    n = len(y_true)
    if pred_a is None:
        pred_a = (scores_a >= 0.5).astype(int)
    if pred_b is None:
        pred_b = (scores_b >= 0.5).astype(int)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {
        "delta_auprc": [],
        "delta_f2": [],
        "delta_recall": [],
        "delta_false_alert_rate": [],
    }

    for _ in range(n_boot):
        idx = _block_starts(n, block_size_hours, rng)
        y = y_true[idx]
        if len(np.unique(y)) < 2:
            continue
        sa, sb = scores_a[idx], scores_b[idx]
        pa, pb = pred_a[idx], pred_b[idx]
        auprc_a = average_precision_score(y, sa)
        auprc_b = average_precision_score(y, sb)
        deltas["delta_auprc"].append(float(auprc_a - auprc_b))
        f2_a = fbeta_score(y, pa, beta=2, zero_division=0)
        f2_b = fbeta_score(y, pb, beta=2, zero_division=0)
        deltas["delta_f2"].append(float(f2_a - f2_b))
        rec_a = recall_score(y, pa, pos_label=1, zero_division=0)
        rec_b = recall_score(y, pb, pos_label=1, zero_division=0)
        deltas["delta_recall"].append(float(rec_a - rec_b))
        far_a = float(((pa == 1) & (y == 0)).sum() / len(y))
        far_b = float(((pb == 1) & (y == 0)).sum() / len(y))
        deltas["delta_false_alert_rate"].append(far_a - far_b)

    alpha = (1 - confidence) / 2
    out: dict[str, Any] = {
        "n_boot": n_boot,
        "block_size_hours": block_size_hours,
        "n_boot_effective": len(deltas["delta_auprc"]),
        "confidence": confidence,
    }
    for key, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        out[f"{key}_mean"] = float(np.mean(arr)) if len(arr) else float("nan")
        out[f"{key}_std"] = float(np.std(arr)) if len(arr) else float("nan")
        out[f"{key}_ci_low"] = float(np.quantile(arr, alpha)) if len(arr) else float("nan")
        out[f"{key}_ci_high"] = float(np.quantile(arr, 1 - alpha)) if len(arr) else float("nan")
    return out


def point_metric_delta(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: Callable[..., float] | None = None,
) -> float:
    """Convenience for a single A−B AUPRC delta (no bootstrap)."""
    del metric_fn
    return float(
        average_precision_score(y_true, scores_a) - average_precision_score(y_true, scores_b)
    )
