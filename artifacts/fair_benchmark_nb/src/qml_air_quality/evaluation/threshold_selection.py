"""Select decision threshold on validation by F-beta (default F2)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import fbeta_score


def select_threshold_fbeta(
    y_true: np.ndarray,
    scores: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, float, dict[str, Any]]:
    """Pick threshold maximizing F_beta on validation scores.

    Returns (best_threshold, best_fbeta, metadata).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.unique(scores)
    if len(thresholds) == 0:
        raise ValueError("no scores available for threshold selection")

    best_threshold = float(thresholds[0])
    best_f = -1.0
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        f = float(fbeta_score(y_true, preds, beta=beta, pos_label=1, zero_division=0))
        if f > best_f:
            best_f = f
            best_threshold = float(thr)

    meta = {
        "metric": f"f{beta:g}",
        "select_on": "validation",
        "best_threshold": best_threshold,
        "best_fbeta": best_f,
        "n_thresholds_evaluated": len(thresholds),
        "selection_split": "validation",
    }
    return best_threshold, best_f, meta


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores) >= threshold).astype(int)
