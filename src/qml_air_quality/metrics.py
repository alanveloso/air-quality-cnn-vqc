"""Evaluation metrics for binary extreme-event classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    model_name: str = "",
    training_seconds: float | None = None,
    inference_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute primary (AUPRC) and secondary metrics."""
    out: dict[str, Any] = {
        "model": model_name,
        "average_precision": float(average_precision_score(y_true, y_proba if y_proba is not None else y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_extreme": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_extreme": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_extreme": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["auroc"] = float("nan")
    if training_seconds is not None:
        out["training_seconds"] = training_seconds
    if inference_seconds is not None:
        out["inference_seconds"] = inference_seconds
    return out


def metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("average_precision", ascending=False).reset_index(drop=True)
