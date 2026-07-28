"""F2 threshold selection on validation scores."""

from __future__ import annotations

import numpy as np

from qml_air_quality.evaluation.threshold_selection import (
    apply_threshold,
    select_threshold_fbeta,
)
from qml_air_quality.experiments.model_selection import freeze_selection


def test_threshold_selection_prefers_recall_weighted_f2():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Higher scores for positives but overlapping
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.55, 0.7, 0.8, 0.9])
    thr, f2, meta = select_threshold_fbeta(y, scores, beta=2.0)
    assert meta["select_on"] == "validation"
    assert meta["selection_split"] == "validation"
    assert meta["metric"] == "f2"
    preds = apply_threshold(scores, thr)
    assert preds.shape == y.shape
    assert f2 >= 0.0


def test_test_not_used_for_selection_metadata():
    selected = {
        "model_family": "logistic",
        "C": 1.0,
        "class_weight": "balanced",
        "val_auprc": 0.4,
        "val_recall_extreme": 0.5,
        "selection_split": "validation",
        "model": object(),
        "val_scores": np.array([0.1, 0.9]),
        "threshold": 0.5,
        "threshold_meta": {"selection_split": "validation"},
    }
    frozen = freeze_selection(selected)
    assert frozen["selection_split"] == "validation"
    assert "model" not in frozen
    assert frozen["frozen"] is True
