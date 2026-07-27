"""Tests for ablation helpers (no full quantum runtime)."""

from __future__ import annotations

import numpy as np

from qml_air_quality.experiments.quantum_ablation import (
    bootstrap_auprc_delta,
    classify_result,
    rank_key,
)
from qml_air_quality.preprocessing.angular_scaling import make_angular_scaler


def test_rank_prefers_higher_auprc():
    a = {"average_precision": 0.4, "recall_extreme": 0.1, "alignment": 0.0, "kernel_seconds": 10}
    b = {"average_precision": 0.2, "recall_extreme": 0.9, "alignment": 0.9, "kernel_seconds": 1}
    assert rank_key(a) < rank_key(b)


def test_classify_not_informative():
    out = classify_result(
        delta_mean=-0.1,
        ci_low=-0.2,
        ci_high=-0.05,
        min_delta=0.02,
        q_auprc=0.105,
        prevalence=0.10,
        alignment=0.01,
        delta_k=0.0,
    )
    assert out["label"] == "QUANTUM_KERNEL_NOT_INFORMATIVE"


def test_classify_candidate_gain():
    out = classify_result(
        delta_mean=0.05,
        ci_low=0.01,
        ci_high=0.09,
        min_delta=0.02,
        q_auprc=0.4,
        prevalence=0.1,
        alignment=0.2,
        delta_k=0.1,
    )
    assert out["label"] == "CANDIDATE_PREDICTIVE_QUANTUM_GAIN"


def test_bootstrap_runs():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 10)
    sa = rng.random(50)
    sb = rng.random(50)
    boot = bootstrap_auprc_delta(y, sa, sb, n_boot=50, seed=0)
    assert "ci_low" in boot and "ci_high" in boot


def test_angular_then_shape_preserved():
    X = np.arange(30, dtype=float).reshape(10, 3)
    for name in ["none", "minmax_0_pi", "quantile_0_pi"]:
        s = make_angular_scaler(name)
        assert s.fit_transform(X).shape == X.shape
