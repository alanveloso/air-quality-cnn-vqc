"""Block bootstrap helpers."""

from __future__ import annotations

import numpy as np

from qml_air_quality.evaluation.block_bootstrap import block_bootstrap_deltas
from qml_air_quality.evaluation.statistical_comparison import (
    classify_fair_result,
    majority_seed_wins,
)


def test_block_bootstrap_runs():
    rng = np.random.default_rng(0)
    n = 120
    y = (rng.random(n) > 0.85).astype(int)
    y[0] = 0
    y[1] = 1
    scores_a = rng.random(n)
    scores_b = rng.random(n)
    out = block_bootstrap_deltas(
        y, scores_a, scores_b, n_boot=50, block_size_hours=24, seed=1
    )
    assert out["n_boot_effective"] > 0
    assert "delta_auprc_mean" in out
    assert "delta_auprc_ci_low" in out


def test_classify_no_clear_gain_when_ci_includes_zero():
    label = classify_fair_result(
        delta_auprc_mean=0.01,
        delta_auprc_ci_low=-0.02,
        delta_f2_mean=0.01,
        wins_majority_seeds=True,
    )
    assert label["label"] == "NO_CLEAR_QUANTUM_GAIN"


def test_classify_candidate_gain():
    label = classify_fair_result(
        delta_auprc_mean=0.05,
        delta_auprc_ci_low=0.01,
        delta_f2_mean=0.0,
        wins_majority_seeds=True,
    )
    assert label["label"] == "CANDIDATE_QUANTUM_GAIN"


def test_majority_seed_wins():
    assert majority_seed_wins([0.1, 0.2, -0.01, 0.05, 0.0]) is True
    assert majority_seed_wins([-0.1, -0.2, 0.01]) is False
