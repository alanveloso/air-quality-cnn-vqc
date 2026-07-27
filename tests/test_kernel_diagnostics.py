"""Tests for kernel diagnostics."""

from __future__ import annotations

import numpy as np

from qml_air_quality.evaluation.kernel_diagnostics import (
    class_contrast,
    diagnose_kernel,
    effective_rank,
    kernel_target_alignment,
    off_diagonal_stats,
)


def test_perfect_alignment_block_kernel():
    # Two class-pure blocks with high within similarity
    K = np.array(
        [
            [1, 0.9, 0.1, 0.1],
            [0.9, 1, 0.1, 0.1],
            [0.1, 0.1, 1, 0.9],
            [0.1, 0.1, 0.9, 1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1])
    a = kernel_target_alignment(K, y)
    assert a > 0.5
    c = class_contrast(K, y)
    assert c["delta_k"] > 0


def test_constant_kernel_low_contrast():
    K = np.ones((6, 6))
    y = np.array([0, 0, 0, 1, 1, 1])
    c = class_contrast(K, y)
    assert abs(c["delta_k"]) < 1e-9
    stats = off_diagonal_stats(K)
    assert abs(stats["off_std"]) < 1e-9


def test_effective_rank_identity():
    K = np.eye(5)
    er = effective_rank(K)
    assert abs(er["effective_rank"] - 5.0) < 1e-6


def test_diagnose_bundle():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 20))
    K = X @ X.T
    # normalize diag
    d = np.sqrt(np.diag(K))
    K = K / np.outer(d, d)
    y = (rng.random(20) > 0.5).astype(int)
    out = diagnose_kernel(K, y)
    assert out["symmetric"]
    assert "alignment" in out
    assert "effective_rank" in out
