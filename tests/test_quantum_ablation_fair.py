"""Quantum ablation matrix for the fair benchmark (scale × reps)."""

from __future__ import annotations

import numpy as np

from qml_air_quality.experiments.fair_benchmark import ANGULAR_ALIASES, QUANTUM_CONFIGS
from qml_air_quality.experiments.model_selection import select_qsvm
from qml_air_quality.preprocessing.angular_scaling import make_angular_scaler


def test_quantum_ablation_four_configs():
    assert len(QUANTUM_CONFIGS) == 4
    keys = {(c["angular_scaler"], c["reps"]) for c in QUANTUM_CONFIGS}
    assert keys == {
        ("minmax_0_1", 1),
        ("minmax_0_1", 2),
        ("minmax_0_pi", 1),
        ("minmax_0_pi", 2),
    }


def test_q02_is_farooq_style():
    q02 = next(c for c in QUANTUM_CONFIGS if c["id"] == "Q02")
    assert q02["angular_scaler"] == "minmax_0_1"
    assert q02["reps"] == 2
    assert q02["farooq_style"] is True
    assert sum(1 for c in QUANTUM_CONFIGS if c["farooq_style"]) == 1


def test_angular_aliases_cover_both_scales():
    assert set(ANGULAR_ALIASES) == {"minmax_0_1", "minmax_0_pi"}


def test_qsvm_selection_tiny():
    """Smoke: QSVM grid + F2 threshold on a tiny synthetic PCA-2 set."""
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(20, 2))
    X_va = rng.normal(size=(12, 2))
    y_tr = np.array([0] * 10 + [1] * 10)
    y_va = np.array([0] * 6 + [1] * 6)
    ang = make_angular_scaler("minmax_0_1")
    X_tr = ang.fit_transform(X_tr)
    X_va = ang.transform(X_va)
    sel = select_qsvm(
        X_tr,
        y_tr,
        X_va,
        y_va,
        C_grid=[1.0],
        class_weight_grid=["balanced"],
        reps=1,
        seed=0,
    )
    assert sel["selection_split"] == "validation"
    assert "threshold" in sel
    assert sel["K_train"].shape == (20, 20)
