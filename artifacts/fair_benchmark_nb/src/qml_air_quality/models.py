"""Classical and quantum model builders for the notebook PoC."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def make_dummy(strategy: str = "prior", seed: int = 42) -> DummyClassifier:
    return DummyClassifier(strategy=strategy, random_state=seed)


def make_logistic(
    seed: int = 42,
    max_iter: int = 2000,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
    )


def make_linear_svm(
    seed: int = 42,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    calibrated: bool = True,
) -> Any:
    base = SVC(kernel="linear", C=C, class_weight=class_weight, random_state=seed)
    if not calibrated:
        return base
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def make_rbf_svm(
    seed: int = 42,
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: str | dict | None = "balanced",
    calibrated: bool = True,
) -> Any:
    base = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight, random_state=seed)
    if not calibrated:
        return base
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def make_precomputed_svm(
    seed: int = 42,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
) -> SVC:
    return SVC(kernel="precomputed", C=C, class_weight=class_weight, random_state=seed)


def make_poly_svm(seed: int = 42, degree: int = 3) -> CalibratedClassifierCV:
    base = SVC(kernel="poly", degree=degree, class_weight="balanced", random_state=seed)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def make_knn(n_neighbors: int = 5) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")


def make_fidelity_kernel(
    n_qubits: int = 4,
    reps: int = 2,
    entanglement: str = "linear",
    feature_map_name: str = "ZZFeatureMap",
    enforce_psd: bool = True,
) -> Any:
    """Build a FidelityQuantumKernel with ZZ or Z feature map."""
    from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    name = feature_map_name.lower().replace("_", "")
    if name in {"zfeaturemap", "z"}:
        feature_map = ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    else:
        # ZZFeatureMap
        ent = None if entanglement in {"none", "None", None} else entanglement
        kwargs: dict[str, Any] = {
            "feature_dimension": n_qubits,
            "reps": reps,
        }
        if ent is not None:
            kwargs["entanglement"] = ent
        feature_map = ZZFeatureMap(**kwargs)

    try:
        from qiskit_machine_learning.state_fidelities import ComputeUncompute

        fidelity = ComputeUncompute(sampler=StatevectorSampler())
        return FidelityQuantumKernel(
            feature_map=feature_map,
            fidelity=fidelity,
            enforce_psd=enforce_psd,
        )
    except TypeError:
        return FidelityQuantumKernel(feature_map=feature_map, enforce_psd=enforce_psd)


def make_qsvc(
    n_qubits: int = 4,
    reps: int = 2,
    entanglement: str = "linear",
    feature_map_name: str = "ZZFeatureMap",
    enforce_psd: bool = True,
) -> tuple[Any, Any]:
    """Build FidelityQuantumKernel + QSVC. Returns (qsvc, quantum_kernel)."""
    from qiskit_machine_learning.algorithms import QSVC

    quantum_kernel = make_fidelity_kernel(
        n_qubits=n_qubits,
        reps=reps,
        entanglement=entanglement,
        feature_map_name=feature_map_name,
        enforce_psd=enforce_psd,
    )
    try:
        qsvc = QSVC(quantum_kernel=quantum_kernel, class_weight="balanced")
    except TypeError:
        qsvc = QSVC(quantum_kernel=quantum_kernel)
    return qsvc, quantum_kernel


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray | None:
    """Return P(class=1) if available, else a ranked score from decision_function."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    return proba[:, classes.index(1)]
            return proba[:, -1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return None
