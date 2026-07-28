"""Hyperparameter and quantum-config selection on validation only."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

from qml_air_quality.evaluation.threshold_selection import select_threshold_fbeta
from qml_air_quality.models import (
    make_fidelity_kernel,
    make_precomputed_svm,
    predict_proba_positive,
)


def _resolve_class_weight(value: Any) -> str | None:
    if value is None or value == "null" or value == "None":
        return None
    return value


def ranking_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Continuous scores for AUPRC / threshold search."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    proba = predict_proba_positive(model, X)
    if proba is None:
        raise RuntimeError("model cannot produce ranking scores")
    return np.asarray(proba, dtype=float)


def _score_candidate(
    y_val: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    auprc = float(average_precision_score(y_val, scores))
    _, _, meta = select_threshold_fbeta(y_val, scores, beta=2.0)
    # temporary F2 after best threshold for tie-break context; recall used later
    thr = meta["best_threshold"]
    preds = (scores >= thr).astype(int)
    recall = float(((preds == 1) & (y_val == 1)).sum() / max(1, int((y_val == 1).sum())))
    return auprc, recall


def select_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_grid: list[float],
    class_weight_grid: list[Any],
    seed: int = 42,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for C, cw in itertools.product(C_grid, class_weight_grid):
        model = LogisticRegression(
            C=C,
            class_weight=_resolve_class_weight(cw),
            max_iter=2000,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        scores = ranking_scores(model, X_val)
        auprc, recall = _score_candidate(y_val, scores)
        cand = {
            "model_family": "logistic",
            "C": C,
            "class_weight": cw,
            "val_auprc": auprc,
            "val_recall_extreme": recall,
            "selection_split": "validation",
        }
        if best is None or (auprc, recall) > (best["val_auprc"], best["val_recall_extreme"]):
            best = cand
            best["model"] = model
            best["val_scores"] = scores
    assert best is not None
    thr, f2, tmeta = select_threshold_fbeta(y_val, best["val_scores"], beta=2.0)
    best["threshold"] = thr
    best["val_f2"] = f2
    best["threshold_meta"] = tmeta
    return best


def select_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    kernel: str,
    C_grid: list[float],
    class_weight_grid: list[Any],
    gamma_grid: list[Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    gamma_grid = gamma_grid or ["scale"]
    best: dict[str, Any] | None = None
    for C, cw, gamma in itertools.product(C_grid, class_weight_grid, gamma_grid):
        kwargs: dict[str, Any] = {
            "kernel": kernel,
            "C": C,
            "class_weight": _resolve_class_weight(cw),
            "random_state": seed,
        }
        if kernel == "rbf":
            kwargs["gamma"] = gamma
        model = SVC(**kwargs)
        model.fit(X_train, y_train)
        scores = ranking_scores(model, X_val)
        auprc, recall = _score_candidate(y_val, scores)
        cand = {
            "model_family": f"svm_{kernel}",
            "C": C,
            "class_weight": cw,
            "gamma": gamma if kernel == "rbf" else None,
            "val_auprc": auprc,
            "val_recall_extreme": recall,
            "selection_split": "validation",
        }
        if best is None or (auprc, recall) > (best["val_auprc"], best["val_recall_extreme"]):
            best = cand
            best["model"] = model
            best["val_scores"] = scores
    assert best is not None
    thr, f2, tmeta = select_threshold_fbeta(y_val, best["val_scores"], beta=2.0)
    best["threshold"] = thr
    best["val_f2"] = f2
    best["threshold_meta"] = tmeta
    return best


def select_qsvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    C_grid: list[float],
    class_weight_grid: list[Any],
    reps: int,
    feature_map: str = "ZZFeatureMap",
    entanglement: str = "linear",
    seed: int = 42,
    kernel_cache: dict[str, np.ndarray] | None = None,
    cache_key: str | None = None,
) -> dict[str, Any]:
    n_qubits = X_train.shape[1]
    if kernel_cache is not None and cache_key and f"{cache_key}_train" in kernel_cache:
        K_tr = kernel_cache[f"{cache_key}_train"]
        K_va = kernel_cache[f"{cache_key}_val"]
    else:
        qk = make_fidelity_kernel(
            n_qubits=n_qubits,
            reps=reps,
            entanglement=entanglement,
            feature_map_name=feature_map,
            enforce_psd=True,
        )
        K_tr = qk.evaluate(x_vec=X_train)
        K_va = qk.evaluate(x_vec=X_val, y_vec=X_train)
        if kernel_cache is not None and cache_key:
            kernel_cache[f"{cache_key}_train"] = K_tr
            kernel_cache[f"{cache_key}_val"] = K_va

    best: dict[str, Any] | None = None
    for C, cw in itertools.product(C_grid, class_weight_grid):
        model = make_precomputed_svm(
            seed=seed,
            C=C,
            class_weight=_resolve_class_weight(cw),
        )
        model.fit(K_tr, y_train)
        scores = ranking_scores(model, K_va)
        auprc, recall = _score_candidate(y_val, scores)
        cand = {
            "model_family": "qsvm",
            "C": C,
            "class_weight": cw,
            "reps": reps,
            "feature_map": feature_map,
            "entanglement": entanglement,
            "val_auprc": auprc,
            "val_recall_extreme": recall,
            "selection_split": "validation",
        }
        if best is None or (auprc, recall) > (best["val_auprc"], best["val_recall_extreme"]):
            best = cand
            best["model"] = model
            best["val_scores"] = scores
            best["K_train"] = K_tr
            best["K_val"] = K_va
    assert best is not None
    thr, f2, tmeta = select_threshold_fbeta(y_val, best["val_scores"], beta=2.0)
    best["threshold"] = thr
    best["val_f2"] = f2
    best["threshold_meta"] = tmeta
    return best


def freeze_selection(selected: dict[str, Any]) -> dict[str, Any]:
    """Drop fitted objects; keep hyperparameters + threshold for final stage."""
    skip = {"model", "val_scores", "K_train", "K_val", "threshold_meta"}
    out = {k: v for k, v in selected.items() if k not in skip}
    out["selection_split"] = "validation"
    out["frozen"] = True
    return out
