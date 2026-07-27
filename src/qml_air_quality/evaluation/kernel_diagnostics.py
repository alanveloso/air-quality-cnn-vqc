"""Structural diagnostics for quantum (and classical) kernel matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def _off_diagonal(K: np.ndarray) -> np.ndarray:
    mask = ~np.eye(K.shape[0], dtype=bool)
    return K[mask]


def off_diagonal_stats(K: np.ndarray) -> dict[str, float]:
    off = _off_diagonal(K)
    if off.size == 0:
        return {
            "off_mean": float("nan"),
            "off_std": float("nan"),
            "off_min": float("nan"),
            "off_max": float("nan"),
            "off_p05": float("nan"),
            "off_p25": float("nan"),
            "off_p50": float("nan"),
            "off_p75": float("nan"),
            "off_p95": float("nan"),
            "cv_k": float("nan"),
        }
    mean = float(np.mean(off))
    std = float(np.std(off))
    return {
        "off_mean": mean,
        "off_std": std,
        "off_min": float(np.min(off)),
        "off_max": float(np.max(off)),
        "off_p05": float(np.percentile(off, 5)),
        "off_p25": float(np.percentile(off, 25)),
        "off_p50": float(np.percentile(off, 50)),
        "off_p75": float(np.percentile(off, 75)),
        "off_p95": float(np.percentile(off, 95)),
        "cv_k": float(std / mean) if abs(mean) > 1e-12 else float("nan"),
    }


def class_contrast(K: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """ΔK = μ_same − μ_different on off-diagonal pairs."""
    y = np.asarray(y).ravel()
    n = len(y)
    same_vals: list[float] = []
    diff_vals: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = float(K[i, j])
            if y[i] == y[j]:
                same_vals.append(v)
            else:
                diff_vals.append(v)
    mu_same = float(np.mean(same_vals)) if same_vals else float("nan")
    mu_diff = float(np.mean(diff_vals)) if diff_vals else float("nan")
    return {
        "mu_same": mu_same,
        "mu_different": mu_diff,
        "delta_k": mu_same - mu_diff,
    }


def effective_rank(K: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    """Shannon effective rank of eigenvalue spectrum of symmetric K."""
    # Numerical symmetrization
    Ks = 0.5 * (K + K.T)
    eigvals = np.linalg.eigvalsh(Ks)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= eps:
        return {"effective_rank": 0.0, "n_positive_eigs": 0}
    p = eigvals / total
    p = p[p > eps]
    entropy = float(-np.sum(p * np.log(p)))
    return {
        "effective_rank": float(np.exp(entropy)),
        "n_positive_eigs": int((eigvals > eps).sum()),
        "eig_max": float(eigvals.max()),
        "eig_min_nonneg": float(eigvals[eigvals > eps].min()) if (eigvals > eps).any() else 0.0,
    }


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Frobenius alignment A(K, yy^T) with labels mapped to {-1, +1}."""
    y = np.asarray(y).ravel().astype(float)
    # map {0,1} or other binary → {-1,+1}
    classes = np.unique(y)
    if len(classes) != 2:
        # degenerate
        return float("nan")
    y_pm = np.where(y == classes.max(), 1.0, -1.0)
    Y = np.outer(y_pm, y_pm)
    Ks = 0.5 * (K + K.T)
    num = float(np.sum(Ks * Y))
    den = float(np.linalg.norm(Ks, ord="fro") * np.linalg.norm(Y, ord="fro"))
    if den < 1e-12:
        return float("nan")
    return num / den


def diagnose_kernel(K: np.ndarray, y: np.ndarray | None = None) -> dict[str, Any]:
    """Aggregate diagnostics for a square kernel matrix."""
    K = np.asarray(K, dtype=float)
    out: dict[str, Any] = {
        "shape": list(K.shape),
        "symmetric": bool(np.allclose(K, K.T, atol=1e-6)),
        "diag_mean": float(np.mean(np.diag(K))),
        "finite": bool(np.all(np.isfinite(K))),
    }
    out.update(off_diagonal_stats(K))
    out.update(effective_rank(K))
    if y is not None:
        out.update(class_contrast(K, y))
        out["alignment"] = kernel_target_alignment(K, y)
    return out
