"""Train-only angular scalers for quantum feature maps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


class AngularScaler(ABC, BaseEstimator, TransformerMixin):
    """Common interface: fit on train only, transform partitions."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: Any = None) -> AngularScaler:
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class NoneAngularScaler(AngularScaler):
    """Identity — control matching the original PoC (no angular remap)."""

    name = "none"

    def fit(self, X: np.ndarray, y: Any = None) -> NoneAngularScaler:
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)


class MinMaxAngularScaler(AngularScaler):
    """MinMax to a chosen angular range, fit on train only."""

    def __init__(self, minimum: float = 0.0, maximum: float = np.pi, name: str | None = None):
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.name = name or f"minmax_{minimum}_{maximum}"
        self._scaler = MinMaxScaler(feature_range=(self.minimum, self.maximum))

    def fit(self, X: np.ndarray, y: Any = None) -> MinMaxAngularScaler:
        self._scaler.fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(X)


class QuantileAngularScaler(AngularScaler):
    """QuantileTransformer to uniform [0,1], then map to [0, π]."""

    name = "quantile_0_pi"

    def __init__(self, n_quantiles: int = 100, seed: int = 42):
        self.n_quantiles = n_quantiles
        self.seed = seed
        self._qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            random_state=seed,
            subsample=int(1e9),
        )

    def fit(self, X: np.ndarray, y: Any = None) -> QuantileAngularScaler:
        n = max(10, min(self.n_quantiles, X.shape[0]))
        self._qt.set_params(n_quantiles=n)
        self._qt.fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        u = self._qt.transform(X)
        return np.pi * u


def make_angular_scaler(name: str, seed: int = 42, **kwargs: Any) -> AngularScaler:
    """Factory for named angular strategies from the ablation plan."""
    key = name.lower().strip()
    if key in {"none", "identity", "sem_escala"}:
        return NoneAngularScaler()
    if key in {"minmax_0_1", "minmax_[0,1]", "[0,1]"}:
        return MinMaxAngularScaler(0.0, 1.0, name="minmax_0_1")
    if key in {"minmax_0_pi", "minmax_[0,pi]", "[0,pi]", "minmax_0_π"}:
        return MinMaxAngularScaler(0.0, np.pi, name="minmax_0_pi")
    if key in {
        "minmax_minus_pi_pi",
        "minmax_[-pi,pi]",
        "[-pi,pi]",
        "minmax_minus_π_π",
    }:
        return MinMaxAngularScaler(-np.pi, np.pi, name="minmax_minus_pi_pi")
    if key in {"quantile_0_pi", "quantile_[0,pi]", "quantile"}:
        return QuantileAngularScaler(
            n_quantiles=int(kwargs.get("n_quantiles", 100)),
            seed=seed,
        )
    # allow explicit min/max from yaml
    if "minimum" in kwargs and "maximum" in kwargs:
        return MinMaxAngularScaler(
            float(kwargs["minimum"]),
            float(kwargs["maximum"]),
            name=name,
        )
    raise ValueError(f"Unknown angular scaler: {name!r}")


def save_angular_scaler(scaler: AngularScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_angular_scaler(path: str | Path) -> AngularScaler:
    return joblib.load(path)
