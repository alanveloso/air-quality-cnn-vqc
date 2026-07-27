"""Tests for train-only angular scaling."""

from __future__ import annotations

import numpy as np
import pytest

from qml_air_quality.preprocessing.angular_scaling import (
    MinMaxAngularScaler,
    NoneAngularScaler,
    QuantileAngularScaler,
    make_angular_scaler,
)


def test_none_scaler_identity():
    X = np.array([[1.0, -2.0], [3.0, 4.0], [0.5, 0.0]])
    s = NoneAngularScaler().fit(X)
    assert np.allclose(s.transform(X), X)


def test_minmax_fit_only_on_train_range():
    train = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    test = np.array([[3.0, 6.0], [-1.0, -2.0]])  # outside train range
    s = MinMaxAngularScaler(0.0, np.pi, name="minmax_0_pi").fit(train)
    Xt = s.transform(train)
    assert Xt.min() >= -1e-9
    assert Xt.max() <= np.pi + 1e-9
    Xte = s.transform(test)
    # values outside train get extrapolated beyond [0, pi] by MinMax — allowed,
    # but scaler parameters must come from train only
    assert hasattr(s._scaler, "data_min_")
    assert np.allclose(s._scaler.data_min_, train.min(axis=0))
    assert np.allclose(s._scaler.data_max_, train.max(axis=0))
    assert Xte.shape == test.shape


def test_quantile_maps_to_0_pi():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 2))
    s = QuantileAngularScaler(n_quantiles=50, seed=0).fit(train)
    Xt = s.transform(train)
    assert Xt.min() >= -1e-6
    assert Xt.max() <= np.pi + 1e-6


@pytest.mark.parametrize(
    "name",
    ["none", "minmax_0_1", "minmax_0_pi", "minmax_minus_pi_pi", "quantile_0_pi"],
)
def test_factory_names(name):
    X = np.arange(20, dtype=float).reshape(10, 2)
    s = make_angular_scaler(name, seed=1)
    out = s.fit_transform(X)
    assert out.shape == X.shape
    assert np.all(np.isfinite(out))


def test_no_leakage_refit_changes_params():
    """Fitting on different trains must yield different scaler params."""
    a = np.array([[0.0], [1.0], [2.0]])
    b = np.array([[10.0], [20.0], [30.0]])
    sa = MinMaxAngularScaler(0.0, 1.0).fit(a)
    sb = MinMaxAngularScaler(0.0, 1.0).fit(b)
    assert not np.allclose(sa._scaler.data_min_, sb._scaler.data_min_)
    # transforming the same point differs
    x = np.array([[1.0]])
    assert not np.allclose(sa.transform(x), sb.transform(x))
