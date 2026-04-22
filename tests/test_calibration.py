"""Smoke tests for Phase-7 calibration module."""

from __future__ import annotations

import numpy as np
import pytest

from readmit_bench.calibration.calibrate import (
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    _grouped_stratified_half_split,
    fit_and_select_calibrator,
    pick_cost_threshold,
)


@pytest.fixture
def synthetic_scores():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.binomial(1, 0.1, size=n)
    # Make scores correlated with y but a bit miscalibrated (logit shift).
    raw = rng.normal(loc=y * 1.5 - 3.0, scale=1.0)
    p = 1 / (1 + np.exp(-raw))
    return y, p


def test_split_is_patient_grouped_and_disjoint():
    rng = np.random.default_rng(7)
    n = 2000
    groups = rng.integers(0, 400, size=n)
    y = rng.binomial(1, 0.1, size=n)
    calib_mask, test_mask = _grouped_stratified_half_split(y, groups, seed=1)
    assert calib_mask.sum() + test_mask.sum() == n
    assert not np.any(calib_mask & test_mask)
    calib_pats = set(groups[calib_mask].tolist())
    test_pats = set(groups[test_mask].tolist())
    assert calib_pats.isdisjoint(test_pats), "patients leaked across calib/test"


def test_split_preserves_positive_rate():
    rng = np.random.default_rng(11)
    n = 5000
    groups = rng.integers(0, 1500, size=n)
    y = (rng.random(n) < 0.12).astype(int)
    calib_mask, test_mask = _grouped_stratified_half_split(y, groups, seed=1)
    assert abs(y[calib_mask].mean() - y[test_mask].mean()) < 0.03


def test_calibrator_selection_returns_valid_object(synthetic_scores):
    y, p = synthetic_scores
    half = len(y) // 2
    name, cal, results, p_best = fit_and_select_calibrator(p[:half], y[:half], p[half:], y[half:])
    assert name in {"uncalibrated", "isotonic", "platt"}
    # The chosen method must equal argmin Brier
    expected = min(results, key=lambda k: results[k].brier)
    assert name == expected
    out = cal.predict(p[half:])
    assert out.shape == p[half:].shape
    assert np.all((out >= 0) & (out <= 1))
    np.testing.assert_allclose(out, p_best)


def test_cost_threshold_minimises_objective(synthetic_scores):
    y, p = synthetic_scores
    t, cost, grid, costs, conf = pick_cost_threshold(y, p, n_grid=201)
    assert 0.0 <= t <= 1.0
    assert cost == pytest.approx(costs.min())
    # Cost at chosen threshold must be no worse than cost at always-treat.
    cost_always = DEFAULT_COST_FP * len(y)
    cost_never = DEFAULT_COST_FN * int(y.sum())
    assert cost <= cost_always + 1e-6
    assert cost <= cost_never + 1e-6
    assert sum(conf.values()) == len(y)


def test_cost_threshold_extremes_are_correct():
    # All negatives, threshold sweep should pick a high threshold (treat ~nobody).
    y = np.zeros(100, dtype=int)
    p = np.linspace(0.0, 0.99, 100)
    t, cost, grid, costs, conf = pick_cost_threshold(y, p, n_grid=21)
    assert cost == 0.0
    assert conf["fn"] == 0
    assert conf["tp"] == 0
    assert t >= 0.99
