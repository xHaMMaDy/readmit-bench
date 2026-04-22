"""Smoke tests for Phase-8 SHAP module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmit_bench.explainability.plots import (
    _pretty,
    plot_global_importance,
)
from readmit_bench.explainability.shap_runner import (
    _grouped_stratified_sample,
    compute_shap_values,
)


def test_pretty_strips_known_prefixes():
    assert _pretty("num__los_days") == "los_days"
    assert _pretty("bin__chronic_copd") == "chronic_copd"
    assert _pretty("cat_low__state_code_9") == "state_code_9"
    assert _pretty("cat_high__admit_dx_code") == "admit_dx_code"
    assert _pretty("no_prefix") == "no_prefix"


def test_grouped_stratified_sample_no_patient_leakage():
    rng = np.random.default_rng(3)
    n = 2000
    groups = rng.integers(0, 400, size=n)
    y = rng.binomial(1, 0.1, size=n)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    Xs, ys = _grouped_stratified_sample(X, y, groups, n_target=400, seed=1)
    assert len(Xs) == len(ys) >= 200  # patient grouping makes exact n_target unlikely


def test_grouped_stratified_sample_returns_full_when_small():
    n = 50
    X = pd.DataFrame({"a": np.arange(n)})
    y = np.zeros(n, dtype=int)
    g = np.arange(n)
    Xs, ys = _grouped_stratified_sample(X, y, g, n_target=200)
    assert len(Xs) == n


def test_global_importance_plot_renders(tmp_path):
    top = pd.DataFrame(
        {
            "feature": [f"num__feat_{i}" for i in range(25)],
            "mean_abs_shap": np.linspace(0.5, 0.01, 25),
            "mean_signed_shap": np.linspace(0.4, -0.4, 25),
        }
    )
    out = plot_global_importance(top, tmp_path, n_top=20)
    assert out.exists()
    assert out.stat().st_size > 5000  # non-trivial PNG


def test_compute_shap_with_tiny_xgb():
    """End-to-end shape check using a tiny XGBClassifier wrapped in a sklearn Pipeline."""
    pytest.importorskip("xgboost")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = (X["a"] + 0.5 * X["b"] + rng.normal(size=n) > 0).astype(int).to_numpy()
    pre = ColumnTransformer([("num", StandardScaler(), ["a", "b", "c"])])
    pipe = Pipeline(
        [
            ("preprocess", pre),
            ("model", XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss")),
        ]
    )
    pipe.fit(X, y)
    shap_vals, X_proc, names, base = compute_shap_values(pipe, X.head(50))
    assert shap_vals.shape == (50, 3)
    assert X_proc.shape == (50, 3)
    assert len(names) == 3
    assert isinstance(base, float)
