"""Tuning module smoke tests — fast, no Optuna optimisation loop, no MLflow."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from readmit_bench.features import FeatureSpec
from readmit_bench.tuning.optimize import _build_pipeline, _grouped_stratified_subsample
from readmit_bench.tuning.spaces import (
    SPACES,
)


def _fake_features(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Tiny synthetic features matching FeatureSpec column groups."""
    rng = np.random.default_rng(seed)
    spec = FeatureSpec()
    cols = list(spec.all_features())
    data = {}
    for c in spec.numeric:
        data[c] = rng.normal(0, 1, size=n)
    for c in spec.binary:
        data[c] = rng.integers(0, 2, size=n).astype("int8")
    for c in spec.cat_lowcard:
        data[c] = rng.choice(["A", "B", "C", "D"], size=n)
    for c in spec.cat_highcard:
        data[c] = rng.choice([f"L{i:04d}" for i in range(50)], size=n)
    df = pd.DataFrame(data)[cols]
    y = rng.binomial(1, 0.10, size=n).astype(int)
    groups = rng.integers(0, n // 3, size=n)
    return df, y, groups


def test_spaces_register_all_three():
    assert set(SPACES.keys()) == {"catboost", "xgboost", "hist_gradient_boosting"}


def test_spaces_emit_dicts():
    """Each space function returns a dict of valid hyperparameters from a real trial."""
    for name, fn in SPACES.items():
        study = optuna.create_study()
        trial = study.ask()
        params = fn(trial)
        assert isinstance(params, dict) and params, name


def test_subsample_returns_smaller_and_keeps_groups_intact():
    X, y, g = _fake_features(n=2000)
    Xs, ys, gs = _grouped_stratified_subsample(X, y, g, n_target=400)
    assert len(Xs) <= len(X)
    assert len(Xs) == len(ys) == len(gs)
    # Every group in subsample must appear fully (or not at all) — no row leak.
    chosen = set(np.unique(gs))
    for grp in chosen:
        full = (g == grp).sum()
        kept = (gs == grp).sum()
        assert full == kept, f"group {grp}: full={full} kept={kept}"


def test_build_pipeline_hist_gb_includes_densify():
    space = SPACES["hist_gradient_boosting"]
    study = optuna.create_study()
    trial = study.ask()
    params = space(trial)
    pipe = _build_pipeline("hist_gradient_boosting", params)
    step_names = [s[0] for s in pipe.steps]
    assert "preprocess" in step_names
    assert "densify" in step_names
    assert "model" in step_names


def test_build_pipeline_catboost_no_densify():
    space = SPACES["catboost"]
    study = optuna.create_study()
    trial = study.ask()
    params = space(trial)
    pipe = _build_pipeline("catboost", params)
    step_names = [s[0] for s in pipe.steps]
    assert "preprocess" in step_names
    assert "densify" not in step_names
    assert "model" in step_names
