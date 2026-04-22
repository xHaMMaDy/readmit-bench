"""Smoke + correctness tests for Phase-9 fairness audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from readmit_bench.fairness.audit import (
    SENSITIVE_ATTRS,
    _parity_gaps,
    _per_attribute_table,
    _slice_metrics,
)


@pytest.fixture()
def synthetic_run():
    rng = np.random.default_rng(0)
    n = 4000
    sex = rng.choice(["Female", "Male"], size=n, p=[0.55, 0.45])
    race = rng.choice(["White", "Black", "Hispanic", "Other"], size=n, p=[0.7, 0.15, 0.10, 0.05])
    age_bin = rng.choice(["<65", "65-74", "75-84", "85+"], size=n, p=[0.2, 0.35, 0.3, 0.15])
    A = pd.DataFrame({"sex": sex, "race": race, "age_bin": age_bin})
    y = rng.binomial(1, 0.10, size=n).astype(int)
    base = rng.beta(1.5, 12, size=n)
    p = np.clip(base + 0.25 * y + rng.normal(0, 0.02, size=n), 1e-4, 1 - 1e-4)
    yhat = (p >= 0.05).astype(int)
    return y, p, yhat, A


def test_slice_metrics_overall_consistent(synthetic_run):
    y, p, yhat, A = synthetic_run
    m = _slice_metrics(y, p, yhat)
    assert m["n"] == len(y)
    assert m["n_pos"] == int(y.sum())
    assert 0.0 <= m["selection_rate"] <= 1.0
    assert 0.0 <= m["fnr_at_t"] <= 1.0
    assert 0.0 < m["pr_auc"] < 1.0


def test_per_attribute_table_has_overall_and_all_slices(synthetic_run):
    y, p, yhat, A = synthetic_run
    table = _per_attribute_table(y, p, yhat, A)
    for attr in SENSITIVE_ATTRS:
        sub = table[table["attribute"] == attr]
        assert "OVERALL" in sub["slice"].values
        # every observed value must show up
        for v in A[attr].unique():
            assert str(v) in sub["slice"].values


def test_overall_fnr_matches_population_fnr(synthetic_run):
    y, p, yhat, A = synthetic_run
    table = _per_attribute_table(y, p, yhat, A)
    pop_fnr = ((yhat == 0) & (y == 1)).sum() / max(1, y.sum())
    for attr in SENSITIVE_ATTRS:
        row = table[(table["attribute"] == attr) & (table["slice"] == "OVERALL")].iloc[0]
        assert pytest.approx(row["fnr_at_t"], rel=1e-9) == pop_fnr


def test_parity_gaps_have_expected_keys_and_bounds(synthetic_run):
    y, p, yhat, A = synthetic_run
    gaps = _parity_gaps(y, p, yhat, A)
    assert set(gaps.keys()) == set(SENSITIVE_ATTRS)
    for _attr, g in gaps.items():
        for k in (
            "demographic_parity_difference",
            "equalized_odds_difference",
            "fnr_max",
            "fnr_min",
            "fnr_gap",
            "worst_fnr_slice",
            "best_fnr_slice",
        ):
            assert k in g
        assert 0.0 <= g["demographic_parity_difference"] <= 1.0
        assert 0.0 <= g["equalized_odds_difference"] <= 1.0
        assert g["fnr_gap"] == pytest.approx(g["fnr_max"] - g["fnr_min"], abs=1e-12)
        assert g["fnr_gap"] >= 0.0


def test_actual_audit_artifacts_exist_and_consistent():
    """Sanity check that the full pipeline output is internally consistent."""
    summary = Path("reports/fairness_summary.csv")
    gaps_path = Path("reports/fairness_gaps.json")
    if not summary.exists() or not gaps_path.exists():
        pytest.skip(
            "Phase-9 outputs not present (run `python -m readmit_bench.fairness.audit` first)"
        )
    table = pd.read_csv(summary)
    gaps = json.loads(gaps_path.read_text())
    for attr in SENSITIVE_ATTRS:
        sub = table[(table["attribute"] == attr) & (table["slice"] != "OVERALL")]
        observed_max = sub["fnr_at_t"].max()
        observed_min = sub["fnr_at_t"].min()
        assert pytest.approx(gaps[attr]["fnr_max"], abs=1e-9) == observed_max
        assert pytest.approx(gaps[attr]["fnr_min"], abs=1e-9) == observed_min
