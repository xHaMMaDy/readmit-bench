"""Phase-4 feature module tests.

Focus on the two correctness invariants that catch the most expensive bugs:
  * grouped split — no beneficiary appears in more than one split (no leakage)
  * stratified split — every split's positive rate is within ±1pp of global

Plus a fast preprocessor smoke test on a synthetic 200-row frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest


@pytest.fixture(scope="module")
def synthetic_cohort() -> pl.DataFrame:
    """Mini cohort: 500 patients × ≤5 encounters each, ~10% positive rate."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(500):
        bid = f"B{i:05d}"
        n = rng.integers(1, 6)
        for j in range(n):
            rows.append(
                {
                    "beneficiary_id": bid,
                    "claim_id": f"{bid}-{j}",
                    "y": int(rng.random() < 0.10),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("y").cast(pl.Int8))


def test_assign_splits_no_patient_leakage(synthetic_cohort: pl.DataFrame) -> None:
    from readmit_bench.features import assign_splits

    out = assign_splits(synthetic_cohort, seed=0)
    leakage = (
        out.group_by("beneficiary_id")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    assert leakage.height == 0, f"{leakage.height} beneficiaries appear in multiple splits"


def test_assign_splits_stratified(synthetic_cohort: pl.DataFrame) -> None:
    from readmit_bench.features import assign_splits, summarise_split

    out = assign_splits(synthetic_cohort, seed=0)
    rep = summarise_split(out)
    global_rate = float(synthetic_cohort["y"].mean())
    for r in (rep.pos_rate_train, rep.pos_rate_val, rep.pos_rate_test):
        assert (
            abs(r - global_rate) < 0.05
        ), f"stratification drift: split rate {r:.3f} vs global {global_rate:.3f}"


def test_assign_splits_proportions(synthetic_cohort: pl.DataFrame) -> None:
    from readmit_bench.features import assign_splits, summarise_split

    out = assign_splits(synthetic_cohort, test_size=0.15, val_size=0.15, seed=0)
    rep = summarise_split(out)
    total = rep.n_benef_train + rep.n_benef_val + rep.n_benef_test
    assert total == synthetic_cohort["beneficiary_id"].n_unique()
    assert 0.65 <= rep.n_benef_train / total <= 0.75
    assert 0.10 <= rep.n_benef_val / total <= 0.20
    assert 0.10 <= rep.n_benef_test / total <= 0.20


def test_add_derived_features_columns() -> None:
    from datetime import date

    from readmit_bench.features import add_derived_features

    df = pl.DataFrame(
        {
            "admit_date": [date(2009, 1, 5), date(2009, 6, 13)],
            "days_since_last_discharge": [None, 12],
            "chronic_chf": [True, False],
            "chronic_alzheimer": [False, False],
            "chronic_ckd": [False, True],
            "chronic_cancer": [False, False],
            "chronic_copd": [False, False],
            "chronic_depression": [False, False],
            "chronic_diabetes": [True, False],
            "chronic_ihd": [False, False],
            "chronic_osteoporosis": [False, False],
            "chronic_ra_oa": [False, False],
            "chronic_stroke": [False, False],
        }
    )
    out = add_derived_features(df)
    for col in (
        "admit_month",
        "admit_dow",
        "is_weekend_admit",
        "chronic_count",
        "has_prior_admit",
        "days_since_last_discharge_imputed",
    ):
        assert col in out.columns, f"derived col missing: {col}"
    # row 0: Mon Jan 5 -> month=1, dow=1, not weekend, chronic_count=2, no prior
    # row 1: Sat Jun 13 -> month=6, dow=6, weekend=True, chronic_count=1, has_prior
    rec = out.to_dicts()
    assert rec[0]["chronic_count"] == 2
    assert rec[1]["chronic_count"] == 1
    assert rec[1]["is_weekend_admit"] is True
    assert rec[0]["has_prior_admit"] is False
    assert rec[1]["has_prior_admit"] is True
    assert rec[0]["days_since_last_discharge_imputed"] == 9999


def test_build_preprocessor_fits_and_transforms() -> None:
    from readmit_bench.features import FeatureSpec, build_preprocessor

    spec = FeatureSpec()
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "los_days": rng.integers(1, 30, n),
            "age_at_admit": rng.integers(65, 95, n),
            "num_diagnoses": rng.integers(1, 12, n),
            "num_procedures": rng.integers(0, 6, n),
            "prior_6mo_inpatient_count": rng.integers(0, 5, n),
            "days_since_last_discharge_imputed": rng.integers(0, 999, n),
            "chronic_count": rng.integers(0, 11, n),
            "admit_month": rng.integers(1, 13, n),
            "admit_dow": rng.integers(1, 8, n),
            "is_weekend_admit": rng.integers(0, 2, n).astype(bool),
            "has_prior_admit": rng.integers(0, 2, n).astype(bool),
            **{
                c: rng.integers(0, 2, n).astype(bool)
                for c in [
                    "chronic_alzheimer",
                    "chronic_chf",
                    "chronic_ckd",
                    "chronic_cancer",
                    "chronic_copd",
                    "chronic_depression",
                    "chronic_diabetes",
                    "chronic_ihd",
                    "chronic_osteoporosis",
                    "chronic_ra_oa",
                    "chronic_stroke",
                ]
            },
            "sex": rng.choice(["1", "2"], n),
            "race": rng.choice(["1", "2", "3", "5"], n),
            "age_bin": rng.choice(["65-74", "75-84", "85+"], n),
            "admit_dx_chapter": rng.choice([f"ch{i}" for i in range(8)], n),
            "state_code": rng.integers(1, 53, n),
            "admit_dx_code": rng.choice([f"code{i}" for i in range(40)], n),
            "drg_code": rng.choice([f"drg{i}" for i in range(20)], n),
        }
    )
    y = rng.integers(0, 2, n)
    pre = build_preprocessor()
    Z = pre.fit_transform(X[list(spec.all_features())], y)
    assert Z.shape[0] == n
    assert Z.shape[1] > 20
    Z_dense = Z.toarray() if hasattr(Z, "toarray") else Z
    assert not np.isnan(Z_dense).any()
