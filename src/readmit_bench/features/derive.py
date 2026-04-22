"""Derived features computed on top of the cohort.

Everything here is *encounter-local* (depends only on columns of the index
encounter or pre-computed prior-history features that already respect the
admit_date cut-off). No future leakage.
"""

from __future__ import annotations

import polars as pl

CHRONIC_COLS: tuple[str, ...] = (
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
)


def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add admit_month, admit_dow, is_weekend_admit, chronic_count,
    has_prior_admit, days_since_last_discharge_imputed.

    Produces only encounter-local features (zero leakage).
    """
    chronic_sum = pl.sum_horizontal([pl.col(c).cast(pl.Int8) for c in CHRONIC_COLS])

    return df.with_columns(
        [
            pl.col("admit_date").dt.month().cast(pl.Int8).alias("admit_month"),
            pl.col("admit_date").dt.weekday().cast(pl.Int8).alias("admit_dow"),
            (pl.col("admit_date").dt.weekday() >= 6).alias("is_weekend_admit"),
            chronic_sum.alias("chronic_count"),
            pl.col("days_since_last_discharge").is_not_null().alias("has_prior_admit"),
            pl.col("days_since_last_discharge")
            .fill_null(9999)
            .alias("days_since_last_discharge_imputed"),
        ]
    )
