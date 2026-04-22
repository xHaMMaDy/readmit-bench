"""End-to-end feature build: cohort -> derived features -> grouped split.

Outputs (all parquet, gitignored):
    data/processed/features.parquet  -- cohort + derived columns + `split` col
    data/processed/splits.parquet    -- thin (claim_id, beneficiary_id, split)
    reports/split_summary.csv        -- counts + positive rates per split

Run:
    python -m readmit_bench.features.build
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from .derive import add_derived_features
from .pipeline import FeatureSpec
from .split import (
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    assign_splits,
    summarise_split,
    write_split_assignments,
)

logger = logging.getLogger(__name__)

DEFAULT_COHORT = Path("data/processed/cohort.parquet")
DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_SPLITS = Path("data/processed/splits.parquet")
DEFAULT_SUMMARY = Path("reports/split_summary.csv")


def build_features(
    cohort_path: Path = DEFAULT_COHORT,
    features_path: Path = DEFAULT_FEATURES,
    splits_path: Path = DEFAULT_SPLITS,
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"cohort not found: {cohort_path}. Run `python -m readmit_bench.data.cohort` first."
        )

    logger.info("loading cohort from %s", cohort_path)
    cohort = pl.read_parquet(cohort_path)
    logger.info("cohort loaded: %d rows x %d cols", cohort.height, cohort.width)

    logger.info("adding derived features")
    feats = add_derived_features(cohort)

    logger.info("assigning grouped + stratified split (seed=%d)", seed)
    feats = assign_splits(feats, test_size=test_size, val_size=val_size, seed=seed)

    spec = FeatureSpec()
    keep_cols = [*spec.id_cols, *spec.all_features(), spec.label, "split"]
    missing = [c for c in keep_cols if c not in feats.columns]
    if missing:
        raise RuntimeError(f"derived feature missing expected columns: {missing}")
    feats = feats.select(keep_cols)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(features_path)
    logger.info("wrote features to %s (%d rows)", features_path, feats.height)

    write_split_assignments(feats, splits_path)

    report = summarise_split(feats)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report.as_table().write_csv(summary_path)
    logger.info("wrote split summary to %s", summary_path)
    logger.info("\n%s", report.as_table())

    return feats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--splits-out", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    build_features(
        cohort_path=args.cohort,
        features_path=args.features_out,
        splits_path=args.splits_out,
        summary_path=args.summary_out,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
