"""Group-stratified 70/15/15 train/val/test split for readmit-bench.

Why grouped: the same beneficiary appears in many encounters; if rows from one
patient leak across splits, the model memorises patient-specific patterns and
the held-out metric is optimistic. We split *beneficiaries*, then assign every
encounter to its beneficiary's split.

Why stratified: positive rate is ~9.6% globally; we want every split to look
the same so PR-AUC / recall numbers are comparable. We stratify on a
per-beneficiary "ever-positive" flag (1 if any encounter for that patient was
a 30-day readmit, else 0).

Why not time-based: Phase 3 EDA confirmed a DE-SynPUF synthesis artefact —
positive rate decays from ~14% (2008) to ~4% (2010). A time-based hold-out
would conflate model drift with synthetic-data drift and report artificially
poor recall on the test fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15


@dataclass(frozen=True)
class SplitReport:
    n_benef_train: int
    n_benef_val: int
    n_benef_test: int
    n_enc_train: int
    n_enc_val: int
    n_enc_test: int
    pos_rate_train: float
    pos_rate_val: float
    pos_rate_test: float

    def as_table(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "split": ["train", "val", "test"],
                "n_beneficiaries": [self.n_benef_train, self.n_benef_val, self.n_benef_test],
                "n_encounters": [self.n_enc_train, self.n_enc_val, self.n_enc_test],
                "positive_rate": [self.pos_rate_train, self.pos_rate_val, self.pos_rate_test],
            }
        )


def assign_splits(
    df: pl.DataFrame,
    *,
    group_col: str = "beneficiary_id",
    label_col: str = "y",
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    """Return df with an extra `split` column ∈ {train, val, test}.

    Splits are computed at the *beneficiary* level (no patient appears in two
    splits) and stratified on each beneficiary's ever-positive flag.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not 0 < val_size < 1 - test_size:
        raise ValueError(f"val_size must be in (0, 1 - test_size); got {val_size}")

    benef = df.group_by(group_col).agg(pl.col(label_col).max().alias("ever_pos")).sort(group_col)
    benef_ids = benef[group_col].to_numpy()
    ever_pos = benef["ever_pos"].to_numpy()

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rem_idx, test_idx = next(sss1.split(benef_ids, ever_pos))

    val_size_adj = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adj, random_state=seed)
    train_rel, val_rel = next(sss2.split(benef_ids[rem_idx], ever_pos[rem_idx]))
    train_idx = rem_idx[train_rel]
    val_idx = rem_idx[val_rel]

    label_map = {bid: "train" for bid in benef_ids[train_idx]}
    label_map.update({bid: "val" for bid in benef_ids[val_idx]})
    label_map.update({bid: "test" for bid in benef_ids[test_idx]})

    split_df = pl.DataFrame(
        {
            group_col: list(label_map.keys()),
            "split": list(label_map.values()),
        }
    )
    return df.join(split_df, on=group_col, how="left")


def summarise_split(df: pl.DataFrame, label_col: str = "y") -> SplitReport:
    grouped = (
        df.group_by("split")
        .agg(
            pl.col("beneficiary_id").n_unique().alias("n_benef"),
            pl.len().alias("n_enc"),
            pl.col(label_col).mean().alias("pos_rate"),
        )
        .sort("split")
    )
    by_name = {row["split"]: row for row in grouped.iter_rows(named=True)}
    return SplitReport(
        n_benef_train=by_name["train"]["n_benef"],
        n_benef_val=by_name["val"]["n_benef"],
        n_benef_test=by_name["test"]["n_benef"],
        n_enc_train=by_name["train"]["n_enc"],
        n_enc_val=by_name["val"]["n_enc"],
        n_enc_test=by_name["test"]["n_enc"],
        pos_rate_train=by_name["train"]["pos_rate"],
        pos_rate_val=by_name["val"]["pos_rate"],
        pos_rate_test=by_name["test"]["pos_rate"],
    )


def write_split_assignments(
    df: pl.DataFrame, out_path: Path, *, key_cols=("claim_id", "beneficiary_id")
) -> Path:
    """Persist a thin (key, split) parquet so downstream code can join cheaply."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.select([*key_cols, "split"]).write_parquet(out_path)
    logger.info("wrote split assignments to %s", out_path)
    return out_path
