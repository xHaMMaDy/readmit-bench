"""Cohort selection + 30-day readmission label engineering for CMS DE-SynPUF.

This script consumes the raw CSVs under ``data/raw/csv/`` and produces a single
encounter-level parquet at ``data/processed/cohort.parquet``. All decisions
follow ``CONTRACT.md`` (Phase 0).

Pipeline
--------
1. Read all 20 inpatient CSVs lazily with Polars; concat.
2. Parse YYYYMMDD integer dates → date32; compute LOS.
3. Sort by (beneficiary, admit_date) and chronologically lag → next admit / prior discharge.
4. Read all beneficiary files (one row per beneficiary per year); long-form per year.
5. Join beneficiary attributes to each admit on the year of admit (with fallback
   to nearest available year if a specific year is missing — handles the known
   broken Sample-1 2010 file).
6. Apply cohort exclusions:
     - died during stay
     - died within 30 days of discharge with no observed readmit (censored)
     - LOS > 365 days
     - age < 18  (defensive — DE-SynPUF should be 65+)
7. Compute features known at/before T_admit (no leakage):
     - age_at_admit, age_bin, sex, race, state_code
     - admit_dx_chapter (ICD-9 first-3-digit grouping)
     - num_diagnoses, num_procedures
     - 11 chronic-condition booleans (from beneficiary file of admit-year)
     - prior_6mo_inpatient_count
     - days_since_last_discharge
8. Derive label ``y`` = 1 iff next admit within (discharge, discharge+30d].
9. Write parquet + a small JSON cohort summary for the EDA notebook.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("cohort")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "raw" / "csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"

INPATIENT_GLOB = "DE1_0_2008_to_2010_Inpatient_Claims_Sample_*.csv"
BENE_GLOB = "DE1_0_*_Beneficiary_Summary_File_Sample_*.csv"

# Inpatient columns we actually keep (drops 81→ ~25)
INP_KEEP = [
    "DESYNPUF_ID",
    "CLM_ID",
    "CLM_FROM_DT",
    "CLM_THRU_DT",
    "PRVDR_NUM",
    "CLM_ADMSN_DT",
    "ADMTNG_ICD9_DGNS_CD",
    "CLM_DRG_CD",
    "CLM_UTLZTN_DAY_CNT",
    "NCH_BENE_DSCHRG_DT",
    *[f"ICD9_DGNS_CD_{i}" for i in range(1, 11)],
    *[f"ICD9_PRCDR_CD_{i}" for i in range(1, 7)],
]

CHRONIC_COLS = [
    "SP_ALZHDMTA",  # Alzheimer / related dementia
    "SP_CHF",  # Heart failure
    "SP_CHRNKIDN",  # Chronic kidney disease
    "SP_CNCR",  # Cancer
    "SP_COPD",  # COPD
    "SP_DEPRESSN",  # Depression
    "SP_DIABETES",  # Diabetes
    "SP_ISCHMCHT",  # Ischemic heart disease
    "SP_OSTEOPRS",  # Osteoporosis
    "SP_RA_OA",  # RA / OA
    "SP_STRKETIA",  # Stroke / TIA
]
CHRONIC_RENAME = {
    "SP_ALZHDMTA": "chronic_alzheimer",
    "SP_CHF": "chronic_chf",
    "SP_CHRNKIDN": "chronic_ckd",
    "SP_CNCR": "chronic_cancer",
    "SP_COPD": "chronic_copd",
    "SP_DEPRESSN": "chronic_depression",
    "SP_DIABETES": "chronic_diabetes",
    "SP_ISCHMCHT": "chronic_ihd",
    "SP_OSTEOPRS": "chronic_osteoporosis",
    "SP_RA_OA": "chronic_ra_oa",
    "SP_STRKETIA": "chronic_stroke",
}

RACE_MAP = {1: "White", 2: "Black", 3: "Other", 5: "Hispanic"}
SEX_MAP = {1: "Male", 2: "Female"}


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
def parse_yyyymmdd(col: str) -> pl.Expr:
    """DE-SynPUF dates are int YYYYMMDD or empty. → polars Date (null on missing)."""
    return (
        pl.col(col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_date(format="%Y%m%d", strict=False)
        .alias(col)
    )


# ---------------------------------------------------------------------------
# Diagnosis-chapter mapping
# ---------------------------------------------------------------------------
ICD9_CHAPTERS = [
    (1, 139, "Infectious"),
    (140, 239, "Neoplasms"),
    (240, 279, "Endocrine_Metabolic"),
    (280, 289, "Blood"),
    (290, 319, "Mental"),
    (320, 389, "Nervous_Sense"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy"),
    (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"),
    (740, 759, "Congenital"),
    (760, 779, "Perinatal"),
    (780, 799, "Symptoms_Illdefined"),
    (800, 999, "Injury_Poisoning"),
]


def icd9_to_chapter_expr(col: str) -> pl.Expr:
    """Map ICD-9 string code → chapter name (handles E/V codes & nulls)."""
    s = pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars()
    first3 = s.str.slice(0, 3)
    starts_e = s.str.starts_with("E")
    starts_v = s.str.starts_with("V")

    # Build a chained when/then for the numeric chapters.
    expr = pl.when(starts_e).then(pl.lit("External"))
    expr = expr.when(starts_v).then(pl.lit("Health_Factors"))
    for lo, hi, name in ICD9_CHAPTERS:
        n = first3.cast(pl.Int32, strict=False)
        expr = expr.when((n >= lo) & (n <= hi)).then(pl.lit(name))
    return expr.otherwise(pl.lit("Unknown")).alias("admit_dx_chapter")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_inpatient(csv_dir: Path) -> pl.DataFrame:
    files = sorted(csv_dir.glob(INPATIENT_GLOB))
    if not files:
        raise FileNotFoundError(f"No inpatient CSVs in {csv_dir}")
    logger.info("Loading %d inpatient CSVs ...", len(files))
    parts = []
    for f in files:
        sample = int(f.stem.rsplit("_", 1)[-1])
        df = pl.read_csv(f, infer_schema_length=0)  # all str → safe for date ints
        parts.append(df.select(INP_KEEP).with_columns(pl.lit(sample).alias("sample")))
    df = pl.concat(parts, how="vertical_relaxed")
    logger.info("  → %d raw inpatient claims", df.height)
    return df


def load_beneficiary(csv_dir: Path) -> pl.DataFrame:
    """Long-form beneficiary table: one row per (beneficiary, year)."""
    files = sorted(csv_dir.glob(BENE_GLOB))
    if not files:
        raise FileNotFoundError(f"No beneficiary CSVs in {csv_dir}")
    logger.info("Loading %d beneficiary CSVs (across years/samples) ...", len(files))
    parts = []
    for f in files:
        # filename pattern: DE1_0_<year>_Beneficiary_Summary_File_Sample_<n>.csv
        toks = f.stem.split("_")
        year = int(toks[2])
        sample = int(toks[-1])
        keep_cols = [
            "DESYNPUF_ID",
            "BENE_BIRTH_DT",
            "BENE_DEATH_DT",
            "BENE_SEX_IDENT_CD",
            "BENE_RACE_CD",
            "SP_STATE_CODE",
            *CHRONIC_COLS,
        ]
        df = pl.read_csv(f, infer_schema_length=0).select(keep_cols)
        df = df.with_columns(
            pl.lit(year).cast(pl.Int32).alias("bene_year"),
            pl.lit(sample).cast(pl.Int32).alias("sample"),
        )
        parts.append(df)
    df = pl.concat(parts, how="vertical_relaxed")
    logger.info("  → %d beneficiary-year rows", df.height)
    return df


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def build_cohort(csv_dir: Path) -> tuple[pl.DataFrame, dict]:
    inp = load_inpatient(csv_dir)
    bene = load_beneficiary(csv_dir)

    logger.info("Parsing dates + computing intermediates ...")
    inp = inp.with_columns(
        parse_yyyymmdd("CLM_FROM_DT"),
        parse_yyyymmdd("CLM_THRU_DT"),
        parse_yyyymmdd("CLM_ADMSN_DT"),
        parse_yyyymmdd("NCH_BENE_DSCHRG_DT"),
    )
    # Use ADMSN/DSCHRG when present, else fall back to CLM_FROM/THRU.
    inp = inp.with_columns(
        pl.coalesce("CLM_ADMSN_DT", "CLM_FROM_DT").alias("admit_date"),
        pl.coalesce("NCH_BENE_DSCHRG_DT", "CLM_THRU_DT").alias("discharge_date"),
    )
    inp = inp.with_columns(
        (pl.col("discharge_date") - pl.col("admit_date")).dt.total_days().cast(pl.Int32).alias("los_days"),
        pl.col("admit_date").dt.year().alias("admit_year").cast(pl.Int32),
    )

    # Drop any row with missing admit/discharge.
    pre_n = inp.height
    inp = inp.drop_nulls(subset=["admit_date", "discharge_date"])
    logger.info("  dropped %d rows with missing admit/discharge dates", pre_n - inp.height)

    # Number of dx / procedures
    dx_cols = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]
    pr_cols = [f"ICD9_PRCDR_CD_{i}" for i in range(1, 7)]
    inp = inp.with_columns(
        pl.sum_horizontal(
            [pl.col(c).is_not_null().cast(pl.Int32) for c in dx_cols]
        ).alias("num_diagnoses"),
        pl.sum_horizontal(
            [pl.col(c).is_not_null().cast(pl.Int32) for c in pr_cols]
        ).alias("num_procedures"),
        icd9_to_chapter_expr("ADMTNG_ICD9_DGNS_CD"),
    )

    logger.info("Sorting and computing per-beneficiary lag/lead features ...")
    inp = inp.sort(["DESYNPUF_ID", "admit_date", "discharge_date"])

    inp = inp.with_columns(
        pl.col("admit_date").shift(-1).over("DESYNPUF_ID").alias("next_admit_date"),
        pl.col("discharge_date").shift(1).over("DESYNPUF_ID").alias("prev_discharge_date"),
    )
    inp = inp.with_columns(
        (pl.col("next_admit_date") - pl.col("discharge_date")).dt.total_days().alias("days_to_next_admit"),
        (pl.col("admit_date") - pl.col("prev_discharge_date")).dt.total_days().cast(pl.Int32).alias("days_since_last_discharge"),
    )

    # Prior 6-month inpatient count (per beneficiary, exclusive of current).
    # Polars rolling time window grouped by beneficiary → count, then subtract 1
    # for the current row. Join back on (id, admit_date) since the rolling op
    # returns one row per input row but reorders.
    logger.info("Computing prior 6-month inpatient count (per beneficiary) ...")
    prior_counts = (
        inp.select(["DESYNPUF_ID", "admit_date"])
        .sort(["DESYNPUF_ID", "admit_date"])
        .rolling(index_column="admit_date", period="180d", group_by="DESYNPUF_ID")
        .agg(pl.len().alias("_inp_in_window_inclusive"))
    )
    # Same beneficiary can have multiple admits on the same day → keep the
    # MAX count for that (id, admit_date) so duplicates inherit the same value.
    prior_counts = (
        prior_counts.group_by(["DESYNPUF_ID", "admit_date"])
        .agg(pl.col("_inp_in_window_inclusive").max())
    )
    inp = inp.join(prior_counts, on=["DESYNPUF_ID", "admit_date"], how="left")
    inp = inp.with_columns(
        (pl.col("_inp_in_window_inclusive") - 1).cast(pl.Int32).alias("prior_6mo_inpatient_count")
    ).drop("_inp_in_window_inclusive")

    # ----- Label -----
    logger.info("Building 30-day readmission label ...")
    inp = inp.with_columns(
        pl.when(pl.col("days_to_next_admit").is_not_null() & (pl.col("days_to_next_admit") > 0) & (pl.col("days_to_next_admit") <= 30))
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("y")
    )

    # ----- Beneficiary join (on year of admit, with fallback) -----
    logger.info("Joining beneficiary attributes (year-of-admit with fallback) ...")
    bene = bene.with_columns(
        parse_yyyymmdd("BENE_BIRTH_DT"),
        parse_yyyymmdd("BENE_DEATH_DT"),
        pl.col("BENE_SEX_IDENT_CD").cast(pl.Int32, strict=False),
        pl.col("BENE_RACE_CD").cast(pl.Int32, strict=False),
        pl.col("SP_STATE_CODE").cast(pl.Int32, strict=False),
        *[
            (pl.col(c).cast(pl.Int32, strict=False) == 1).alias(CHRONIC_RENAME[c])
            for c in CHRONIC_COLS
        ],
    )

    # exact join on (id, year, sample); then fallback for rows without a match
    bene_keep = ["DESYNPUF_ID", "bene_year", "sample", "BENE_BIRTH_DT", "BENE_DEATH_DT",
                 "BENE_SEX_IDENT_CD", "BENE_RACE_CD", "SP_STATE_CODE",
                 *CHRONIC_RENAME.values()]
    bene_small = bene.select(bene_keep)

    joined = inp.join(
        bene_small,
        left_on=["DESYNPUF_ID", "admit_year", "sample"],
        right_on=["DESYNPUF_ID", "bene_year", "sample"],
        how="left",
    )

    # Fallback: where BENE_BIRTH_DT is null after the exact join, take the most
    # recent beneficiary record for that (id, sample) in any year.
    missing_mask = joined["BENE_BIRTH_DT"].is_null()
    n_missing = int(missing_mask.sum())
    if n_missing:
        logger.info(
            "  %d encounters had no exact-year beneficiary match (e.g. Sample 1 2010) — using nearest-year fallback",
            n_missing,
        )
        # Most recent year per (id, sample) regardless of admit year.
        fallback = (
            bene_small.sort("bene_year", descending=True)
            .group_by(["DESYNPUF_ID", "sample"])
            .agg(pl.all().first())
            .drop("bene_year")
        )
        # Drop the originally-joined null bene cols, then re-join.
        bene_value_cols = [c for c in bene_small.columns if c not in ("DESYNPUF_ID", "bene_year", "sample")]
        joined_filled = joined.filter(~missing_mask)
        joined_to_fill = joined.filter(missing_mask).drop(bene_value_cols)
        joined_to_fill = joined_to_fill.join(
            fallback,
            on=["DESYNPUF_ID", "sample"],
            how="left",
        )
        joined = pl.concat([joined_filled, joined_to_fill], how="vertical_relaxed")

    # Age + sex/race human labels + age_bin
    logger.info("Computing age + categorical labels ...")
    joined = joined.with_columns(
        ((pl.col("admit_date") - pl.col("BENE_BIRTH_DT")).dt.total_days() / 365.25)
        .floor()
        .cast(pl.Int32, strict=False)
        .alias("age_at_admit"),
        pl.col("BENE_SEX_IDENT_CD").replace_strict(SEX_MAP, default="Unknown").alias("sex"),
        pl.col("BENE_RACE_CD").replace_strict(RACE_MAP, default="Unknown").alias("race"),
    )
    joined = joined.with_columns(
        pl.when(pl.col("age_at_admit") < 65).then(pl.lit("<65"))
        .when(pl.col("age_at_admit") < 75).then(pl.lit("65-74"))
        .when(pl.col("age_at_admit") < 85).then(pl.lit("75-84"))
        .otherwise(pl.lit("85+"))
        .alias("age_bin")
    )

    # ----- Cohort exclusions -----
    logger.info("Applying cohort exclusions ...")
    n_total = joined.height

    # Right-censoring at end of observation window: an encounter discharged within
    # 30 days of the last observed admit date in the dataset cannot have its
    # 30-day readmission outcome fully observed. Drop these to avoid an
    # artifactual decline in label rate near the data window's tail.
    max_obs_date = joined.select(pl.col("admit_date").max()).item()
    censor_cutoff = max_obs_date  # discharge must be <= max_obs_date - 30d
    logger.info("  observation window ends %s — dropping discharges within 30d of cutoff", max_obs_date)

    death = pl.col("BENE_DEATH_DT")
    died_during = (
        death.is_not_null()
        & (death >= pl.col("admit_date"))
        & (death <= pl.col("discharge_date"))
    )
    censored_by_death = (
        death.is_not_null()
        & (death > pl.col("discharge_date"))
        & (death <= pl.col("discharge_date").dt.offset_by("30d"))
        & (pl.col("y") == 0)
    )
    censored_by_window = (
        pl.col("discharge_date").dt.offset_by("30d") > pl.lit(censor_cutoff)
    )
    long_los = pl.col("los_days") > 365
    too_young = pl.col("age_at_admit") < 18

    excl = joined.with_columns(
        died_during.alias("_excl_died_during"),
        censored_by_death.alias("_excl_censored_death"),
        censored_by_window.alias("_excl_censored_window"),
        long_los.alias("_excl_long_los"),
        too_young.alias("_excl_too_young"),
    )
    drop_mask = (
        excl["_excl_died_during"]
        | excl["_excl_censored_death"]
        | excl["_excl_censored_window"]
        | excl["_excl_long_los"]
        | excl["_excl_too_young"]
    )
    cohort = excl.filter(~drop_mask).drop(
        [
            "_excl_died_during",
            "_excl_censored_death",
            "_excl_censored_window",
            "_excl_long_los",
            "_excl_too_young",
        ]
    )

    n_after = cohort.height
    logger.info(
        "  cohort: %d → %d (dropped %d : died_during=%d censored_death=%d censored_window=%d longLOS=%d young=%d)",
        n_total,
        n_after,
        n_total - n_after,
        int(excl["_excl_died_during"].sum()),
        int(excl["_excl_censored_death"].sum()),
        int(excl["_excl_censored_window"].sum()),
        int(excl["_excl_long_los"].sum()),
        int(excl["_excl_too_young"].sum()),
    )

    # ----- Final shape -----
    final_cols = [
        "DESYNPUF_ID",
        "CLM_ID",
        "sample",
        "admit_date",
        "discharge_date",
        "admit_year",
        "los_days",
        "age_at_admit",
        "age_bin",
        "sex",
        "race",
        "SP_STATE_CODE",
        "admit_dx_chapter",
        "ADMTNG_ICD9_DGNS_CD",
        "CLM_DRG_CD",
        "num_diagnoses",
        "num_procedures",
        *CHRONIC_RENAME.values(),
        "prior_6mo_inpatient_count",
        "days_since_last_discharge",
        "y",
    ]
    cohort = cohort.select(final_cols).rename(
        {
            "DESYNPUF_ID": "beneficiary_id",
            "CLM_ID": "claim_id",
            "SP_STATE_CODE": "state_code",
            "ADMTNG_ICD9_DGNS_CD": "admit_dx_code",
            "CLM_DRG_CD": "drg_code",
        }
    )

    # ----- Summary stats -----
    pos = int(cohort["y"].sum())
    summary = {
        "n_encounters": cohort.height,
        "n_beneficiaries": cohort["beneficiary_id"].n_unique(),
        "positive_count": pos,
        "positive_rate": round(pos / cohort.height, 6) if cohort.height else 0.0,
        "n_samples_present": int(cohort["sample"].n_unique()),
        "admit_year_range": [int(cohort["admit_year"].min()), int(cohort["admit_year"].max())],
        "median_los": float(cohort["los_days"].median()),
        "median_age": float(cohort["age_at_admit"].median()),
        "n_dropped_total": n_total - n_after,
    }
    logger.info("Final cohort summary: %s", json.dumps(summary, indent=2))
    return cohort, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build inpatient readmission cohort.")
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort, summary = build_cohort(args.csv_dir)

    out_pq = args.out_dir / "cohort.parquet"
    cohort.write_parquet(out_pq, compression="zstd")
    logger.info("Wrote %s (%.1f MB)", out_pq, out_pq.stat().st_size / (1024 * 1024))

    out_json = args.out_dir / "cohort_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
