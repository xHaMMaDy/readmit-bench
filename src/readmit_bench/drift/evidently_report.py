"""Phase-15 — train-vs-val drift report (Evidently 0.7.x).

Compares the training cohort against a holdout (default = val split) and
writes an HTML report + a compact JSON summary suitable for monitoring.

Run:
    python -m readmit_bench.drift.evidently_report
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import polars as pl
from evidently import Report
from evidently.presets import DataDriftPreset

from readmit_bench.features import FeatureSpec

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_HTML = Path("reports/drift_report.html")
DEFAULT_JSON = Path("reports/drift_summary.json")


def _load(features_path: Path, split: str, sample: int | None = None):
    spec = FeatureSpec()
    cols = list(spec.all_features()) + ["y"]
    df = pl.read_parquet(features_path).filter(pl.col("split") == split).select(cols)
    if sample is not None and df.height > sample:
        df = df.sample(n=sample, seed=42)
    pdf = df.to_pandas()
    # Evidently 0.7 is brittle with pandas extension dtypes (Int8/Int64/boolean
    # all break in different ways). Force every column to a plain numpy dtype:
    # numerics → float64 (NaNs allowed), everything else → object str.
    for c in pdf.columns:
        s = pdf[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            pdf[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            # Replace pandas <NA> with numpy NaN to keep .eq() comparisons safe.
            ser = s.astype("string")
            pdf[c] = ser.where(ser.notna(), None).astype("object")
    return pdf


def run(
    features_path: Path = DEFAULT_FEATURES,
    html_out: Path = DEFAULT_HTML,
    json_out: Path = DEFAULT_JSON,
    sample: int = 50000,
) -> dict:
    logger.info("loading reference (train) and current (val) cohorts")
    ref = _load(features_path, "train", sample=sample)
    cur = _load(features_path, "val", sample=sample)
    logger.info(
        "reference rows=%d  current rows=%d  features=%d", len(ref), len(cur), ref.shape[1] - 1
    )

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)

    html_out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(html_out.as_posix())
    logger.info("wrote HTML drift report → %s", html_out)

    payload = snapshot.dict()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=str))

    # Surface a one-line headline.
    drifted = 0
    total = 0
    for m in payload.get("metrics", []):
        if "DriftedColumnsCount" in str(m.get("metric_id", "")):
            v = m.get("value", {}) if isinstance(m.get("value"), dict) else {}
            drifted = int(v.get("count", 0))
            total = int(v.get("share", 0) and v.get("count", 0) / v["share"] or 0)
            break
    logger.info("drift summary: %d / %s columns drifted", drifted, str(total or "?"))
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    p.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    p.add_argument("--sample", type=int, default=50000)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    args = _parse_args()
    run(
        features_path=args.features,
        html_out=args.html_out,
        json_out=args.json_out,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()
