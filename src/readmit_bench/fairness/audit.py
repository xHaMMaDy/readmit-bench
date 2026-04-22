"""Phase-9 — Fairness audit (slice metrics + Fairlearn parity gaps).

Reuses the **exact** Phase-7 patient-grouped val/test split and the deployed
threshold ``t* = 0.0320`` so audit numbers match what the model would actually
do in production.

Per-slice metrics for each sensitive attribute (sex, race, age_bin):
    n, n_pos, prevalence, PR-AUC, ROC-AUC,
    recall@top-10%, FNR@t*, FPR@t*, selection_rate@t*, brier

Plus Fairlearn aggregate parity gaps (demographic-parity diff, equalized-odds
diff) computed by ``MetricFrame``.

Outputs
-------
- ``reports/fairness_summary.csv``    -- long-format per-slice metric table
- ``reports/fairness_gaps.json``      -- per-attribute parity gaps + worst slice
- ``mlruns/`` experiment ``readmit-bench-fairness``
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from readmit_bench.calibration.calibrate import (
    _grouped_stratified_half_split,
    _IdentityWrapper,  # noqa: F401  -- needed for joblib unpickling of identity calibrator
    _IsoWrapper,  # noqa: F401
    _PlattWrapper,  # noqa: F401
)
from readmit_bench.evaluation.metrics import _at_top_k
from readmit_bench.features import FeatureSpec

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_WINNER = Path("models/tuned/xgboost.joblib")
DEFAULT_CALIBRATOR = Path("models/winner_calibrator.joblib")
DEFAULT_THRESHOLD = Path("models/winner_threshold.json")
DEFAULT_SUMMARY_OUT = Path("reports/fairness_summary.csv")
DEFAULT_GAPS_OUT = Path("reports/fairness_gaps.json")
DEFAULT_PRED_OUT = Path("reports/fairness_predictions.parquet")
DEFAULT_MLRUNS = Path("mlruns")

EXPERIMENT = "readmit-bench-fairness"
RANDOM_STATE = 42
SENSITIVE_ATTRS = ("sex", "race", "age_bin")
TOP_FRAC = 0.10


# ---------- helpers ----------


def _load_val_with_demographics(
    features_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load val split: feature matrix X, label y, group g, sensitive demographics frame A."""
    spec = FeatureSpec()
    feat_cols = list(spec.all_features())
    df = pl.read_parquet(features_path).filter(pl.col("split") == "val")
    X = df.select(feat_cols).to_pandas()
    y = df["y"].to_numpy().astype(int)
    g = df["beneficiary_id"].to_numpy()
    A = df.select(list(SENSITIVE_ATTRS)).to_pandas()
    logger.info(
        "val: %d × %d  (pos %.3f, %d patients)  ·  sensitive cols: %s",
        X.shape[0],
        X.shape[1],
        y.mean(),
        len(np.unique(g)),
        SENSITIVE_ATTRS,
    )
    return X, y, g, A


def _slice_metrics(y: np.ndarray, p: np.ndarray, yhat: np.ndarray) -> dict:
    """All metrics we report for one slice (or the overall row)."""
    n = int(len(y))
    n_pos = int(y.sum())
    out = {
        "n": n,
        "n_pos": n_pos,
        "prevalence": float(n_pos / n) if n else float("nan"),
        "selection_rate": float(yhat.mean()) if n else float("nan"),
        "brier": float(brier_score_loss(y, p)) if n_pos and n_pos < n else float("nan"),
    }
    if n_pos and n_pos < n:
        out["pr_auc"] = float(average_precision_score(y, p))
        out["roc_auc"] = float(roc_auc_score(y, p))
        recall_k, _ = _at_top_k(y, p, TOP_FRAC)
        out["recall_at_top10"] = float(recall_k)
        out["fnr_at_t"] = float(false_negative_rate(y, yhat))
        out["fpr_at_t"] = float(false_positive_rate(y, yhat))
    else:
        for k in ("pr_auc", "roc_auc", "recall_at_top10", "fnr_at_t", "fpr_at_t"):
            out[k] = float("nan")
    return out


def _per_attribute_table(
    y: np.ndarray, p: np.ndarray, yhat: np.ndarray, A: pd.DataFrame
) -> pd.DataFrame:
    """Long-format slice table for every attribute × value, plus an OVERALL row per attribute."""
    rows: list[dict] = []
    for attr in SENSITIVE_ATTRS:
        col = A[attr].to_numpy()
        rows.append({"attribute": attr, "slice": "OVERALL", **_slice_metrics(y, p, yhat)})
        for val in pd.unique(col):
            mask = col == val
            rows.append(
                {
                    "attribute": attr,
                    "slice": str(val),
                    **_slice_metrics(y[mask], p[mask], yhat[mask]),
                }
            )
    return pd.DataFrame(rows)


def _parity_gaps(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, A: pd.DataFrame) -> dict:
    """Fairlearn aggregate gaps + worst-slice FNR identification per attribute."""
    out: dict = {}
    metric_funcs = {
        "selection_rate": selection_rate,
        "fnr": false_negative_rate,
        "fpr": false_positive_rate,
    }
    for attr in SENSITIVE_ATTRS:
        sf = pd.Series(A[attr].to_numpy(), name=attr)
        mf = MetricFrame(metrics=metric_funcs, y_true=y, y_pred=yhat, sensitive_features=sf)
        per_group = mf.by_group.reset_index().rename(columns={attr: "slice"})
        worst_fnr_row = per_group.iloc[per_group["fnr"].idxmax()]
        best_fnr_row = per_group.iloc[per_group["fnr"].idxmin()]
        out[attr] = {
            "demographic_parity_difference": float(
                demographic_parity_difference(y, yhat, sensitive_features=sf)
            ),
            "equalized_odds_difference": float(
                equalized_odds_difference(y, yhat, sensitive_features=sf)
            ),
            "fnr_max": float(worst_fnr_row["fnr"]),
            "fnr_min": float(best_fnr_row["fnr"]),
            "fnr_gap": float(worst_fnr_row["fnr"] - best_fnr_row["fnr"]),
            "worst_fnr_slice": str(worst_fnr_row["slice"]),
            "best_fnr_slice": str(best_fnr_row["slice"]),
        }
    return out


# ---------- main pipeline ----------


def run_pipeline(
    features_path: Path = DEFAULT_FEATURES,
    winner_path: Path = DEFAULT_WINNER,
    calibrator_path: Path = DEFAULT_CALIBRATOR,
    threshold_path: Path = DEFAULT_THRESHOLD,
    summary_out: Path = DEFAULT_SUMMARY_OUT,
    gaps_out: Path = DEFAULT_GAPS_OUT,
    pred_out: Path = DEFAULT_PRED_OUT,
    mlruns_dir: Path = DEFAULT_MLRUNS,
    seed: int = RANDOM_STATE,
) -> dict:
    """End-to-end Phase-9 routine."""
    t0 = time.time()
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)

    threshold_payload = json.loads(threshold_path.read_text())
    t_star = float(threshold_payload["threshold"])
    logger.info("loading winner pipeline: %s  ·  t* = %.4f", winner_path, t_star)

    pipe = joblib.load(winner_path)
    cal = joblib.load(calibrator_path)
    X_val, y_val, g_val, A_val = _load_val_with_demographics(features_path)

    logger.info("scoring winner on val ...")
    p_raw = pipe.predict_proba(X_val)[:, 1]
    p_cal = cal.predict(p_raw)

    # Reuse Phase-7 split — audit is computed on the test half, exactly the
    # rows the threshold was selected on, so there is no train/eval leakage.
    _, test_mask = _grouped_stratified_half_split(y_val, g_val, seed=seed)
    y, p, A = y_val[test_mask], p_cal[test_mask], A_val.loc[test_mask].reset_index(drop=True)
    yhat = (p >= t_star).astype(int)
    logger.info(
        "audit on test split: n=%d  pos=%.3f  flagged=%.3f",
        len(y),
        y.mean(),
        yhat.mean(),
    )

    table = _per_attribute_table(y, p, yhat, A)
    gaps = _parity_gaps(y, p, yhat, A)

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_out, index=False)
    gaps_out.write_text(json.dumps(gaps, indent=2))

    pred_df = A.copy()
    pred_df["y"] = y
    pred_df["p"] = p
    pred_df["yhat"] = yhat
    pred_df.to_parquet(pred_out, index=False)

    with mlflow.start_run(run_name="fairness_audit"):
        mlflow.log_param("threshold", t_star)
        mlflow.log_param("n_test", int(len(y)))
        mlflow.log_param("attributes", ",".join(SENSITIVE_ATTRS))
        for attr, g in gaps.items():
            for k, v in g.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{attr}.{k}", float(v))
        mlflow.log_artifact(str(summary_out))
        mlflow.log_artifact(str(gaps_out))

    elapsed = time.time() - t0
    logger.info(
        "Phase 9 done in %.1fs  ·  worst FNR slices: %s",
        elapsed,
        {a: f"{v['worst_fnr_slice']} ({v['fnr_max']:.3f})" for a, v in gaps.items()},
    )
    return {
        "summary_path": str(summary_out),
        "gaps_path": str(gaps_out),
        "n_test": int(len(y)),
        "threshold": t_star,
        "gaps": gaps,
    }


# ---------- CLI ----------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase-9 fairness audit")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--gaps-out", type=Path, default=DEFAULT_GAPS_OUT)
    parser.add_argument("--pred-out", type=Path, default=DEFAULT_PRED_OUT)
    parser.add_argument("--mlruns", type=Path, default=DEFAULT_MLRUNS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--log-file", type=Path, default=Path("reports/_fairness.log"))
    args = parser.parse_args()

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, mode="w")],
    )
    for noisy in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run_pipeline(
        features_path=args.features,
        winner_path=args.winner,
        calibrator_path=args.calibrator,
        threshold_path=args.threshold,
        summary_out=args.summary_out,
        gaps_out=args.gaps_out,
        pred_out=args.pred_out,
        mlruns_dir=args.mlruns,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
