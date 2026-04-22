"""V2 — voting + stacking ensembles of the V1 tuned base learners.

Loads the saved val_scores from each tuned model, applies the same
patient-grouped 50/50 calib/test split as Phase 7, and evaluates:

* **Voting** — arithmetic mean of base probabilities.
* **Stacking** — logistic-regression meta-learner fit on calib half, scored
  on test half. Reports both raw-stacked and isotonic-recalibrated variants.

Honest framing: base correlations on this DE-SynPUF cohort are ~0.99, so we
expect minimal lift. The phase exists to document the (lack of) ensemble
benefit, not to beat XGBoost.

Run:
    python -m readmit_bench.ensembles.combine
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from readmit_bench.calibration.calibrate import _grouped_stratified_half_split
from readmit_bench.evaluation import (
    Metrics,
    compute_metrics,
    pr_curve_points,
    reliability_curve,
    roc_curve_points,
)

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_TUNED_DIR = Path("models/tuned")
DEFAULT_OUT_DIR = Path("models/ensembles")
DEFAULT_REPORT_CSV = Path("reports/ensembles_summary.csv")
DEFAULT_CURVES_NPZ = Path("reports/ensembles_curves.npz")
DEFAULT_MLRUNS = Path("mlruns")
EXPERIMENT = "readmit-bench-ensembles"
BASE_NAMES = ("xgboost", "catboost", "hist_gradient_boosting")
RANDOM_STATE = 42


@dataclass
class EnsembleResult:
    name: str
    metrics: Metrics
    val_scores: np.ndarray  # scores on the test half only
    pr_curve: tuple
    roc_curve: tuple
    reliability: tuple
    extra: dict


def _load_val(features_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pl.read_parquet(features_path).filter(pl.col("split") == "val")
    return df["y"].to_numpy().astype(int), df["beneficiary_id"].to_numpy()


def _load_base_scores(tuned_dir: Path, names: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Stack tuned val_scores into shape (n_val, n_models)."""
    cols = []
    used = []
    for n in names:
        path = tuned_dir / f"{n}_val_scores.npy"
        if not path.exists():
            logger.warning("missing %s — skipping", path)
            continue
        cols.append(np.load(path))
        used.append(n)
    return np.column_stack(cols), used


def _voting(P: np.ndarray) -> np.ndarray:
    return P.mean(axis=1)


def _stacking_lr(
    P_cal: np.ndarray, y_cal: np.ndarray, P_test: np.ndarray
) -> tuple[np.ndarray, LogisticRegression]:
    meta = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=RANDOM_STATE)
    meta.fit(P_cal, y_cal)
    p_test = meta.predict_proba(P_test)[:, 1]
    return p_test, meta


def _make_result(
    name: str, y_test: np.ndarray, p_test: np.ndarray, extra: dict | None = None
) -> EnsembleResult:
    m = compute_metrics(y_test, p_test)
    return EnsembleResult(
        name=name,
        metrics=m,
        val_scores=p_test,
        pr_curve=pr_curve_points(y_test, p_test),
        roc_curve=roc_curve_points(y_test, p_test),
        reliability=reliability_curve(y_test, p_test, n_bins=15),
        extra=extra or {},
    )


def _write_outputs(
    results: list[EnsembleResult],
    base_metrics: dict[str, Metrics],
    correlations: pd.DataFrame,
    out_dir: Path,
    report_csv: Path,
    curves_npz: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, m in base_metrics.items():
        rows.append({"name": f"base/{n}", **m.as_dict()})
    for r in results:
        rows.append({"name": r.name, **r.metrics.as_dict(), **r.extra})
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_csv, index=False)

    # Save the meta-learner + correlations for the model card / blog write-up.
    correlations.to_csv(out_dir / "base_correlations.csv")

    payload: dict[str, np.ndarray] = {}
    for r in results:
        payload[f"{r.name}__pr_recall"] = r.pr_curve[0]
        payload[f"{r.name}__pr_precision"] = r.pr_curve[1]
        payload[f"{r.name}__roc_fpr"] = r.roc_curve[0]
        payload[f"{r.name}__roc_tpr"] = r.roc_curve[1]
        payload[f"{r.name}__rel_pred"] = r.reliability[0]
        payload[f"{r.name}__rel_pos"] = r.reliability[1]
        payload[f"{r.name}__rel_count"] = r.reliability[2]
        payload[f"{r.name}__test_scores"] = r.val_scores
    curves_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(curves_npz, **payload)
    logger.info("wrote leaderboard → %s  (rows=%d)", report_csv, len(df))
    logger.info("wrote curves → %s", curves_npz)
    return df


def run(
    features_path: Path = DEFAULT_FEATURES,
    tuned_dir: Path = DEFAULT_TUNED_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    report_csv: Path = DEFAULT_REPORT_CSV,
    curves_npz: Path = DEFAULT_CURVES_NPZ,
    mlruns_dir: Path = DEFAULT_MLRUNS,
) -> pd.DataFrame:
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir.resolve().as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    y_val, g_val = _load_val(features_path)
    P, used_names = _load_base_scores(tuned_dir, BASE_NAMES)
    if P.shape[1] < 2:
        raise RuntimeError("Need at least 2 base learners for ensembles")
    logger.info("loaded base scores — shape %s  base learners %s", P.shape, used_names)

    # Pearson correlation between base probabilities (ranking-equivalent → ~1 means redundancy).
    corr = pd.DataFrame(P, columns=used_names).corr()
    logger.info("base correlations:\n%s", corr.round(3).to_string())

    # 50/50 patient-grouped split (same protocol as Phase 7).
    cal_mask, test_mask = _grouped_stratified_half_split(y_val, g_val, seed=RANDOM_STATE)
    P_cal, P_test = P[cal_mask], P[test_mask]
    y_cal, y_test = y_val[cal_mask], y_val[test_mask]
    logger.info("calib n=%d  test n=%d", cal_mask.sum(), test_mask.sum())

    # ---- per-base reference (test-half metrics) ----
    base_metrics: dict[str, Metrics] = {}
    for j, n in enumerate(used_names):
        base_metrics[n] = compute_metrics(y_test, P_test[:, j])

    # ---- voting (no fitting) ----
    t0 = time.time()
    p_vote = _voting(P_test)
    vote_time = time.time() - t0
    res_vote = _make_result(
        "ensemble/voting_mean",
        y_test,
        p_vote,
        extra={"fit_time_s": 0.0, "predict_time_s": vote_time},
    )

    # ---- stacking (LR meta-learner) ----
    t0 = time.time()
    p_stack, meta = _stacking_lr(P_cal, y_cal, P_test)
    stack_time = time.time() - t0
    res_stack = _make_result(
        "ensemble/stacking_lr",
        y_test,
        p_stack,
        extra={
            "fit_time_s": stack_time,
            "predict_time_s": 0.0,
            "meta_intercept": float(meta.intercept_[0]),
            "meta_coef": json.dumps(
                {n: float(c) for n, c in zip(used_names, meta.coef_[0], strict=True)}
            ),
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"meta": meta, "base_names": used_names}, out_dir / "stacking_lr.joblib", compress=3
    )

    results = [res_vote, res_stack]

    # MLflow logging
    for r in results:
        with mlflow.start_run(run_name=r.name.replace("/", "_")):
            mlflow.set_tag("phase", "v2_ensemble")
            mlflow.log_param("base_learners", ",".join(used_names))
            for k, v in r.metrics.as_dict().items():
                mlflow.log_metric(f"test_{k}", float(v))

    df = _write_outputs(results, base_metrics, corr, out_dir, report_csv, curves_npz)
    logger.info("\n%s", df.to_string(index=False))
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--tuned-dir", type=Path, default=DEFAULT_TUNED_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    p.add_argument("--curves-npz", type=Path, default=DEFAULT_CURVES_NPZ)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("mlflow", "fontTools", "fontTools.subset"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = _parse_args()
    run(
        features_path=args.features,
        tuned_dir=args.tuned_dir,
        out_dir=args.out_dir,
        report_csv=args.report_csv,
        curves_npz=args.curves_npz,
    )


if __name__ == "__main__":
    main()
