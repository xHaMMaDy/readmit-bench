"""Train all V1 baseline models on the full train split, eval on val, log to MLflow.

Run:
    python -m readmit_bench.models.baselines

Outputs:
    models/baselines/<name>.joblib       -- fitted Pipeline (preprocessor + estimator)
    models/baselines/<name>_val_scores.npy
    reports/baselines.csv                -- leaderboard
    reports/baselines_curves.npz         -- PR/ROC/calibration arrays for plotting
    mlruns/                              -- MLflow file store
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from readmit_bench.evaluation import (
    Metrics,
    compute_metrics,
    pr_curve_points,
    reliability_curve,
    roc_curve_points,
)
from readmit_bench.features import FeatureSpec, build_preprocessor
from readmit_bench.models.registry import DISPLAY_NAMES, REGISTRY, safe_make

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_OUT_DIR = Path("models/baselines")
DEFAULT_REPORT_CSV = Path("reports/baselines.csv")
DEFAULT_CURVES_NPZ = Path("reports/baselines_curves.npz")
DEFAULT_MLRUNS = Path("mlruns")
EXPERIMENT = "readmit-bench-baselines"


def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else X


@dataclass
class TrainResult:
    name: str
    display_name: str
    metrics: Metrics
    fit_time_s: float
    predict_time_s: float
    pipeline: Pipeline
    val_scores: np.ndarray
    pr_curve: tuple[np.ndarray, np.ndarray] = field(default=None)
    roc_curve: tuple[np.ndarray, np.ndarray] = field(default=None)
    reliability: tuple[np.ndarray, np.ndarray, np.ndarray] = field(default=None)


def _load_split(features_path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    spec = FeatureSpec()
    df = pl.read_parquet(features_path)
    train = df.filter(pl.col("split") == "train")
    val = df.filter(pl.col("split") == "val")
    cols = list(spec.all_features())
    X_tr = train.select(cols).to_pandas()
    X_va = val.select(cols).to_pandas()
    y_tr = train["y"].to_numpy().astype(int)
    y_va = val["y"].to_numpy().astype(int)
    logger.info(
        "loaded splits — train: %d × %d  val: %d × %d (pos rate %.3f / %.3f)",
        X_tr.shape[0], X_tr.shape[1], X_va.shape[0], X_va.shape[1],
        y_tr.mean(), y_va.mean(),
    )
    return X_tr, y_tr, X_va, y_va


def _train_one(
    name: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
) -> TrainResult | None:
    estimator = safe_make(name)
    if estimator is None:
        return None

    steps = [("preprocess", build_preprocessor())]
    if isinstance(estimator, HistGradientBoostingClassifier):
        steps.append(("densify", FunctionTransformer(
            _to_dense, accept_sparse=True, validate=False,
        )))
    steps.append(("model", estimator))
    pipe = Pipeline(steps=steps)

    logger.info("[%s] fitting on %d rows…", name, len(y_tr))
    t0 = time.time()
    try:
        pipe.fit(X_tr, y_tr)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] fit failed: %s — skipping", name, exc)
        return None
    fit_time = time.time() - t0

    t0 = time.time()
    scores = pipe.predict_proba(X_va)[:, 1]
    predict_time = time.time() - t0

    metrics = compute_metrics(y_va, scores)
    logger.info(
        "[%s] PR-AUC=%.4f ROC-AUC=%.4f Brier=%.4f recall@10%%=%.3f  fit=%.1fs predict=%.2fs",
        name, metrics.pr_auc, metrics.roc_auc, metrics.brier,
        metrics.recall_at_top10, fit_time, predict_time,
    )

    return TrainResult(
        name=name,
        display_name=DISPLAY_NAMES[name],
        metrics=metrics,
        fit_time_s=fit_time,
        predict_time_s=predict_time,
        pipeline=pipe,
        val_scores=scores,
        pr_curve=pr_curve_points(y_va, scores),
        roc_curve=roc_curve_points(y_va, scores),
        reliability=reliability_curve(y_va, scores, n_bins=15),
    )


def _persist(result: TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, out_dir / f"{result.name}.joblib", compress=3)
    np.save(out_dir / f"{result.name}_val_scores.npy", result.val_scores)


def _log_to_mlflow(result: TrainResult, X_tr_shape: tuple[int, int]) -> None:
    with mlflow.start_run(run_name=result.name):
        mlflow.set_tag("model_family", result.name)
        mlflow.set_tag("display_name", result.display_name)
        mlflow.log_params(
            {
                "n_train": X_tr_shape[0],
                "n_features_input": X_tr_shape[1],
                "estimator": type(result.pipeline.named_steps["model"]).__name__,
            }
        )
        for k, v in result.metrics.as_dict().items():
            mlflow.log_metric(f"val_{k}", float(v))
        mlflow.log_metric("fit_time_s", result.fit_time_s)
        mlflow.log_metric("predict_time_s", result.predict_time_s)


def _write_leaderboard(results: list[TrainResult], out_csv: Path) -> pd.DataFrame:
    rows = []
    for r in results:
        d = r.metrics.as_dict()
        rows.append(
            {
                "name": r.name,
                "display_name": r.display_name,
                **d,
                "fit_time_s": r.fit_time_s,
                "predict_time_s": r.predict_time_s,
            }
        )
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("wrote leaderboard → %s", out_csv)
    return df


def _write_curves(results: list[TrainResult], out_path: Path) -> None:
    payload: dict[str, np.ndarray] = {}
    for r in results:
        payload[f"{r.name}__pr_recall"] = r.pr_curve[0]
        payload[f"{r.name}__pr_precision"] = r.pr_curve[1]
        payload[f"{r.name}__roc_fpr"] = r.roc_curve[0]
        payload[f"{r.name}__roc_tpr"] = r.roc_curve[1]
        payload[f"{r.name}__rel_pred"] = r.reliability[0]
        payload[f"{r.name}__rel_pos"] = r.reliability[1]
        payload[f"{r.name}__rel_count"] = r.reliability[2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    logger.info("wrote curves → %s", out_path)


def train_all(
    features_path: Path = DEFAULT_FEATURES,
    out_dir: Path = DEFAULT_OUT_DIR,
    report_csv: Path = DEFAULT_REPORT_CSV,
    curves_npz: Path = DEFAULT_CURVES_NPZ,
    mlruns_dir: Path = DEFAULT_MLRUNS,
    only: list[str] | None = None,
) -> pd.DataFrame:
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir.resolve().as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    X_tr, y_tr, X_va, y_va = _load_split(features_path)
    names = only or list(REGISTRY.keys())

    results: list[TrainResult] = []
    for name in names:
        result = _train_one(name, X_tr, y_tr, X_va, y_va)
        if result is None:
            continue
        _persist(result, out_dir)
        _log_to_mlflow(result, X_tr.shape)
        results.append(result)

    if not results:
        raise RuntimeError("no models trained successfully")

    df = _write_leaderboard(results, report_csv)
    _write_curves(results, curves_npz)
    logger.info("\n%s", df.to_string(index=False))
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    p.add_argument("--curves-npz", type=Path, default=DEFAULT_CURVES_NPZ)
    p.add_argument("--mlruns", type=Path, default=DEFAULT_MLRUNS)
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(REGISTRY.keys()),
        help="Train only these models (default: all).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("mlflow", "fontTools", "fontTools.subset"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = _parse_args()
    train_all(
        features_path=args.features,
        out_dir=args.out_dir,
        report_csv=args.report_csv,
        curves_npz=args.curves_npz,
        mlruns_dir=args.mlruns,
        only=args.only,
    )


if __name__ == "__main__":
    main()
