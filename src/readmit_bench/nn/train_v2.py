"""V2 train driver — fits MLP, TabNet, FT-Transformer, FLAML AutoML on a
patient-grouped subsample of the train split and scores them on the *full*
val split. Persists pipelines, val scores, leaderboard, and curve arrays in
the same shape as the V1 baselines so the visualization layer can be reused.

Run:
    python -m readmit_bench.nn.train_v2 --train-rows 100000 --automl-budget 240
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit

from readmit_bench.evaluation import (
    Metrics,
    compute_metrics,
    pr_curve_points,
    reliability_curve,
    roc_curve_points,
)
from readmit_bench.features import FeatureSpec, build_preprocessor
from readmit_bench.nn.automl import FlamlAutoML
from readmit_bench.nn.models import FTTransformerClassifier, MLPClassifier, TabNetWrapper

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_OUT_DIR = Path("models/v2")
DEFAULT_REPORT_CSV = Path("reports/v2_leaderboard.csv")
DEFAULT_CURVES_NPZ = Path("reports/v2_curves.npz")
DEFAULT_MLRUNS = Path("mlruns")
EXPERIMENT = "readmit-bench-v2"

DISPLAY_NAMES = {
    "mlp": "PyTorch MLP",
    "tabnet": "TabNet",
    "ft_transformer": "FT-Transformer",
    "flaml_automl": "FLAML AutoML",
}


@dataclass
class V2Result:
    name: str
    display_name: str
    metrics: Metrics
    fit_time_s: float
    predict_time_s: float
    estimator: object
    val_scores: np.ndarray
    pr_curve: tuple
    roc_curve: tuple
    reliability: tuple


def _load_train_val(
    features_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    spec = FeatureSpec()
    df = pl.read_parquet(features_path)
    train = df.filter(pl.col("split") == "train")
    val = df.filter(pl.col("split") == "val")
    cols = list(spec.all_features())
    X_tr = train.select(cols).to_pandas()
    g_tr = train["beneficiary_id"].to_numpy()
    y_tr = train["y"].to_numpy().astype(int)
    X_va = val.select(cols).to_pandas()
    y_va = val["y"].to_numpy().astype(int)
    logger.info(
        "loaded — train %d  val %d  pos rate %.3f / %.3f",
        len(X_tr),
        len(X_va),
        y_tr.mean(),
        y_va.mean(),
    )
    return X_tr, y_tr, g_tr, X_va, y_va


def _patient_grouped_subsample(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_target: int, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """Take a patient-grouped, label-stratified subsample of approximately n_target rows."""
    if len(X) <= n_target:
        return X, y
    # Approximate group-aware sampling: pick a random fraction of unique patients
    # such that expected encounter count ≈ n_target. Stratify on per-patient max label
    # so positive prevalence is preserved.
    rng = np.random.default_rng(seed)
    groups_arr = np.asarray(groups)
    unique_g, inverse = np.unique(groups_arr, return_inverse=True)
    # Per-patient label = max of encounter labels (1 if any positive)
    g_y = np.zeros(len(unique_g), dtype=int)
    np.maximum.at(g_y, inverse, y)
    keep_frac = n_target / len(X)
    # Stratified pick over groups
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=min(0.999, keep_frac), random_state=seed
    )
    idx_g, _ = next(splitter.split(np.zeros(len(unique_g)), g_y))
    _ = rng  # silence unused
    keep_groups = set(unique_g[idx_g].tolist())
    mask = np.array([g in keep_groups for g in groups_arr])
    return X.loc[mask].reset_index(drop=True), y[mask]


def _to_dense(M) -> np.ndarray:
    return np.asarray(M.toarray() if sparse.issparse(M) else M, dtype=np.float32)


def _make_estimator(name: str, automl_budget: int):
    if name == "mlp":
        return MLPClassifier()
    if name == "tabnet":
        return TabNetWrapper()
    if name == "ft_transformer":
        return FTTransformerClassifier()
    if name == "flaml_automl":
        return FlamlAutoML(time_budget=automl_budget)
    raise ValueError(name)


def _train_one(
    name: str,
    X_tr_dense: np.ndarray,
    y_tr_sub: np.ndarray,
    X_va_dense: np.ndarray,
    y_va: np.ndarray,
    automl_budget: int,
) -> V2Result:
    est = _make_estimator(name, automl_budget=automl_budget)
    logger.info("[%s] fit on %d rows × %d features", name, len(X_tr_dense), X_tr_dense.shape[1])
    t0 = time.time()
    est.fit(X_tr_dense, y_tr_sub)
    fit_time = time.time() - t0

    t0 = time.time()
    proba = est.predict_proba(X_va_dense)[:, 1]
    predict_time = time.time() - t0
    metrics = compute_metrics(y_va, proba)
    logger.info(
        "[%s] PR-AUC=%.4f ROC-AUC=%.4f Brier=%.4f recall@10%%=%.3f  fit=%.1fs predict=%.2fs",
        name,
        metrics.pr_auc,
        metrics.roc_auc,
        metrics.brier,
        metrics.recall_at_top10,
        fit_time,
        predict_time,
    )
    return V2Result(
        name=name,
        display_name=DISPLAY_NAMES[name],
        metrics=metrics,
        fit_time_s=fit_time,
        predict_time_s=predict_time,
        estimator=est,
        val_scores=proba,
        pr_curve=pr_curve_points(y_va, proba),
        roc_curve=roc_curve_points(y_va, proba),
        reliability=reliability_curve(y_va, proba, n_bins=15),
    )


def _persist(result: V2Result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.estimator, out_dir / f"{result.name}.joblib", compress=3)
    np.save(out_dir / f"{result.name}_val_scores.npy", result.val_scores)


def _log_to_mlflow(result: V2Result, n_train_sub: int, n_features: int) -> None:
    with mlflow.start_run(run_name=result.name):
        mlflow.set_tag("model_family", result.name)
        mlflow.set_tag("display_name", result.display_name)
        mlflow.set_tag("phase", "v2")
        mlflow.log_params(
            {
                "n_train_sub": n_train_sub,
                "n_features_dense": n_features,
                "estimator": type(result.estimator).__name__,
            }
        )
        for k, v in result.metrics.as_dict().items():
            mlflow.log_metric(f"val_{k}", float(v))
        mlflow.log_metric("fit_time_s", result.fit_time_s)
        mlflow.log_metric("predict_time_s", result.predict_time_s)


def _write_leaderboard(results: list[V2Result], out_csv: Path) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "name": r.name,
                "display_name": r.display_name,
                **r.metrics.as_dict(),
                "fit_time_s": r.fit_time_s,
                "predict_time_s": r.predict_time_s,
            }
        )
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("wrote leaderboard → %s", out_csv)
    return df


def _write_curves(results: list[V2Result], out_path: Path) -> None:
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
    train_rows: int = 100_000,
    automl_budget: int = 240,
    only: list[str] | None = None,
) -> pd.DataFrame:
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir.resolve().as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    X_tr, y_tr, g_tr, X_va, y_va = _load_train_val(features_path)
    X_tr_sub, y_tr_sub = _patient_grouped_subsample(X_tr, y_tr, g_tr, train_rows)
    logger.info("subsample → %d rows (pos rate %.3f)", len(X_tr_sub), y_tr_sub.mean())

    pre = build_preprocessor()
    pre.fit(X_tr_sub, y_tr_sub)
    X_tr_dense = _to_dense(pre.transform(X_tr_sub))
    X_va_dense = _to_dense(pre.transform(X_va))
    n_features = X_tr_dense.shape[1]
    logger.info(
        "preprocessed dense matrices — train %s  val %s", X_tr_dense.shape, X_va_dense.shape
    )

    # Persist the fitted preprocessor so we can reuse the exact projection at scoring time.
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, out_dir / "preprocessor.joblib", compress=3)

    names = only or list(DISPLAY_NAMES.keys())
    results: list[V2Result] = []
    for name in names:
        try:
            r = _train_one(name, X_tr_dense, y_tr_sub, X_va_dense, y_va, automl_budget)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] failed: %s — skipping", name, exc)
            continue
        _persist(r, out_dir)
        _log_to_mlflow(r, len(X_tr_sub), n_features)
        results.append(r)

    if not results:
        raise RuntimeError("no V2 models trained successfully")
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
    p.add_argument("--train-rows", type=int, default=100_000)
    p.add_argument("--automl-budget", type=int, default=240)
    p.add_argument("--only", nargs="+", choices=list(DISPLAY_NAMES.keys()))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("mlflow", "fontTools", "fontTools.subset", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = _parse_args()
    train_all(
        features_path=args.features,
        out_dir=args.out_dir,
        report_csv=args.report_csv,
        curves_npz=args.curves_npz,
        mlruns_dir=args.mlruns,
        train_rows=args.train_rows,
        automl_budget=args.automl_budget,
        only=args.only,
    )


if __name__ == "__main__":
    main()
