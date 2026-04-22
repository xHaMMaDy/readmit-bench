"""Phase-6 Optuna tuning orchestrator. See module docstring of optimize.main."""

from __future__ import annotations

import argparse
import logging
import pickle
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from readmit_bench.evaluation import (
    compute_metrics,
    pr_curve_points,
    reliability_curve,
    roc_curve_points,
)
from readmit_bench.features import FeatureSpec, build_preprocessor
from readmit_bench.models.baselines import _to_dense
from readmit_bench.models.registry import DISPLAY_NAMES, safe_make
from readmit_bench.tuning.spaces import SPACES

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_MODELS_DIR = Path("models/tuned")
DEFAULT_REPORTS_DIR = Path("reports/tuning")
DEFAULT_SUMMARY_CSV = Path("reports/tuned_summary.csv")
DEFAULT_CURVES_NPZ = Path("reports/tuned_curves.npz")
DEFAULT_MLRUNS = Path("mlruns")

EXPERIMENT = "readmit-bench-tuning"
TOP3 = ("catboost", "xgboost", "hist_gradient_boosting")
RANDOM_STATE = 42


def _load_train_val(features_path: Path):
    spec = FeatureSpec()
    df = pl.read_parquet(features_path)
    train = df.filter(pl.col("split") == "train")
    val = df.filter(pl.col("split") == "val")

    cols = list(spec.all_features())
    X_tr = train.select(cols).to_pandas()
    y_tr = train["y"].to_numpy().astype(int)
    g_tr = train["beneficiary_id"].to_numpy()

    X_va = val.select(cols).to_pandas()
    y_va = val["y"].to_numpy().astype(int)

    logger.info(
        "train: %d × %d (pos %.3f, %d patients) | val: %d × %d (pos %.3f)",
        X_tr.shape[0],
        X_tr.shape[1],
        y_tr.mean(),
        len(np.unique(g_tr)),
        X_va.shape[0],
        X_va.shape[1],
        y_va.mean(),
    )
    return X_tr, y_tr, g_tr, X_va, y_va


def _grouped_stratified_subsample(X, y, groups, n_target, seed=RANDOM_STATE):
    if len(X) <= n_target:
        return X, y, groups
    rng = np.random.default_rng(seed)
    pat_df = pd.DataFrame({"g": groups, "y": y})
    pat_pos = pat_df.groupby("g")["y"].max()
    pos_pats = pat_pos[pat_pos == 1].index.to_numpy()
    neg_pats = pat_pos[pat_pos == 0].index.to_numpy()
    rng.shuffle(pos_pats)
    rng.shuffle(neg_pats)
    overall_pos_share = float(pat_pos.mean())
    avg_enc_per_pat = pat_df.groupby("g").size().mean()
    n_pats_target = int(n_target / avg_enc_per_pat)
    n_pos_target = int(round(n_pats_target * overall_pos_share))
    n_neg_target = n_pats_target - n_pos_target
    chosen = np.concatenate([pos_pats[:n_pos_target], neg_pats[:n_neg_target]])
    mask = pd.Series(groups).isin(set(chosen)).to_numpy()
    return X.iloc[mask].reset_index(drop=True), y[mask], groups[mask]


def _build_pipeline(name: str, params: dict) -> Pipeline:
    estimator = safe_make(name)
    if estimator is None:
        raise RuntimeError(f"could not instantiate {name} (missing dependency?)")
    estimator.set_params(**params)
    steps = [("preprocess", build_preprocessor())]
    if isinstance(estimator, HistGradientBoostingClassifier):
        steps.append(
            (
                "densify",
                FunctionTransformer(
                    _to_dense,
                    accept_sparse=True,
                    validate=False,
                ),
            )
        )
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def _cv_pr_auc(name, params, X, y, groups, n_splits=3):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    for tr_idx, va_idx in cv.split(X, y, groups):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        pipe = _build_pipeline(name, params)
        pipe.fit(Xtr, ytr)
        scores = pipe.predict_proba(Xva)[:, 1]
        fold_scores.append(compute_metrics(yva, scores).pr_auc)
    return float(np.mean(fold_scores))


def _make_objective(name, X, y, groups, n_splits):
    space = SPACES[name]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        t0 = time.time()
        try:
            pr_auc = _cv_pr_auc(name, params, X, y, groups, n_splits=n_splits)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] trial %d failed: %s", name, trial.number, exc)
            raise optuna.TrialPruned() from exc
        elapsed = time.time() - t0
        trial.set_user_attr("cv_pr_auc", pr_auc)
        trial.set_user_attr("seconds", elapsed)
        with mlflow.start_run(run_name=f"{name}_trial_{trial.number:03d}", nested=True):
            mlflow.set_tag("phase", "tuning")
            mlflow.set_tag("model_family", name)
            mlflow.log_params(params)
            mlflow.log_metric("cv_pr_auc", pr_auc)
            mlflow.log_metric("trial_seconds", elapsed)
        logger.info(
            "[%s] trial %3d  cv-PR-AUC=%.4f  in %5.1fs", name, trial.number, pr_auc, elapsed
        )
        return pr_auc

    return objective


def tune_one(
    name, X_tr, y_tr, g_tr, X_va, y_va, n_trials, n_splits, sample_size, models_dir, reports_dir
):
    logger.info("=" * 78)
    logger.info(
        "Tuning %s — %d trials, %d-fold CV on %dK-row patient subsample",
        name,
        n_trials,
        n_splits,
        sample_size // 1000,
    )
    logger.info("=" * 78)
    Xs, ys, gs = _grouped_stratified_subsample(X_tr, y_tr, g_tr, sample_size)
    logger.info(
        "[%s] subsample: %d rows (pos %.3f) from %d patients",
        name,
        len(Xs),
        ys.mean(),
        len(np.unique(gs)),
    )

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name=f"readmit_bench_{name}"
    )

    with mlflow.start_run(run_name=f"{name}_tuning_parent"):
        mlflow.set_tag("phase", "tuning")
        mlflow.set_tag("model_family", name)
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "n_splits": n_splits,
                "sample_size_target": sample_size,
                "sample_size_actual": len(Xs),
                "sample_pos_rate": float(ys.mean()),
            }
        )
        t0 = time.time()
        study.optimize(
            _make_objective(name, Xs, ys, gs, n_splits=n_splits),
            n_trials=n_trials,
            show_progress_bar=False,
            gc_after_trial=True,
        )
        tuning_seconds = time.time() - t0
        best = study.best_trial
        logger.info(
            "[%s] BEST cv-PR-AUC=%.4f  trial #%d  params=%s",
            name,
            best.value,
            best.number,
            best.params,
        )

        logger.info("[%s] refitting on full train (%d rows)...", name, len(X_tr))
        full_params = SPACES[name](optuna.trial.FixedTrial(best.params))
        pipe = _build_pipeline(name, full_params)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        refit_seconds = time.time() - t0

        scores = pipe.predict_proba(X_va)[:, 1]
        metrics = compute_metrics(y_va, scores)
        logger.info(
            "[%s] VAL  PR-AUC=%.4f  ROC-AUC=%.4f  Brier=%.4f  recall@10%%=%.3f  refit=%.1fs",
            name,
            metrics.pr_auc,
            metrics.roc_auc,
            metrics.brier,
            metrics.recall_at_top10,
            refit_seconds,
        )

        mlflow.log_metric("best_cv_pr_auc", best.value)
        mlflow.log_metric("tuning_seconds", tuning_seconds)
        mlflow.log_metric("refit_seconds", refit_seconds)
        for k, v in metrics.as_dict().items():
            mlflow.log_metric(f"val_{k}", float(v))

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, models_dir / f"{name}.joblib", compress=3)
    np.save(models_dir / f"{name}_val_scores.npy", scores)
    with open(reports_dir / f"{name}_study.pkl", "wb") as f:
        pickle.dump(study, f)
    study.trials_dataframe().to_csv(reports_dir / f"{name}_trials.csv", index=False)

    pr_recall, pr_precision = pr_curve_points(y_va, scores)
    roc_fpr, roc_tpr = roc_curve_points(y_va, scores)
    rel_pred, rel_pos, rel_count = reliability_curve(y_va, scores, n_bins=15)

    return {
        "name": name,
        "display_name": DISPLAY_NAMES[name],
        "best_cv_pr_auc": best.value,
        "best_trial": best.number,
        "best_params": best.params,
        "tuning_seconds": tuning_seconds,
        "refit_seconds": refit_seconds,
        **metrics.as_dict(),
        "val_scores": scores,
        "pr_curve": (pr_recall, pr_precision),
        "roc_curve": (roc_fpr, roc_tpr),
        "reliability": (rel_pred, rel_pos, rel_count),
    }


def _write_summary(results, summary_csv, curves_npz):
    rows, payload = [], {}
    for r in results:
        rows.append(
            {
                "name": r["name"],
                "display_name": r["display_name"],
                "best_cv_pr_auc": r["best_cv_pr_auc"],
                "best_trial": r["best_trial"],
                "pr_auc": r["pr_auc"],
                "roc_auc": r["roc_auc"],
                "brier": r["brier"],
                "log_loss": r["log_loss"],
                "recall_at_top10": r["recall_at_top10"],
                "precision_at_top10": r["precision_at_top10"],
                "n": r["n"],
                "n_pos": r["n_pos"],
                "tuning_seconds": r["tuning_seconds"],
                "refit_seconds": r["refit_seconds"],
                "best_params": r["best_params"],
            }
        )
        n = r["name"]
        payload[f"{n}__pr_recall"] = r["pr_curve"][0]
        payload[f"{n}__pr_precision"] = r["pr_curve"][1]
        payload[f"{n}__roc_fpr"] = r["roc_curve"][0]
        payload[f"{n}__roc_tpr"] = r["roc_curve"][1]
        payload[f"{n}__rel_pred"] = r["reliability"][0]
        payload[f"{n}__rel_pos"] = r["reliability"][1]
        payload[f"{n}__rel_count"] = r["reliability"][2]

    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    np.savez_compressed(curves_npz, **payload)
    logger.info("wrote %s and %s", summary_csv, curves_npz)
    return df


def tune_all(
    features_path=DEFAULT_FEATURES,
    models_dir=DEFAULT_MODELS_DIR,
    reports_dir=DEFAULT_REPORTS_DIR,
    summary_csv=DEFAULT_SUMMARY_CSV,
    curves_npz=DEFAULT_CURVES_NPZ,
    mlruns_dir=DEFAULT_MLRUNS,
    only=None,
    n_trials=25,
    n_splits=3,
    sample_size=500_000,
):
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir.resolve().as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr, y_tr, g_tr, X_va, y_va = _load_train_val(features_path)
    names = list(only) if only else list(TOP3)
    results = []
    for name in names:
        if name not in SPACES:
            logger.warning("no search space for %s — skipping", name)
            continue
        res = tune_one(
            name,
            X_tr,
            y_tr,
            g_tr,
            X_va,
            y_va,
            n_trials=n_trials,
            n_splits=n_splits,
            sample_size=sample_size,
            models_dir=models_dir,
            reports_dir=reports_dir,
        )
        if res is not None:
            results.append(res)
    if not results:
        raise RuntimeError("no models tuned")
    df = _write_summary(results, summary_csv, curves_npz)
    logger.info(
        "\nFINAL TUNED LEADERBOARD:\n%s",
        df[
            [
                "display_name",
                "best_cv_pr_auc",
                "pr_auc",
                "roc_auc",
                "brier",
                "recall_at_top10",
                "tuning_seconds",
            ]
        ].to_string(index=False),
    )
    return df


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    p.add_argument("--curves-npz", type=Path, default=DEFAULT_CURVES_NPZ)
    p.add_argument("--mlruns", type=Path, default=DEFAULT_MLRUNS)
    p.add_argument("--only", nargs="+", choices=list(TOP3))
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--sample-size", type=int, default=500_000)
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("mlflow", "fontTools", "fontTools.subset", "matplotlib.font_manager"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = _parse_args()
    tune_all(
        features_path=args.features,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        summary_csv=args.summary_csv,
        curves_npz=args.curves_npz,
        mlruns_dir=args.mlruns,
        only=args.only,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
