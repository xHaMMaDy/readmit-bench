"""Phase-7 — probability calibration + cost-based threshold selection.

Pipeline
--------
1. Load tuned winner pipeline (``models/tuned/xgboost.joblib``) + features parquet.
2. Score the validation set, then split val 50/50 into ``calib`` and ``test``
   with **patient-grouped, stratified** assignment so no patient ends up in both
   halves and the positive rate is preserved.
3. Fit two post-hoc calibrators on ``calib``:

   - **Isotonic regression** (non-parametric, monotonic)
   - **Platt scaling** (logistic regression on raw scores)

4. Score both on ``test``; the one with the lower **Brier score** wins.
5. Pick the operating threshold that minimises a clinical cost function:
   ``cost = $15,000 · FN + $500 · (TP + FP)`` — every miss costs $15K of
   re-admission spend; every flag costs $500 of intervention.
6. Persist:

   - ``models/winner_calibrator.joblib`` (the chosen calibrator object)
   - ``models/winner_threshold.json``  (threshold, cost, confusion, costs)
   - ``reports/calibration_curves.npz`` (reliability + cost-curve arrays)
   - ``reports/calibration_summary.csv`` (Brier / cost / threshold table)

7. Log everything to MLflow under experiment ``readmit-bench-calibration``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from readmit_bench.evaluation import compute_metrics, reliability_curve
from readmit_bench.features import FeatureSpec

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_WINNER = Path("models/tuned/xgboost.joblib")
DEFAULT_WINNER_NAME = "xgboost"
DEFAULT_CALIBRATOR_OUT = Path("models/winner_calibrator.joblib")
DEFAULT_THRESHOLD_OUT = Path("models/winner_threshold.json")
DEFAULT_CURVES_OUT = Path("reports/calibration_curves.npz")
DEFAULT_SUMMARY_OUT = Path("reports/calibration_summary.csv")
DEFAULT_MLRUNS = Path("mlruns")

EXPERIMENT = "readmit-bench-calibration"
RANDOM_STATE = 42

# Clinical cost defaults (USD per encounter)
DEFAULT_COST_FN = 15_000.0  # missed readmission
DEFAULT_COST_FP = 500.0  # unneeded intervention


@dataclass(frozen=True)
class CalibrationResult:
    method: str  # "isotonic" | "platt" | "uncalibrated"
    brier: float
    pr_auc: float
    roc_auc: float
    log_loss: float


# ---------- helpers ----------


def _load_val(features_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    spec = FeatureSpec()
    df = pl.read_parquet(features_path).filter(pl.col("split") == "val")
    cols = list(spec.all_features())
    X = df.select(cols).to_pandas()
    y = df["y"].to_numpy().astype(int)
    g = df["beneficiary_id"].to_numpy()
    logger.info(
        "val: %d × %d  (pos %.3f, %d patients)", X.shape[0], X.shape[1], y.mean(), len(np.unique(g))
    )
    return X, y, g


def _grouped_stratified_half_split(y: np.ndarray, groups: np.ndarray, seed: int = RANDOM_STATE):
    """50/50 patient-grouped, ever-positive-stratified split on val.

    Returns boolean masks (calib_mask, test_mask) over the encounter rows.
    """
    rng = np.random.default_rng(seed)
    pat_df = pd.DataFrame({"g": groups, "y": y})
    ever_pos = pat_df.groupby("g")["y"].max()
    pos_pats = ever_pos[ever_pos == 1].index.to_numpy()
    neg_pats = ever_pos[ever_pos == 0].index.to_numpy()
    rng.shuffle(pos_pats)
    rng.shuffle(neg_pats)
    calib_pats = np.concatenate([pos_pats[: len(pos_pats) // 2], neg_pats[: len(neg_pats) // 2]])
    calib_set = set(calib_pats.tolist())
    calib_mask = np.array([g in calib_set for g in groups])
    return calib_mask, ~calib_mask


def fit_and_select_calibrator(
    p_calib: np.ndarray,
    y_calib: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
):
    """Fit isotonic + Platt on (p_calib, y_calib); pick lowest-Brier on test set.

    The candidate set is {uncalibrated, isotonic, platt}. If the raw model is
    already well-calibrated (common for XGBoost trained with log-loss), the
    identity calibrator wins and is returned — no post-hoc transform applied.

    Returns (best_name, best_calibrator, results_by_method, p_test_best).
    Each calibrator object exposes ``.predict(p)`` returning calibrated probs.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_calib, y_calib)
    p_test_iso = iso.predict(p_test)

    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(p_calib.reshape(-1, 1), y_calib)
    p_test_platt = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]

    results = {
        "uncalibrated": _result("uncalibrated", y_test, p_test),
        "isotonic": _result("isotonic", y_test, p_test_iso),
        "platt": _result("platt", y_test, p_test_platt),
    }
    candidates = {
        "uncalibrated": (_IdentityWrapper(), p_test),
        "isotonic": (_IsoWrapper(iso), p_test_iso),
        "platt": (_PlattWrapper(platt), p_test_platt),
    }
    best_name = min(results, key=lambda k: results[k].brier)
    best_calibrator, p_test_best = candidates[best_name]
    return best_name, best_calibrator, results, p_test_best


def _result(method: str, y: np.ndarray, p: np.ndarray) -> CalibrationResult:
    m = compute_metrics(y, p)
    return CalibrationResult(
        method=method, brier=m.brier, pr_auc=m.pr_auc, roc_auc=m.roc_auc, log_loss=m.log_loss
    )


class _IdentityWrapper:
    """No-op calibrator — used when raw model is already best-calibrated."""

    method = "uncalibrated"

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float)


class _IsoWrapper:
    """Picklable wrapper exposing ``.predict(probs)`` for downstream use."""

    def __init__(self, iso: IsotonicRegression):
        self.method = "isotonic"
        self.iso = iso

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(np.asarray(p, dtype=float))


class _PlattWrapper:
    def __init__(self, lr: LogisticRegression):
        self.method = "platt"
        self.lr = lr

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(np.asarray(p, dtype=float).reshape(-1, 1))[:, 1]


def pick_cost_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_grid: int = 501,
):
    """Sweep thresholds in [0, 1]; pick the one that minimises total cost.

    Returns (threshold, cost, thresholds_grid, costs_grid, confusion_dict).
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    grid = np.linspace(0.0, 1.0, n_grid)
    n_pos = int(y_true.sum())
    costs = np.empty_like(grid)
    for i, t in enumerate(grid):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = n_pos - tp
        costs[i] = cost_fn * fn + cost_fp * (tp + fp)
    best_i = int(np.argmin(costs))
    t_star = float(grid[best_i])
    pred_star = (p >= t_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_star, labels=[0, 1]).ravel()
    confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return t_star, float(costs[best_i]), grid, costs, confusion


# ---------- main pipeline ----------


def run_pipeline(
    features_path: Path = DEFAULT_FEATURES,
    winner_path: Path = DEFAULT_WINNER,
    winner_name: str = DEFAULT_WINNER_NAME,
    calibrator_out: Path = DEFAULT_CALIBRATOR_OUT,
    threshold_out: Path = DEFAULT_THRESHOLD_OUT,
    curves_out: Path = DEFAULT_CURVES_OUT,
    summary_out: Path = DEFAULT_SUMMARY_OUT,
    mlruns_dir: Path = DEFAULT_MLRUNS,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    seed: int = RANDOM_STATE,
) -> dict:
    """End-to-end Phase-7 routine. Returns a small dict for the caller / tests."""
    import mlflow
    
    t0 = time.time()
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)

    logger.info("loading winner pipeline: %s", winner_path)
    pipe = joblib.load(winner_path)
    X_val, y_val, g_val = _load_val(features_path)

    logger.info("scoring winner on val ...")
    p_val = pipe.predict_proba(X_val)[:, 1]

    calib_mask, test_mask = _grouped_stratified_half_split(y_val, g_val, seed=seed)
    p_calib, y_calib = p_val[calib_mask], y_val[calib_mask]
    p_test, y_test = p_val[test_mask], y_val[test_mask]
    logger.info(
        "calib: %d rows (pos %.3f) | test: %d rows (pos %.3f)",
        len(p_calib),
        y_calib.mean(),
        len(p_test),
        y_test.mean(),
    )

    best_name, best_cal, results, p_test_best = fit_and_select_calibrator(
        p_calib, y_calib, p_test, y_test
    )
    logger.info(
        "Brier  uncal=%.5f  iso=%.5f  platt=%.5f  --> winner=%s",
        results["uncalibrated"].brier,
        results["isotonic"].brier,
        results["platt"].brier,
        best_name,
    )

    t_star, cost_star, grid, costs, confusion = pick_cost_threshold(
        y_test,
        p_test_best,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )
    n_test = len(y_test)
    cost_per_encounter = cost_star / n_test
    # Reference: cost of "always treat" and "never treat" baselines on test.
    cost_never = cost_fn * int(y_test.sum())
    cost_always = cost_fp * n_test
    logger.info(
        "cost-min threshold=%.4f  cost=$%.0f  ($%.2f/enc)  vs always=$%.0f  never=$%.0f",
        t_star,
        cost_star,
        cost_per_encounter,
        cost_always,
        cost_never,
    )
    logger.info("confusion @ threshold: %s", confusion)

    # ---------- persist ----------
    calibrator_out.parent.mkdir(parents=True, exist_ok=True)
    threshold_out.parent.mkdir(parents=True, exist_ok=True)
    curves_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_cal, calibrator_out)
    threshold_payload = {
        "winner_model": winner_name,
        "calibrator": best_name,
        "threshold": t_star,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "test_total_cost_usd": cost_star,
        "test_cost_per_encounter_usd": cost_per_encounter,
        "test_cost_always_treat_usd": cost_always,
        "test_cost_never_treat_usd": cost_never,
        "confusion": confusion,
        "test_n": int(n_test),
        "test_n_pos": int(y_test.sum()),
        "seed": seed,
    }
    threshold_out.write_text(json.dumps(threshold_payload, indent=2))

    # Reliability curves (uncal + chosen) on TEST set.
    mp_uncal, fp_uncal, c_uncal = reliability_curve(y_test, p_test, n_bins=15)
    mp_cal, fp_cal, c_cal = reliability_curve(y_test, p_test_best, n_bins=15)
    np.savez(
        curves_out,
        threshold_grid=grid,
        cost_grid=costs,
        rel_uncal_mean_pred=mp_uncal,
        rel_uncal_frac_pos=fp_uncal,
        rel_uncal_counts=c_uncal,
        rel_cal_mean_pred=mp_cal,
        rel_cal_frac_pos=fp_cal,
        rel_cal_counts=c_cal,
        p_test_uncal=p_test,
        p_test_cal=p_test_best,
        y_test=y_test,
    )

    summary_rows = []
    for r in results.values():
        summary_rows.append(
            {
                "method": r.method,
                "brier": r.brier,
                "pr_auc": r.pr_auc,
                "roc_auc": r.roc_auc,
                "log_loss": r.log_loss,
                "is_winner": r.method == best_name,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_out, index=False)

    # ---------- MLflow ----------
    with mlflow.start_run(run_name=f"{winner_name}_calibration"):
        mlflow.set_tag("phase", "7")
        mlflow.set_tag("winner_model", winner_name)
        for r in results.values():
            mlflow.log_metric(f"{r.method}_brier", r.brier)
            mlflow.log_metric(f"{r.method}_pr_auc", r.pr_auc)
            mlflow.log_metric(f"{r.method}_roc_auc", r.roc_auc)
        mlflow.log_metric("chosen_threshold", t_star)
        mlflow.log_metric("test_total_cost_usd", cost_star)
        mlflow.log_metric("test_cost_per_encounter_usd", cost_per_encounter)
        mlflow.log_param("calibrator", best_name)
        mlflow.log_param("cost_fn", cost_fn)
        mlflow.log_param("cost_fp", cost_fp)
        mlflow.log_artifact(str(threshold_out))
        mlflow.log_artifact(str(summary_out))

    elapsed = time.time() - t0
    logger.info("Phase-7 complete in %.1fs", elapsed)
    return {
        "calibrator": best_name,
        "threshold": t_star,
        "cost": cost_star,
        "results": {k: asdict(v) for k, v in results.items()},
        "confusion": confusion,
        "elapsed": elapsed,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-7 calibration + cost threshold for the tuned winner."
    )
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    p.add_argument("--winner-name", default=DEFAULT_WINNER_NAME)
    p.add_argument("--cost-fn", type=float, default=DEFAULT_COST_FN)
    p.add_argument("--cost-fp", type=float, default=DEFAULT_COST_FP)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_argparser().parse_args()
    run_pipeline(
        features_path=args.features,
        winner_path=args.winner,
        winner_name=args.winner_name,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
