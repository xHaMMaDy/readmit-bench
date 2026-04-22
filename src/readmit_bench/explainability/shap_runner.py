"""Phase-8 — SHAP interpretability for the tuned XGBoost winner.

Pipeline
--------
1. Load winner pipeline (`models/tuned/xgboost.joblib`) — a 2-step ``Pipeline``
   with ``preprocess`` (ColumnTransformer) + ``model`` (XGBClassifier).
2. Sample N rows from the val split (patient-grouped, stratified) so the
   explainer sees a representative slice without scoring all 198K encounters.
3. Apply ``preprocess.transform`` to get the matrix the booster actually saw,
   densify (TreeExplainer wants dense), then run ``shap.TreeExplainer``.
4. Persist:

   - ``reports/shap_values.npz``        — shap_values, X_processed, base_value, y_sample
   - ``reports/shap_top_features.csv``  — mean(|SHAP|) per feature, ranked

5. Log to MLflow under experiment ``readmit-bench-shap``.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import shap

from readmit_bench.features import FeatureSpec

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_WINNER = Path("models/tuned/xgboost.joblib")
DEFAULT_SHAP_NPZ = Path("reports/shap_values.npz")
DEFAULT_TOP_CSV = Path("reports/shap_top_features.csv")
DEFAULT_MLRUNS = Path("mlruns")

EXPERIMENT = "readmit-bench-shap"
RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 5000


def _load_val(features_path: Path):
    spec = FeatureSpec()
    df = pl.read_parquet(features_path).filter(pl.col("split") == "val")
    cols = list(spec.all_features())
    X = df.select(cols).to_pandas()
    y = df["y"].to_numpy().astype(int)
    g = df["beneficiary_id"].to_numpy()
    return X, y, g


def _grouped_stratified_sample(X, y, groups, n_target, seed=RANDOM_STATE):
    """Pick whole patients up to ~n_target rows, preserving ever-positive rate."""
    if len(X) <= n_target:
        return X, y
    rng = np.random.default_rng(seed)
    pat_df = pd.DataFrame({"g": groups, "y": y, "i": np.arange(len(y))})
    ever_pos = pat_df.groupby("g")["y"].max()
    pos_pats = ever_pos[ever_pos == 1].index.to_numpy()
    neg_pats = ever_pos[ever_pos == 0].index.to_numpy()
    rng.shuffle(pos_pats)
    rng.shuffle(neg_pats)
    pos_share = len(pos_pats) / (len(pos_pats) + len(neg_pats))
    target_pos = max(1, int(round(n_target * pos_share)))
    target_neg = max(1, n_target - target_pos)
    chosen = set(pos_pats[:target_pos].tolist()) | set(neg_pats[:target_neg].tolist())
    mask = pat_df["g"].isin(chosen).to_numpy()
    return X.iloc[mask].reset_index(drop=True), y[mask]


def compute_shap_values(pipe, X_sample: pd.DataFrame):
    """Return (shap_values, X_processed_dense, feature_names, base_value)."""
    pre = pipe.named_steps["preprocess"]
    booster = pipe.named_steps["model"]
    Xt = pre.transform(X_sample)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    feature_names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    shap_vals = explainer.shap_values(Xt)
    if isinstance(shap_vals, list):  # sklearn-style binary output (rare for xgb)
        shap_vals = shap_vals[1]
    base_value = float(np.asarray(explainer.expected_value).ravel()[-1])
    return np.asarray(shap_vals, dtype=np.float32), Xt.astype(np.float32), feature_names, base_value


def run_pipeline(
    features_path: Path = DEFAULT_FEATURES,
    winner_path: Path = DEFAULT_WINNER,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    shap_npz_out: Path = DEFAULT_SHAP_NPZ,
    top_csv_out: Path = DEFAULT_TOP_CSV,
    mlruns_dir: Path = DEFAULT_MLRUNS,
    seed: int = RANDOM_STATE,
) -> dict:
    t0 = time.time()
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)

    logger.info("loading winner pipeline: %s", winner_path)
    pipe = joblib.load(winner_path)
    X_val, y_val, g_val = _load_val(features_path)
    logger.info(
        "val: %d rows  (pos %.3f, %d patients)", len(X_val), y_val.mean(), len(np.unique(g_val))
    )

    X_sample, y_sample = _grouped_stratified_sample(X_val, y_val, g_val, sample_size, seed=seed)
    logger.info("SHAP sample: %d rows  (pos %.3f)", len(X_sample), y_sample.mean())

    logger.info("computing SHAP values via TreeExplainer ...")
    t1 = time.time()
    shap_vals, X_proc, feat_names, base_value = compute_shap_values(pipe, X_sample)
    logger.info(
        "SHAP done in %.1fs  (shape %s, base=%.4f)", time.time() - t1, shap_vals.shape, base_value
    )

    # ---------- top-feature table ----------
    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_signed = shap_vals.mean(axis=0)
    top_df = (
        pd.DataFrame(
            {
                "feature": feat_names,
                "mean_abs_shap": mean_abs,
                "mean_signed_shap": mean_signed,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_npz_out.parent.mkdir(parents=True, exist_ok=True)
    top_csv_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shap_npz_out,
        shap_values=shap_vals,
        X_processed=X_proc,
        feature_names=np.array(feat_names),
        base_value=np.float32(base_value),
        y_sample=y_sample.astype(np.int8),
    )
    top_df.to_csv(top_csv_out, index=False)
    logger.info("wrote %s  (%d features) and %s", top_csv_out, len(feat_names), shap_npz_out)

    # Pretty top-10 in log
    logger.info("top-10 features by mean(|SHAP|):")
    for _, r in top_df.head(10).iterrows():
        logger.info(
            "  %-44s  mean|SHAP|=%.4f  signed=%+.4f",
            r["feature"],
            r["mean_abs_shap"],
            r["mean_signed_shap"],
        )

    with mlflow.start_run(run_name="xgboost_shap"):
        mlflow.set_tag("phase", "8")
        mlflow.set_tag("winner_model", "xgboost")
        mlflow.log_param("sample_size", len(X_sample))
        mlflow.log_param("n_features_processed", len(feat_names))
        mlflow.log_metric("base_value", base_value)
        mlflow.log_metric("top1_mean_abs_shap", float(top_df.iloc[0]["mean_abs_shap"]))
        mlflow.log_artifact(str(top_csv_out))

    elapsed = time.time() - t0
    logger.info("Phase-8 complete in %.1fs", elapsed)
    return {
        "n_sample": int(len(X_sample)),
        "n_features": int(len(feat_names)),
        "base_value": base_value,
        "top10": top_df.head(10).to_dict("records"),
        "elapsed": elapsed,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    p = argparse.ArgumentParser(description="Phase-8 SHAP for the tuned XGBoost winner.")
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = p.parse_args()
    run_pipeline(
        features_path=args.features,
        winner_path=args.winner,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
