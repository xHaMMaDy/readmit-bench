"""Phase-15 — LIME local explanations on the V1 winner.

Generates LIME explanations for a few high-risk and low-risk patients so the
model card has a "why this prediction" story to tell. Wraps the tuned XGBoost
pipeline (preprocessor + booster) and runs lime_tabular against the *raw*
feature space so the explanations are interpretable to a clinician.

Run:
    python -m readmit_bench.explainability.lime_runner
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from lime.lime_tabular import LimeTabularExplainer

from readmit_bench.features import FeatureSpec

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_WINNER = Path("models/tuned/xgboost.joblib")
DEFAULT_OUT_DIR = Path("reports/lime")
RANDOM_STATE = 42


def _load_split(features_path: Path, split: str):
    spec = FeatureSpec()
    df = pl.read_parquet(features_path).filter(pl.col("split") == split)
    cols = list(spec.all_features())
    X = df.select(cols).to_pandas()
    y = df["y"].to_numpy().astype(int)
    return X, y, cols


def _categorical_indices(
    cols: list[str], spec: FeatureSpec
) -> tuple[list[int], dict[int, list[str]]]:
    cat_cols = set(spec.cat_lowcard) | set(spec.cat_highcard) | set(spec.binary)
    idx, names = [], {}
    for i, c in enumerate(cols):
        if c in cat_cols:
            idx.append(i)
    return idx, names


def run(
    features_path: Path = DEFAULT_FEATURES,
    winner_path: Path = DEFAULT_WINNER,
    out_dir: Path = DEFAULT_OUT_DIR,
    n_each: int = 3,
    n_train_background: int = 5000,
    n_features: int = 10,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = FeatureSpec()
    pipe = joblib.load(winner_path)

    X_train, _y_train, cols = _load_split(features_path, "train")
    X_val, _y_val, _ = _load_split(features_path, "val")
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=min(n_train_background, len(X_train)), replace=False)
    X_bg = X_train.iloc[bg_idx].reset_index(drop=True)

    cat_cols = set(spec.cat_lowcard) | set(spec.cat_highcard)
    cat_idx = [i for i, c in enumerate(cols) if c in cat_cols]

    # Label-encode every categorical/binary column for LIME (which needs a float matrix).
    # Build the encoding from background+val so that any val value the explainer
    # samples maps back cleanly to the original raw value for the predict_fn.
    cat_maps: dict[int, list] = {}
    inverse_maps: dict[int, dict] = {}
    X_bg_enc = X_bg.copy()
    X_val_enc = X_val.copy()
    for i, c in enumerate(cols):
        if c in cat_cols:
            combined = (
                pd.concat([X_bg[c], X_val[c]], ignore_index=True)
                .astype("string")
                .fillna("__missing__")
            )
            uniques = pd.Index(combined.unique())
            cat_maps[i] = uniques.tolist()
            mapping = {v: k for k, v in enumerate(uniques)}
            inverse_maps[i] = {k: (v if v != "__missing__" else None) for v, k in mapping.items()}
            X_bg_enc[c] = X_bg[c].astype("string").fillna("__missing__").map(mapping).astype(float)
            X_val_enc[c] = (
                X_val[c].astype("string").fillna("__missing__").map(mapping).astype(float)
            )
        else:
            X_bg_enc[c] = pd.to_numeric(X_bg[c], errors="coerce").astype(float).fillna(0.0)
            X_val_enc[c] = pd.to_numeric(X_val[c], errors="coerce").astype(float).fillna(0.0)

    # Score val once on the *raw* feature frame (the pipeline expects raw cols).
    p_val = pipe.predict_proba(X_val)[:, 1]

    def _predict_proba(arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(arr, columns=cols)
        # Decode integer-coded categoricals back to their original raw values.
        for i, c in enumerate(cols):
            if c in cat_cols:
                inv = inverse_maps[i]
                df[c] = df[c].round().astype("Int64").map(inv)
        # Cast known-int numeric cols back to integer dtype to satisfy the pipeline.
        for c in cols:
            if c in X_train.columns and pd.api.types.is_integer_dtype(X_train[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
        return pipe.predict_proba(df)

    explainer = LimeTabularExplainer(
        training_data=X_bg_enc.to_numpy(dtype=float),
        feature_names=cols,
        class_names=["no_readmit", "readmit"],
        categorical_features=cat_idx,
        categorical_names={i: [str(v) for v in cat_maps[i]] for i in cat_idx},
        discretize_continuous=True,
        random_state=RANDOM_STATE,
        mode="classification",
    )

    order = np.argsort(p_val)
    pick = list(order[-n_each:][::-1]) + list(order[:n_each])
    rows = []
    for rank, i in enumerate(pick):
        risk = "high" if rank < n_each else "low"
        instance = X_val_enc.iloc[int(i)].to_numpy(dtype=float)
        exp = explainer.explain_instance(
            instance, _predict_proba, num_features=n_features, num_samples=1500
        )
        local_rank = rank if rank < n_each else rank - n_each
        html_path = out_dir / f"{risk}_risk_{local_rank}.html"
        exp.save_to_file(html_path.as_posix())
        for feat, weight in exp.as_list():
            rows.append(
                {
                    "risk_bucket": risk,
                    "rank": local_rank,
                    "score": float(p_val[i]),
                    "feature": feat,
                    "weight": weight,
                }
            )
        logger.info(
            "LIME %-4s rank=%d  score=%.4f → %s", risk, local_rank, p_val[i], html_path.name
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "lime_summary.csv", index=False)
    logger.info("LIME complete — %d explanations × %d features each", len(pick), n_features)
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--n-each", type=int, default=3)
    p.add_argument("--n-features", type=int, default=10)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    args = _parse_args()
    run(
        features_path=args.features,
        winner_path=args.winner,
        out_dir=args.out_dir,
        n_each=args.n_each,
        n_features=args.n_features,
    )


if __name__ == "__main__":
    main()
