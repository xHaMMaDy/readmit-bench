"""Model loader + scoring utilities for the readmit-bench prediction API.

Loads:
    - the saved sklearn Pipeline (preprocessor + classifier) for the winner model
    - the chosen calibrator wrapper (identity / isotonic / Platt)
    - the cost-optimal operating threshold + metadata

A single ``Predictor`` instance is created at app startup and reused across
requests. It is process-local — no shared state across workers needed.
"""

from __future__ import annotations

import json
import logging
import sys as _sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from readmit_bench.calibration import calibrate as _calibrate
from readmit_bench.features.pipeline import (
    BINARY_COLS,
    CAT_HIGHCARD_COLS,
    CAT_LOWCARD_COLS,
    NUMERIC_COLS,
)

# The calibrator was pickled while ``calibrate.py`` was executing as ``__main__``,
# so joblib needs to find the wrapper classes there. Register them on the
# real ``__main__`` module before any unpickling happens.
_main = _sys.modules.get("__main__")
if _main is not None:
    for _cls_name in ("_IdentityWrapper", "_IsoWrapper", "_PlattWrapper"):
        if not hasattr(_main, _cls_name):
            setattr(_main, _cls_name, getattr(_calibrate, _cls_name))

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_WINNER_THRESHOLD = DEFAULT_MODELS_DIR / "winner_threshold.json"
DEFAULT_CALIBRATOR = DEFAULT_MODELS_DIR / "winner_calibrator.joblib"
DEFAULT_TUNED_DIR = DEFAULT_MODELS_DIR / "tuned"

FEATURE_ORDER: tuple[str, ...] = (
    *NUMERIC_COLS,
    *BINARY_COLS,
    *CAT_LOWCARD_COLS,
    *CAT_HIGHCARD_COLS,
)


def _risk_band(p: float) -> str:
    if p < 0.05:
        return "low"
    if p < 0.15:
        return "moderate"
    if p < 0.35:
        return "high"
    return "very_high"


@dataclass
class PredictorMeta:
    winner_model: str
    calibrator: str
    threshold: float
    cost_fn_usd: float
    cost_fp_usd: float
    test_n: int
    test_n_pos: int


class Predictor:
    """Loads the production model pipeline + calibrator and serves predictions."""

    def __init__(
        self,
        models_dir: Path = DEFAULT_MODELS_DIR,
        threshold_path: Path | None = None,
        calibrator_path: Path | None = None,
    ) -> None:
        models_dir = Path(models_dir)
        threshold_path = threshold_path or (models_dir / "winner_threshold.json")
        calibrator_path = calibrator_path or (models_dir / "winner_calibrator.joblib")

        if not threshold_path.exists():
            raise FileNotFoundError(f"winner_threshold.json not found at {threshold_path}")

        meta = json.loads(threshold_path.read_text())
        winner_name = meta["winner_model"]
        pipeline_path = models_dir / "tuned" / f"{winner_name}.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"winner pipeline not found: {pipeline_path}")
        if not calibrator_path.exists():
            raise FileNotFoundError(f"calibrator not found: {calibrator_path}")

        logger.info("loading pipeline: %s", pipeline_path)
        self.pipeline = joblib.load(pipeline_path)
        logger.info("loading calibrator: %s", calibrator_path)
        self.calibrator = joblib.load(calibrator_path)

        self.meta = PredictorMeta(
            winner_model=winner_name,
            calibrator=meta.get("calibrator", "uncalibrated"),
            threshold=float(meta["threshold"]),
            cost_fn_usd=float(meta["cost_fn"]),
            cost_fp_usd=float(meta["cost_fp"]),
            test_n=int(meta["test_n"]),
            test_n_pos=int(meta["test_n_pos"]),
        )
        logger.info(
            "predictor ready (model=%s, calibrator=%s, threshold=%.4f)",
            self.meta.winner_model,
            self.meta.calibrator,
            self.meta.threshold,
        )

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _to_frame(self, encounters: Iterable[dict]) -> pd.DataFrame:
        df = pd.DataFrame(list(encounters))
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            raise ValueError(f"missing required feature columns: {missing}")
        df = df[list(FEATURE_ORDER)].copy()
        # Match training dtypes: pandas Pipeline expects strings for cats and
        # numeric/bool dtypes for the rest. Pydantic already enforced types.
        for col in BINARY_COLS:
            df[col] = df[col].astype(bool)
        for col in NUMERIC_COLS:
            df[col] = df[col].astype("float64")
        for col in (*CAT_LOWCARD_COLS, *CAT_HIGHCARD_COLS):
            df[col] = df[col].astype(str)
        return df

    def predict_proba(self, encounters: Iterable[dict]) -> np.ndarray:
        X = self._to_frame(encounters)
        raw = self.pipeline.predict_proba(X)[:, 1]
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def predict(self, encounters: Iterable[dict]) -> list[dict]:
        probs = self.predict_proba(encounters)
        thr = self.meta.threshold
        out: list[dict] = []
        for p in probs:
            p = float(p)
            out.append(
                {
                    "probability": p,
                    "threshold": thr,
                    "decision": "flag" if p >= thr else "skip",
                    "risk_band": _risk_band(p),
                }
            )
        return out
