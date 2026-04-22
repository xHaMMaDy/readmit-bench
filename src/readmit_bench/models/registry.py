"""Baseline model registry — single source of truth for V1 model zoo.

Each entry returns a fresh, *unfitted* estimator with sensible defaults that
will run on full train (~927K rows × 104 features) within a few minutes on
CPU. Hyperparameter tuning happens in Phase 6.

CatBoost / XGBoost / LightGBM are imported lazily so an install hiccup on
Windows degrades gracefully (the offending model is skipped, others still run).
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def _logreg() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _hist_gb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
    )


def _xgboost():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def _catboost():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=0,
        thread_count=-1,
        allow_writing_files=False,
    )


REGISTRY: dict[str, Callable[[], object]] = {
    "logistic_regression": _logreg,
    "random_forest": _random_forest,
    "extra_trees": _extra_trees,
    "hist_gradient_boosting": _hist_gb,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
}

DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Hist Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def safe_make(name: str):
    """Build an estimator, returning ``None`` if optional deps are missing."""
    factory = REGISTRY[name]
    try:
        return factory()
    except ImportError as exc:
        logger.warning("skipping %s — optional dependency missing: %s", name, exc)
        return None
