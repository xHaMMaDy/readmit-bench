"""Optuna search spaces for the top-3 baseline models.

Each ``suggest_*`` function takes an Optuna trial and returns a kwargs dict
suitable for the corresponding scikit-learn / boosting estimator constructor.
Search ranges are chosen to span "sensible" hyperparameters without being
exotic — the goal is honest tuning, not a leaderboard hack.
"""

from __future__ import annotations

from typing import Any

import optuna

RANDOM_STATE = 42


def suggest_catboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        # fixed
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }


def suggest_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        # fixed
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }


def suggest_hist_gradient_boosting(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [128, 255]),
        # fixed
        "random_state": RANDOM_STATE,
        "early_stopping": False,
    }


SPACES = {
    "catboost": suggest_catboost,
    "xgboost": suggest_xgboost,
    "hist_gradient_boosting": suggest_hist_gradient_boosting,
}
