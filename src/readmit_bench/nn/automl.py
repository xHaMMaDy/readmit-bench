"""FLAML AutoML driver for V2 — runs a time-budgeted search over a
classification estimator pool with PR-AUC as the objective.

Wrapped in an sklearn-style ``fit`` / ``predict_proba`` interface so the V2
training driver can treat it identically to the NN models. The fitted
underlying estimator is exposed as ``automl_.model.estimator`` for inspection
and is what gets pickled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SEED = 42


@dataclass
class FlamlAutoML:
    time_budget: int = 240  # seconds
    estimator_list: tuple[str, ...] = ("lgbm", "xgboost", "rf", "extra_tree", "lrl2")
    metric: str = "ap"  # average precision == PR-AUC
    n_jobs: int = -1

    def __post_init__(self) -> None:
        self.automl_ = None
        self.best_estimator_: str | None = None
        self.best_config_: dict | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FlamlAutoML:
        from flaml import AutoML

        self.automl_ = AutoML()
        settings = {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "task": "classification",
            "estimator_list": list(self.estimator_list),
            "log_file_name": "",  # disable file logging
            "verbose": 1,
            "seed": SEED,
            "n_jobs": self.n_jobs,
            "early_stop": True,
        }
        logger.info(
            "FLAML fit — budget=%ds, estimators=%s, metric=%s",
            self.time_budget,
            list(self.estimator_list),
            self.metric,
        )
        self.automl_.fit(X_train=np.asarray(X), y_train=np.asarray(y), **settings)
        self.best_estimator_ = self.automl_.best_estimator
        self.best_config_ = self.automl_.best_config
        logger.info("FLAML best=%s  config=%s", self.best_estimator_, self.best_config_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.automl_ is None:
            raise RuntimeError("FlamlAutoML is not fitted")
        return self.automl_.predict_proba(np.asarray(X))
