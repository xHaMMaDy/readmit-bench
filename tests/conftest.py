"""Shared pytest fixtures and skip-conditions for readmit-bench tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
WINNER_THRESHOLD = MODELS_DIR / "winner_threshold.json"
WINNER_CALIBRATOR = MODELS_DIR / "winner_calibrator.joblib"


def _model_artifacts_present() -> bool:
    if not WINNER_THRESHOLD.exists() or not WINNER_CALIBRATOR.exists():
        return False
    import json

    try:
        meta = json.loads(WINNER_THRESHOLD.read_text())
    except Exception:  # noqa: BLE001
        return False
    pipeline = MODELS_DIR / "tuned" / f"{meta.get('winner_model', '')}.joblib"
    return pipeline.exists()


MODEL_ARTIFACTS_AVAILABLE = _model_artifacts_present()

requires_trained_model = pytest.mark.skipif(
    not MODEL_ARTIFACTS_AVAILABLE,
    reason="trained model artifacts not present (run Phases 5–7 first)",
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS_DIR
