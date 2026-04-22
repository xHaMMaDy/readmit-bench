"""Smoke tests for the readmit-bench FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from readmit_bench.api.main import app
from readmit_bench.api.schemas import SAMPLE_ENCOUNTER

from .conftest import requires_trained_model

pytestmark = requires_trained_model


def _client() -> TestClient:
    # TestClient triggers the lifespan handler, which loads the predictor.
    return TestClient(app)


def test_health_ok() -> None:
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["calibrator_loaded"] is True


def test_model_info() -> None:
    with _client() as client:
        r = client.get("/model_info")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "readmit-bench"
        assert body["winner_model"]
        assert 0.0 < body["threshold"] < 1.0
        assert body["feature_count"] == 29


def test_predict_single() -> None:
    with _client() as client:
        r = client.post("/predict", json=SAMPLE_ENCOUNTER)
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["decision"] in {"flag", "skip"}
        assert body["risk_band"] in {"low", "moderate", "high", "very_high"}


def test_predict_batch() -> None:
    with _client() as client:
        r = client.post("/predict_batch", json={"encounters": [SAMPLE_ENCOUNTER, SAMPLE_ENCOUNTER]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n"] == 2
        assert len(body["predictions"]) == 2


def test_predict_validation_error() -> None:
    with _client() as client:
        bad = dict(SAMPLE_ENCOUNTER)
        bad.pop("los_days")
        r = client.post("/predict", json=bad)
        assert r.status_code == 422  # pydantic validation
