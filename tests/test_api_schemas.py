"""Unit tests for the API layer that don't require trained model artifacts.

These complement ``test_api.py`` (which is integration-style and gated on
artifacts being present). The schema + landing-page tests run on every CI
build so we always catch breaking contract changes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from readmit_bench.api.main import app
from readmit_bench.api.predictor import FEATURE_ORDER, _risk_band
from readmit_bench.api.schemas import (
    SAMPLE_ENCOUNTER,
    BatchPredictRequest,
    Encounter,
    PredictionResponse,
)

# ----------------------------------------------------------------------
# Pure schema tests (no model needed)
# ----------------------------------------------------------------------


def test_sample_encounter_validates() -> None:
    enc = Encounter.model_validate(SAMPLE_ENCOUNTER)
    dumped = enc.model_dump()
    assert set(dumped) == set(SAMPLE_ENCOUNTER)


def test_encounter_rejects_negative_los() -> None:
    bad = dict(SAMPLE_ENCOUNTER, los_days=-1)
    with pytest.raises(ValidationError):
        Encounter.model_validate(bad)


def test_encounter_rejects_invalid_month() -> None:
    bad = dict(SAMPLE_ENCOUNTER, admit_month=13)
    with pytest.raises(ValidationError):
        Encounter.model_validate(bad)


def test_encounter_rejects_missing_field() -> None:
    bad = {k: v for k, v in SAMPLE_ENCOUNTER.items() if k != "drg_code"}
    with pytest.raises(ValidationError):
        Encounter.model_validate(bad)


def test_batch_request_size_bounds() -> None:
    # at least 1
    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate({"encounters": []})
    # at most 1000
    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate({"encounters": [SAMPLE_ENCOUNTER] * 1001})


def test_prediction_response_decision_enum() -> None:
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(
            {"probability": 0.1, "threshold": 0.05, "decision": "maybe", "risk_band": "low"}
        )


def test_feature_order_matches_pipeline_signature() -> None:
    """Schema field order should match what the saved pipeline expects."""
    schema_fields = {*Encounter.model_fields.keys()}
    expected = set(FEATURE_ORDER)
    assert schema_fields == expected, schema_fields ^ expected


def test_risk_band_buckets() -> None:
    assert _risk_band(0.01) == "low"
    assert _risk_band(0.10) == "moderate"
    assert _risk_band(0.20) == "high"
    assert _risk_band(0.50) == "very_high"


# ----------------------------------------------------------------------
# Public surface tests (no model needed — TestClient short-circuits)
# ----------------------------------------------------------------------


def test_landing_page_renders() -> None:
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "readmit-bench" in r.text.lower()


def test_openapi_schema_exposes_endpoints() -> None:
    with TestClient(app) as client:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        paths = set(spec["paths"].keys())
        assert {"/health", "/model_info", "/predict", "/predict_batch"} <= paths


def test_health_returns_status_either_way() -> None:
    """Health works even when model fails to load (degraded but 200)."""
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in {"ok", "degraded"}
        assert isinstance(body["model_loaded"], bool)
