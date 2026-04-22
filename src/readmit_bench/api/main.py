"""FastAPI service for readmit-bench: 30-day readmission risk scoring.

Endpoints:
    GET  /            — landing page (model card summary)
    GET  /health      — liveness + model-load status
    GET  /model_info  — winner model metadata (name, threshold, costs)
    POST /predict     — score a single encounter
    POST /predict_batch — score up to 1,000 encounters
    GET  /docs        — auto Swagger UI
    GET  /openapi.json — OpenAPI schema
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from readmit_bench.api.predictor import FEATURE_ORDER, Predictor
from readmit_bench.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Encounter,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
PROJECT_NAME = "readmit-bench"


def _build_predictor() -> Predictor | None:
    models_dir = Path(os.getenv("READMIT_MODELS_DIR", "models"))
    try:
        return Predictor(models_dir=models_dir)
    except Exception:  # noqa: BLE001
        logger.exception("predictor failed to load")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = _build_predictor()
    yield


app = FastAPI(
    title=f"{PROJECT_NAME} API",
    version=API_VERSION,
    description=(
        "Production scoring service for the **readmit-bench** project — "
        "30-day all-cause inpatient readmission risk on CMS DE-SynPUF.\n\n"
        "The model is a tuned **XGBoost** pipeline with leak-free preprocessing "
        "(median imputation, target encoding, one-hot). Operating threshold is "
        "cost-optimal under FN=$15,000 and FP=$500. See `/model_info` for details."
    ),
    lifespan=lifespan,
)


def _get_predictor() -> Predictor:
    pred = app.state.predictor
    if pred is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return pred


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing() -> str:
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{PROJECT_NAME} API</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
       max-width: 760px; margin: 4rem auto; padding: 0 1.25rem; color: #111827; line-height: 1.5; }}
h1 {{ font-size: 1.75rem; margin-bottom: .25rem; }}
.sub {{ color: #6B7280; margin-top: 0; }}
code {{ background: #F3F4F6; padding: 2px 6px; border-radius: 4px; }}
.btn {{ display: inline-block; padding: .55rem 1rem; background: #2E5BFF; color: white;
        border-radius: 6px; text-decoration: none; margin-right: .5rem; }}
.btn.alt {{ background: #111827; }}
ul {{ padding-left: 1.25rem; }}
</style></head>
<body>
<h1>readmit-bench API</h1>
<p class="sub">30-day inpatient readmission risk scoring · v{API_VERSION}</p>
<p>Production scoring service for an end-to-end ML benchmark on CMS DE-SynPUF synthetic
Medicare claims. The winner is a tuned XGBoost pipeline with cost-optimised threshold.</p>
<p>
  <a class="btn" href="/docs">Open Swagger UI</a>
  <a class="btn alt" href="/model_info">Model info</a>
</p>
<h3>Endpoints</h3>
<ul>
  <li><code>POST /predict</code> — score one encounter</li>
  <li><code>POST /predict_batch</code> — up to 1,000 encounters</li>
  <li><code>GET  /health</code> — liveness</li>
  <li><code>GET  /model_info</code> — model metadata</li>
</ul>
</body></html>"""


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pred = app.state.predictor
    return HealthResponse(
        status="ok" if pred is not None else "degraded",
        model_loaded=pred is not None,
        calibrator_loaded=pred is not None and pred.calibrator is not None,
    )


@app.get("/model_info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    pred = _get_predictor()
    return ModelInfo(
        name=PROJECT_NAME,
        version=API_VERSION,
        winner_model=pred.meta.winner_model,
        calibrator=pred.meta.calibrator,
        threshold=pred.meta.threshold,
        cost_fn_usd=pred.meta.cost_fn_usd,
        cost_fp_usd=pred.meta.cost_fp_usd,
        test_n=pred.meta.test_n,
        test_n_pos=pred.meta.test_n_pos,
        feature_count=len(FEATURE_ORDER),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: Encounter) -> PredictionResponse:
    pred = _get_predictor()
    try:
        result = pred.predict([encounter.model_dump()])[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction error: {exc}") from exc
    return PredictionResponse(**result)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    pred = _get_predictor()
    payload = [enc.model_dump() for enc in req.encounters]
    try:
        results = pred.predict(payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("batch prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction error: {exc}") from exc
    return BatchPredictResponse(
        n=len(results),
        predictions=[PredictionResponse(**r) for r in results],
    )
