---
title: readmit-bench API
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 30-day readmission risk scoring (XGBoost + cost-optimal threshold)
---

# readmit-bench API — HuggingFace Space

Production inference service for the **readmit-bench** project: 30-day all-cause
inpatient readmission risk on CMS DE-SynPUF synthetic Medicare claims.

## Stack

- **Model** — tuned XGBoost pipeline (sklearn `Pipeline` with leak-free
  preprocessing: median imputation, one-hot for low-card cats, target encoding
  for high-card cats).
- **Calibration** — chosen via Brier minimisation on a held-out half of val
  (uncalibrated / isotonic / Platt). Uncalibrated wins for XGBoost.
- **Threshold** — cost-optimal under FN = $15,000 and FP = $500.
- **Serving** — FastAPI + uvicorn, single-process, model loaded at startup.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/`              | Landing page |
| GET  | `/docs`          | Auto Swagger UI |
| GET  | `/health`        | Liveness + model-load status |
| GET  | `/model_info`    | Winner model metadata |
| POST | `/predict`       | Score a single encounter |
| POST | `/predict_batch` | Score up to 1,000 encounters |

## Deploy

This Space is built from the `Dockerfile` at the repo root. To deploy
manually:

```bash
# 1. Create a new Space on HuggingFace (SDK: Docker)
# 2. Push this repo to the Space remote
git remote add space https://huggingface.co/spaces/<your-user>/readmit-bench
git push space main
```

## Sample request

```bash
curl -X POST http://localhost:7860/predict \
  -H "content-type: application/json" \
  -d @scripts/sample_payload.json
```
