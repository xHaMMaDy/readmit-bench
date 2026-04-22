# readmit-bench API — production image for HuggingFace Spaces (SDK: docker)
# Exposes FastAPI on port 7860 (HF Spaces convention).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    READMIT_MODELS_DIR=/app/models \
    PORT=7860

# System deps for xgboost / lightgbm / catboost wheels at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached on source-only changes).
COPY requirements-api.txt ./requirements-api.txt
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Copy source + model artifacts.
COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app/src

EXPOSE 7860

# HF Spaces requires the app to listen on 0.0.0.0:7860.
CMD ["uvicorn", "readmit_bench.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
