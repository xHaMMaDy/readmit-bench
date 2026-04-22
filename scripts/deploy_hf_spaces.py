"""Deploy both Hugging Face Spaces (Flask + FastAPI) for readmit-bench.

Run with HF_TOKEN env var set.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

TOKEN = os.environ["HF_TOKEN"]
USER = "Hammady1"
REPO_ROOT = Path(__file__).resolve().parents[1]

api = HfApi(token=TOKEN)


FLASK_README = """---
title: Readmit Bench Dashboard
emoji: 🏥
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: 30-day hospital readmission risk dashboard
---

# readmit-bench — Clinician Dashboard

Interactive Flask dashboard scoring 30-day inpatient readmission risk.

- **Model**: Calibrated XGBoost (Optuna-tuned, isotonic-calibrated)
- **Cohort**: CMS DE-SynPUF synthetic Medicare claims (~1.33M encounters)
- **API mirror**: <https://huggingface.co/spaces/Hammady1/readmit-bench-api>
- **Source**: <https://github.com/xHaMMaDy/readmit-bench>

This Space runs `flask_app.py` behind gunicorn on port 7860.
"""

API_README = """---
title: Readmit Bench API
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FastAPI scoring service for readmission risk
---

# readmit-bench — FastAPI Service

REST API for 30-day inpatient readmission risk scoring.

- `GET  /health` — liveness + model metadata
- `POST /predict` — score one or more encounters (see `/docs` for schema)
- `GET  /docs` — Swagger UI
- `GET  /openapi.json` — OpenAPI 3 schema

**Model**: Calibrated XGBoost on CMS DE-SynPUF.
**Dashboard**: <https://huggingface.co/spaces/Hammady1/readmit-bench>
**Source**: <https://github.com/xHaMMaDy/readmit-bench>
"""


def stage_flask(stage: Path) -> None:
    """Copy files needed by the Flask Space."""
    for item in [
        "src",
        "models",
        "templates",
        "static",
        "reports",
        "flask_app.py",
        "requirements-flask.txt",
        ".dockerignore",
    ]:
        src = REPO_ROOT / item
        if not src.exists():
            print(f"  WARNING: {src} does not exist, skipping")
            continue
        dest = stage / item
        if src.is_dir():
            print(f"  copying {src.name}/ ...")
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.parquet", "*.npz", "tuning",
                "lime", "figures", "*.html", "*.gif", "*.png", "*.log",
            ))
        else:
            print(f"  copying {src.name} ...")
            shutil.copy2(src, dest)
    shutil.copy2(REPO_ROOT / "Dockerfile.flask", stage / "Dockerfile")
    (stage / "README.md").write_text(FLASK_README, encoding="utf-8")


def stage_api(stage: Path) -> None:
    """Copy files needed by the API Space."""
    for item in ["src", "models", "requirements-api.txt", ".dockerignore"]:
        src = REPO_ROOT / item
        if not src.exists():
            continue
        dest = stage / item
        if src.is_dir():
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc",
            ))
        else:
            shutil.copy2(src, dest)
    shutil.copy2(REPO_ROOT / "Dockerfile", stage / "Dockerfile")
    (stage / "README.md").write_text(API_README, encoding="utf-8")


def deploy(name: str, stager) -> str:
    repo_id = f"{USER}/{name}"
    print(f"\n=== {repo_id} ===")
    create_repo(
        repo_id=repo_id,
        token=TOKEN,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "stage"
        stage.mkdir()
        stager(stage)
        size_mb = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"  staged {size_mb:.1f} MB")
        upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="space",
            token=TOKEN,
            commit_message="deploy: initial upload",
        )
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"  -> {url}")
    return url


if __name__ == "__main__":
    flask_url = deploy("readmit-bench", stage_flask)
    api_url = deploy("readmit-bench-api", stage_api)
    print("\n=== DONE ===")
    print(f"Flask dashboard: {flask_url}")
    print(f"FastAPI service: {api_url}")
