"""Redeploy API Space with complete repo structure and fresh config."""
import os
import tempfile
import shutil
from pathlib import Path
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).parent.parent
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not TOKEN:
    raise ValueError("HF_TOKEN or HUGGINGFACE_TOKEN env var required")

api = HfApi(token=TOKEN)

API_README = """---
title: Readmit Bench API
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_file: main.py
pinned: false
license: mit
short_description: FastAPI scoring service for readmission risk
---

# readmit-bench — FastAPI Service

REST API for 30-day inpatient readmission risk scoring.

**Dashboard Mirror**: [Flask UI](https://huggingface.co/spaces/Hammady1/readmit-bench)

## Features
- Real-time patient risk inference
- Calibrated XGBoost pipeline
- OpenAPI documentation at `/docs`
- Health check at `/health`

## API Endpoints
- `POST /predict` — Score patient demographics
- `GET /health` — Service health status
- `GET /docs` — Interactive API docs (Swagger)
"""

with tempfile.TemporaryDirectory() as td:
    stage = Path(td) / "api_stage"
    stage.mkdir()
    
    skip_dirs = {'data', 'mlruns', 'notebooks', 'tests', '.git', 'models/v1', 'models/v2', 
                 'venv', '.venv', 'env', '__pycache__', 'templates', 'static', 'flask_app.py'}
    skip_patterns = ('__pycache__', '*.pyc', '.pytest_cache', '.ruff_cache', 
                     '*.parquet', '*.npz', '*.log')
    
    for item in REPO_ROOT.iterdir():
        if item.name.startswith('.') and item.name not in {'.dockerignore', '.gitignore'}:
            continue
        if item.name in skip_dirs or item.name == 'flask_app.py':
            continue
        
        if item.is_dir():
            shutil.copytree(item, stage / item.name, 
                          ignore=shutil.ignore_patterns(*skip_patterns))
            print(f"  {item.name}/")
        else:
            shutil.copy2(item, stage / item.name)
    
    # Create HF Spaces README with fresh config
    (stage / 'README.md').write_text(API_README)
    print("✓ Created HF Spaces README with fresh config")
    
    print(f"\nUploading API Space...")
    api.upload_folder(
        folder_path=str(stage),
        repo_id="Hammady1/readmit-bench-api",
        repo_type="space",
        token=TOKEN,
        commit_message="redeploy: fresh config with polars fix"
    )
    print("✓ API Space uploaded\n")
