"""Redeploy Flask Space with complete repo structure."""
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

with tempfile.TemporaryDirectory() as td:
    stage = Path(td) / "flask_stage"
    stage.mkdir()
    
    skip_dirs = {'data', 'mlruns', 'notebooks', 'tests', '.git', 'models/v1', 'models/v2', 'venv', '.venv', 'env', '.git', '__pycache__'}
    skip_patterns = ('__pycache__', '*.pyc', '.pytest_cache', '.ruff_cache', 
                     '*.parquet', '*.npz', '*.log')
    
    for item in REPO_ROOT.iterdir():
        if item.name.startswith('.') and item.name not in {'.dockerignore', '.gitignore'}:
            continue
        if item.name in skip_dirs:
            continue
        
        if item.is_dir():
            shutil.copytree(item, stage / item.name, 
                          ignore=shutil.ignore_patterns(*skip_patterns))
            print(f"  {item.name}/")
        else:
            shutil.copy2(item, stage / item.name)
    
    templates_check = list((stage / 'templates').glob('*'))
    print(f"\n✓ templates/ staged: {templates_check}")
    
    # Create HF Spaces README with front matter
    hf_readme = """---
title: Readmit Bench Dashboard
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: flask_app.py
pinned: false
---

# 📊 Readmit Bench: 30-Day Hospital Readmission Risk Dashboard

Interactive medical SaaS dashboard for predicting hospital readmission risk using machine learning.

**Live API**: [API Documentation](https://huggingface.co/spaces/Hammady1/readmit-bench-api/blob/main/docs)

## Features
- Real-time patient risk stratification
- XGBoost-based prediction pipeline
- Isotonic calibration for reliability
- Web-based interface with medical UX

## Usage
1. Enter patient demographics and clinical features
2. Submit for prediction
3. View risk score with calibrated confidence
"""
    (stage / 'README.md').write_text(hf_readme)
    print("✓ Created HF Spaces README with front matter")
    
    print(f"\nUploading Flask Space...")
    api.upload_folder(
        folder_path=str(stage),
        repo_id="Hammady1/readmit-bench",
        repo_type="space",
        token=TOKEN,
        commit_message="redeploy: full repo structure with templates"
    )
    print("✓ Flask Space uploaded\n")
