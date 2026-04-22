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
    
    skip_dirs = {'data', 'mlruns', 'notebooks', 'tests', '.git', 'models/v1', 'models/v2'}
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
    
    print(f"\nUploading Flask Space...")
    api.upload_folder(
        folder_path=str(stage),
        repo_id="Hammady1/readmit-bench",
        repo_type="space",
        token=TOKEN,
        commit_message="redeploy: full repo structure with templates"
    )
    print("✓ Flask Space uploaded\n")
