"""Generate notebooks/03_baselines.ipynb from a Python source-of-truth."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_baselines.ipynb"


MD_INTRO = """# 03 — Baseline Benchmark

**Project:** readmit-bench — 30-day hospital readmission prediction
**Phase:** 5 — Baselines (V1 milestone)

---

This notebook documents the **untuned** baseline benchmark that establishes
the floor every later experiment must beat.

### What was trained

Seven model families, each with one well-known sensible default configuration,
each fitted on the **full train split** (927,893 encounters) and evaluated on
the **val split** (198,597 encounters):

| # | Model | Library |
|---|---|---|
| 1 | Logistic Regression  | scikit-learn (lbfgs) |
| 2 | Random Forest        | scikit-learn |
| 3 | Extra Trees          | scikit-learn |
| 4 | Histogram GBM        | scikit-learn |
| 5 | XGBoost              | xgboost (hist) |
| 6 | LightGBM             | lightgbm |
| 7 | CatBoost             | catboost |

### What is honest about this comparison

* **Same preprocessor for everyone** — one `ColumnTransformer` (numeric scale,
  binary cast, low-card OneHot, high-card TargetEncoder) is fitted *inside the
  pipeline* for each model. No leakage, no shortcuts.
* **No tuning** — every model uses the defaults defined in `models/registry.py`.
  Tuning happens in Phase 6, on top-3 only.
* **Imbalance-aware reporting** — primary metric is **PR-AUC** because the
  positive class is ~9.6%. ROC-AUC, Brier, log-loss, and operational metrics
  (recall / precision at top-10% risk) are also reported.
* **Tracked** — every run is logged to a local MLflow file store (`mlruns/`)
  with parameters, metrics, and timing.
"""

SETUP = """import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for lib in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
    logging.getLogger(lib).setLevel(logging.WARNING)

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from readmit_bench.viz import apply_style
from readmit_bench.models import plots as model_plots

apply_style()
leaderboard = pd.read_csv("../reports/baselines.csv")
curves = dict(np.load("../reports/baselines_curves.npz"))
leaderboard
"""

SECTIONS = [
    (
        "## 1. Leaderboard — sorted by PR-AUC",
        """show = leaderboard.sort_values("pr_auc", ascending=False).reset_index(drop=True)
show.style.format({
    "pr_auc": "{:.4f}", "roc_auc": "{:.4f}", "brier": "{:.4f}", "log_loss": "{:.4f}",
    "recall_at_top10": "{:.3f}", "precision_at_top10": "{:.3f}",
    "fit_time_s": "{:.1f}", "predict_time_s": "{:.2f}",
}).background_gradient(subset=["pr_auc", "roc_auc", "recall_at_top10"], cmap="Blues")""",
    ),
    (
        "## 2. Plot 13 — Leaderboard bar chart (4 metrics)",
        "fig = model_plots.plot_leaderboard(leaderboard); fig",
    ),
    (
        "## 3. Plot 14 — Precision–Recall curves (the right view for imbalanced)",
        "fig = model_plots.plot_pr_curves(leaderboard, curves, prevalence=0.0964); fig",
    ),
    (
        "## 4. Plot 15 — ROC curves",
        "fig = model_plots.plot_roc_curves(leaderboard, curves); fig",
    ),
    (
        "## 5. Plot 16 — Calibration / reliability per model",
        "fig = model_plots.plot_calibration(leaderboard, curves); fig",
    ),
    (
        "## 6. Reading the numbers — what to focus on",
        """top = leaderboard.sort_values("pr_auc", ascending=False).iloc[0]
prevalence = 0.0964
lift = top["pr_auc"] / prevalence
print(f"Best model:        {top['display_name']}")
print(f"Best PR-AUC:       {top['pr_auc']:.4f}  (lift over random = {lift:.2f}×)")
print(f"Best ROC-AUC:      {top['roc_auc']:.4f}")
print(f"Recall @ top-10%:  {top['recall_at_top10']:.3f}  (i.e. capturing this share of true readmits by flagging the riskiest 10% of stays)")
print(f"Brier score:       {top['brier']:.4f}  (lower = better-calibrated probabilities)")""",
    ),
]

NEXT_STEPS = """---
## Phase-5 outputs (committed)

| Path | Purpose |
|---|---|
| `models/baselines/<name>.joblib`         | fitted Pipeline (preprocessor + estimator) |
| `models/baselines/<name>_val_scores.npy` | cached val-set probabilities |
| `reports/baselines.csv`                  | leaderboard (one row per model) |
| `reports/baselines_curves.npz`           | PR / ROC / calibration arrays for plotting |
| `reports/figures/13_baselines_leaderboard.{png,pdf}` | ranked-bar leaderboard |
| `reports/figures/14_baselines_pr_curves.{png,pdf}`   | PR overlay |
| `reports/figures/15_baselines_roc_curves.{png,pdf}`  | ROC overlay |
| `reports/figures/16_baselines_calibration.{png,pdf}` | reliability curves |
| `mlruns/`                                | MLflow file store (one run per model) |

## Next → Phase 6 (Tuning)

Take the **top 3 by PR-AUC**, run **Optuna** with 25 trials each and 3-fold
CV on a 500K-row stratified-grouped sample, then refit the chosen
hyperparameters on the full train split. Compare against the baseline number
in this notebook to quantify the lift from tuning.
"""


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(MD_INTRO))
    nb.cells.append(nbf.v4.new_code_cell(SETUP))
    for md, code in SECTIONS:
        nb.cells.append(nbf.v4.new_markdown_cell(md))
        nb.cells.append(nbf.v4.new_code_cell(code))
    nb.cells.append(nbf.v4.new_markdown_cell(NEXT_STEPS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}")
