"""Generate notebooks/04_tuning.ipynb from a Python source-of-truth."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_tuning.ipynb"


MD_INTRO = """# 04 — Hyperparameter Tuning (Optuna)

**Project:** readmit-bench — 30-day hospital readmission prediction
**Phase:** 6 — Tuning (V1 milestone)

---

## What was tuned and why

The Phase-5 baselines clustered very tightly at the top:

| Rank | Model | Baseline PR-AUC |
|---|---|---|
| 1 | CatBoost | 0.1719 |
| 2 | XGBoost | 0.1716 |
| 3 | HistGradientBoosting | 0.1705 |

Within 0.0015 PR-AUC of each other — a margin smaller than the trial-to-trial
variance we'd expect from CV. That's exactly the regime where **honest tuning
matters more than picking the model family**, so all three are tuned with the
same protocol.

## Protocol

| Decision | Choice | Why |
|---|---|---|
| Optimiser | Optuna TPE sampler + MedianPruner | Strong default; pruning kills slow-bad trials early |
| Trials per model | 20 | Enough for TPE to converge on a 6–8-dim space without burning all day |
| CV scheme | 3-fold **StratifiedGroupKFold** on `beneficiary_id` | Patients can't cross folds; stratification keeps positive rate stable |
| Sample size | 300K rows (patient-grouped subsample) | Fast enough per trial; large enough to track real performance |
| Refit | **Best params on full train (927K)** | The number we actually report comes from full-data refit on the val split |
| Tracking | MLflow nested runs (`tuning_parent` → `trial_NNN`) | Every hyperparameter combination is reproducible |
| Primary metric | **PR-AUC** | Same as baselines — class is 9.6% positive |
"""

SETUP = """import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for lib in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
    logging.getLogger(lib).setLevel(logging.WARNING)

%matplotlib inline
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from readmit_bench.viz import apply_style
from readmit_bench.tuning import plots as tune_plots

apply_style()
summary = pd.read_csv("../reports/tuned_summary.csv")
baselines = pd.read_csv("../reports/baselines.csv")
summary
"""

SECTIONS = [
    (
        "## 1. Tuned leaderboard (validation set, full-train refit)",
        """show = summary.sort_values("pr_auc", ascending=False).reset_index(drop=True)
show.style.format({
    "best_cv_pr_auc": "{:.4f}", "pr_auc": "{:.4f}", "roc_auc": "{:.4f}",
    "brier": "{:.4f}", "log_loss": "{:.4f}",
    "recall_at_top10": "{:.3f}", "precision_at_top10": "{:.3f}",
    "tuning_seconds": "{:.0f}", "refit_seconds": "{:.1f}",
}).background_gradient(subset=["pr_auc", "roc_auc", "recall_at_top10"], cmap="Blues")""",
    ),
    (
        "## 2. Plot 17 — Optuna trial history (best-so-far per model)",
        "fig = tune_plots.plot_trial_history(summary, '../reports/tuning'); fig",
    ),
    (
        "## 3. Plot 18 — Hyperparameter importance (fANOVA)",
        "fig = tune_plots.plot_param_importance(summary, '../reports/tuning'); fig",
    ),
    (
        "## 4. Plot 19 — Tuned vs untuned (what did Optuna buy us?)",
        "fig = tune_plots.plot_tuned_vs_baseline(summary, baselines); fig",
    ),
    (
        "## 5. Best hyperparameters per model",
        """for _, row in summary.sort_values("pr_auc", ascending=False).iterrows():
    print(f"=== {row['display_name']}  (val PR-AUC = {row['pr_auc']:.4f}) ===")
    params = row['best_params'] if isinstance(row['best_params'], dict) else json.loads(row['best_params'].replace("'", '"'))
    for k, v in params.items():
        print(f"  {k:>22s} : {v}")
    print()""",
    ),
    (
        "## 6. The winner — pick the model we promote into Phase 7",
        """winner = summary.sort_values("pr_auc", ascending=False).iloc[0]
print(f"Winner: {winner['display_name']}")
print(f"  CV PR-AUC (300K subsample, 3-fold): {winner['best_cv_pr_auc']:.4f}")
print(f"  Val PR-AUC (full-train refit):       {winner['pr_auc']:.4f}")
print(f"  Val ROC-AUC:                         {winner['roc_auc']:.4f}")
print(f"  Val Brier:                           {winner['brier']:.4f}")
print(f"  Val recall @ top-10% risk:           {winner['recall_at_top10']:.3f}")
prevalence = 0.0964
print(f"  Lift over random:                    {winner['pr_auc']/prevalence:.2f}x")""",
    ),
]

NEXT_STEPS = """---
## Phase-6 outputs (committed)

| Path | Purpose |
|---|---|
| `models/tuned/<name>.joblib`             | best-params Pipeline refitted on full train |
| `models/tuned/<name>_val_scores.npy`     | val-set probabilities |
| `reports/tuning/<name>_trials.csv`       | per-trial parameters + CV PR-AUC |
| `reports/tuning/<name>_study.pkl`        | full Optuna study (for re-plotting) |
| `reports/tuned_summary.csv`              | leaderboard — one row per tuned model |
| `reports/tuned_curves.npz`               | PR/ROC/calibration arrays |
| `reports/figures/17_tuning_history.{png,pdf}`         | best-so-far per model |
| `reports/figures/18_tuning_param_importance.{png,pdf}`| fANOVA importances |
| `reports/figures/19_tuned_vs_baseline.{png,pdf}`      | side-by-side gains |
| `mlruns/` (experiment `readmit-bench-tuning`)         | parent + 20 nested runs per model |

## Next → Phase 7 (Calibration + cost-based threshold)

Take the **winning tuned model**, fit isotonic + Platt scaling on the val
split, pick the calibrator with lower Brier, then choose an operating
threshold by minimising expected cost given:
- $15,000 per missed readmission (false negative)
- $500 per intervention (false positive on a flagged but-true-negative case)

Output: a single deployment-ready (model, calibrator, threshold) tuple, plus
Plot 20–22 (reliability before/after, cost surface, confusion matrix at
chosen threshold).
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
