"""Generate notebooks/05_calibration.ipynb from a Python source-of-truth."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_calibration.ipynb"


MD_INTRO = """# 05 — Calibration & Cost-Based Threshold

**Project:** readmit-bench — 30-day hospital readmission prediction
**Phase:** 7 — Calibration + clinical cost optimisation (V1 milestone)

---

## What this phase does

The Phase-6 winner — **XGBoost (tuned)** — produces probability scores. Two
questions remain before it's deployable:

1. **Are the probabilities trustworthy?** A model that predicts "12% risk" should
   actually be wrong about 12% of the time on those patients. We test this with
   a **reliability diagram** and the **Brier score**.
2. **Where do we set the operating threshold?** PR-AUC is threshold-free; a
   real intervention program isn't. We pick the threshold that **minimises
   expected dollar cost** to the hospital.

## Protocol

| Decision | Choice | Why |
|---|---|---|
| Holdout strategy | 50 / 50 patient-grouped split of the val set into `calib` and `test` | Calibrator must never see the data it's evaluated on |
| Candidates | **uncalibrated** vs **isotonic** vs **Platt** | Isotonic is non-parametric (flexible), Platt is parametric (smoother on small N); uncalibrated is the honest baseline (XGBoost trained with log-loss is often already calibrated) |
| Selection rule | **Lowest Brier on `test` split** wins | Brier directly penalises miscalibration |
| Cost model | `total = $15,000 · FN + $500 · (TP + FP)` | $15K = avg cost of an unplanned 30-day readmission; $500 = a transitional-care call package |
| Threshold sweep | 501 evenly-spaced points in [0, 1] | Fast and exact for a 1-D objective |
| Tracking | MLflow experiment `readmit-bench-calibration` | Brier, threshold, total cost, confusion all logged |
"""

SETUP = """import json
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for lib in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
    logging.getLogger(lib).setLevel(logging.WARNING)

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from readmit_bench.viz import apply_style
from readmit_bench.calibration import plots as cal_plots

apply_style()
summary = pd.read_csv("../reports/calibration_summary.csv")
threshold = json.loads(open("../models/winner_threshold.json").read())
curves = np.load("../reports/calibration_curves.npz")
summary
"""

SECTIONS = [
    (
        "## 1. Calibration leaderboard (test split — 50% of val, patient-grouped)",
        """summary.style.format({
    "brier": "{:.5f}", "pr_auc": "{:.4f}", "roc_auc": "{:.4f}", "log_loss": "{:.4f}",
}).background_gradient(subset=["brier"], cmap="Blues_r")""",
    ),
    (
        "## 2. Plot 20 — Reliability: uncalibrated vs chosen calibrator",
        "fig = cal_plots.plot_reliability_before_after(curves, summary, threshold['calibrator'], __import__('pathlib').Path('../reports/figures')); fig",
    ),
    (
        "## 3. Plot 21 — Cost surface: where does total $$ bottom out?",
        "fig = cal_plots.plot_cost_surface(curves, threshold, __import__('pathlib').Path('../reports/figures')); fig",
    ),
    (
        "## 4. Plot 22 — Confusion + cost decomposition at chosen threshold",
        "fig = cal_plots.plot_confusion_at_threshold(threshold, __import__('pathlib').Path('../reports/figures')); fig",
    ),
    (
        "## 5. The deployment tuple — model + calibrator + threshold",
        """print(f"Winner model:    {threshold['winner_model']}")
print(f"Calibrator:      {threshold['calibrator']}")
print(f"Threshold:       {threshold['threshold']:.4f}")
print()
print(f"Test split size:           {threshold['test_n']:,}  (pos = {threshold['test_n_pos']:,})")
print(f"Total cost @ threshold:    ${threshold['test_total_cost_usd']:,.0f}")
print(f"Per-encounter cost:        ${threshold['test_cost_per_encounter_usd']:.2f}")
print(f"Cost if always-treat:      ${threshold['test_cost_always_treat_usd']:,.0f}")
print(f"Cost if never-treat:       ${threshold['test_cost_never_treat_usd']:,.0f}")
saved = threshold['test_cost_always_treat_usd'] - threshold['test_total_cost_usd']
print(f"Savings vs always-treat:   ${saved:,.0f}")
print(f"Confusion @ threshold:     {threshold['confusion']}")""",
    ),
]

NEXT_STEPS = """---
## Phase-7 outputs (committed)

| Path | Purpose |
|---|---|
| `models/winner_calibrator.joblib`         | chosen post-hoc calibrator (or identity) |
| `models/winner_threshold.json`            | threshold + cost + confusion (deployment-ready) |
| `reports/calibration_summary.csv`         | Brier / PR-AUC / ROC-AUC per method |
| `reports/calibration_curves.npz`          | reliability + cost-curve arrays for plotting |
| `reports/figures/20_reliability_before_after.{png,pdf}` | reliability uncal vs chosen |
| `reports/figures/21_cost_surface.{png,pdf}`            | cost vs threshold sweep |
| `reports/figures/22_confusion_at_threshold.{png,pdf}`  | confusion + cost decomposition |
| `mlruns/` (experiment `readmit-bench-calibration`)     | run with all metrics + artifacts |

## Next → Phase 8 (Fairness + slice metrics)

With a deployment-ready (model, calibrator, threshold) tuple, the next
question is: **does it fail unequally across patient subgroups?** We will
slice the test predictions by demographic attributes (age band, sex, race
proxy, dual-eligibility, chronic-condition burden) and report per-slice
PR-AUC, recall@top-10%, and false-negative rate — looking for any subgroup
where the model is silently worse.
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
