"""Generate notebooks/07_fairness.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "07_fairness.ipynb"

MD_INTRO = """# 07 — Fairness audit (XGBoost winner @ deployment threshold)

**Project:** readmit-bench — 30-day hospital readmission prediction
**Phase:** 9 — Fairness audit (V1 milestone)

---

## Why this matters

A risk model used to allocate clinical resources can be **technically accurate
yet unequal** across patient groups. Equal accuracy is not the only thing that
matters: **equal recall on the costly outcome** (a missed readmission) is the
fairness question that aligns with the cost function we minimised in Phase 7.

## Audit protocol

| Decision | Choice | Why |
|---|---|---|
| Eval set | val-test half (n=99,554) — exact rows used to pick t* | No leakage; same numbers we'd report to operations |
| Sensitive attributes | `sex`, `race`, `age_bin` | Available in DE-SynPUF; standard CMS demographic axes |
| Operating point | calibrator = identity, **t\\* = 0.0320** | The deployed tuple from Phase 7 |
| Per-slice metrics | n, prevalence, PR-AUC, ROC-AUC, recall@top-10%, FNR, FPR, selection rate | Clinical (FNR) and ranking (PR-AUC) views |
| Aggregate gaps | Fairlearn `MetricFrame` → DP-diff, EO-diff, FNR gap | Standard parity statistics |
| Tooling | `fairlearn>=0.13` | Industry-standard fairness toolkit |
"""

SETUP = """import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for lib in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
    logging.getLogger(lib).setLevel(logging.WARNING)

%matplotlib inline
import json
from pathlib import Path
import numpy as np
import pandas as pd

from readmit_bench.viz import apply_style
from readmit_bench.fairness import plots as fair_plots

apply_style()
table = pd.read_csv("../reports/fairness_summary.csv")
pred = pd.read_parquet("../reports/fairness_predictions.parquet")
gaps = json.loads(Path("../reports/fairness_gaps.json").read_text())
print(f"Audit rows: {len(pred):,}  ·  attributes: {list(gaps.keys())}  ·  threshold = 0.0320")
table.head()
"""

SECTIONS = [
    (
        "## 1. Per-slice metrics — full table",
        """display = table.copy()
for c in ["prevalence", "selection_rate", "brier", "pr_auc", "roc_auc", "recall_at_top10", "fnr_at_t", "fpr_at_t"]:
    display[c] = display[c].astype(float)
display.style.format({
    "n": "{:,}", "n_pos": "{:,}", "prevalence": "{:.3f}",
    "selection_rate": "{:.3f}", "brier": "{:.4f}",
    "pr_auc": "{:.3f}", "roc_auc": "{:.3f}",
    "recall_at_top10": "{:.3f}", "fnr_at_t": "{:.3f}", "fpr_at_t": "{:.3f}",
})""",
    ),
    (
        "## 2. Aggregate parity gaps (Fairlearn)",
        """gap_rows = []
for attr, g in gaps.items():
    gap_rows.append({
        "attribute": attr,
        "DP diff": g["demographic_parity_difference"],
        "EO diff": g["equalized_odds_difference"],
        "FNR gap": g["fnr_gap"],
        "worst FNR slice": f"{g['worst_fnr_slice']} ({g['fnr_max']:.3f})",
        "best FNR slice": f"{g['best_fnr_slice']} ({g['fnr_min']:.3f})",
    })
pd.DataFrame(gap_rows).style.format({
    "DP diff": "{:.4f}", "EO diff": "{:.4f}", "FNR gap": "{:.4f}",
})""",
    ),
    (
        "## 3. Plot 26 — Per-slice PR-AUC vs overall",
        "fair_plots.plot_slice_pr_auc(table, Path('../reports/figures'))",
    ),
    (
        "## 4. Plot 27 — FNR by group at t* = 0.0320",
        "fair_plots.plot_fnr_by_group(table, gaps, Path('../reports/figures'))",
    ),
    (
        "## 5. Plot 28 — Reliability curves by group",
        "fair_plots.plot_reliability_by_group(pred, Path('../reports/figures'))",
    ),
]

INTERPRET = """## Interpretation

- **Sex:** essentially equal — DP-diff ≈ 0.001, FNR gap ≈ 0.002. No mitigation
  required on this axis.
- **Race / ethnicity:** small but real FNR gap (~3 pp). The "Other" race slice
  has both the smallest sample (n=3,313) and the highest FNR — partly noise,
  partly genuinely lower selection rate. A real deployment would either lower
  the threshold for under-flagged groups (group-specific t*) or surface this
  in monitoring.
- **Age band:** the 65–74 group has the highest FNR even though prevalence
  there is the lowest — the model under-flags younger Medicare patients.
  Worth investigating whether age-specific features (e.g. specific chronic
  combos common in 65–74) are missing from the feature set.

## Phase-9 outputs (committed)

| Path | Purpose |
|---|---|
| `reports/fairness_summary.csv`         | per-slice metric table (long format) |
| `reports/fairness_gaps.json`           | per-attribute Fairlearn parity gaps + worst slices |
| `reports/fairness_predictions.parquet` | y / p / yhat + sensitive cols on the test split |
| `reports/figures/26_fairness_slice_pr_auc.{png,pdf}`         | per-slice PR-AUC bars |
| `reports/figures/27_fairness_fnr_by_group.{png,pdf}`         | FNR @ t* with worst-slice highlighted |
| `reports/figures/28_fairness_reliability_by_group.{png,pdf}` | calibration curves per group |
| `mlruns/` (experiment `readmit-bench-fairness`)              | DP/EO/FNR gaps logged per attribute |

## Next → Phase 10 (Drift monitoring)

With model + threshold + interpretability + fairness now documented, the
deployment package is functionally complete. Phase 10 will add a drift
monitoring suite (PSI, KS, distribution shifts) so that the team running the
model in production can detect when the input distribution moves away from
the training period.
"""


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(MD_INTRO))
    nb.cells.append(nbf.v4.new_code_cell(SETUP))
    for md, code in SECTIONS:
        nb.cells.append(nbf.v4.new_markdown_cell(md))
        nb.cells.append(nbf.v4.new_code_cell(code))
    nb.cells.append(nbf.v4.new_markdown_cell(INTERPRET))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
