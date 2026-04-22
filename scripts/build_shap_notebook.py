"""Generate notebooks/06_shap.ipynb from a Python source-of-truth."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "06_shap.ipynb"


MD_INTRO = """# 06 — SHAP Interpretability (XGBoost winner)

**Project:** readmit-bench — 30-day hospital readmission prediction
**Phase:** 8 — Interpretability (V1 milestone)

---

## What this phase does

A model that scores patient risk needs to be **defensible**: clinicians ask
*"why this patient?"* and program managers ask *"which signals drive the
score in aggregate?"*. SHAP gives both:

- **Global importance** — `mean(|SHAP|)` per feature ranks the drivers.
- **Beeswarm** — shows direction and spread: e.g. "high X always raises risk"
  vs "X has scattered impact".
- **Dependence** — for each top feature, shows how SHAP varies across the
  feature's value range, surfacing non-linear patterns the model learned.

## Protocol

| Decision | Choice | Why |
|---|---|---|
| Explainer | `shap.TreeExplainer(feature_perturbation="tree_path_dependent")` | Exact algorithm for tree ensembles; uses training distribution |
| Sample size | ~5,000 val rows (patient-grouped, stratified) | Big enough for stable global ranks; explainer is O(N · trees · max_depth) |
| Feature space | **Post-preprocessor (105 cols)** | Explanations align with the matrix the booster actually sees |
| Top-K shown | 20 (importance), 15 (beeswarm), 4 (dependence) | Readable, focused on what moves the prediction |
| Tracking | MLflow experiment `readmit-bench-shap` | top1 importance + sample size logged |
"""

SETUP = """import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for lib in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
    logging.getLogger(lib).setLevel(logging.WARNING)

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from readmit_bench.viz import apply_style
from readmit_bench.explainability import plots as shap_plots

apply_style()
top = pd.read_csv("../reports/shap_top_features.csv")
data = np.load("../reports/shap_values.npz", allow_pickle=False)
shap_vals = data["shap_values"]
X_proc = data["X_processed"]
feat_names = list(data["feature_names"])
y_sample = data["y_sample"]
print(f"SHAP matrix: {shap_vals.shape}  ·  {len(feat_names)} processed features  ·  base value {float(data['base_value']):+.4f}")
top.head(15)
"""

SECTIONS = [
    (
        "## 1. Plot 23 — Global feature importance (mean |SHAP|, top-20)",
        "fig = shap_plots.plot_global_importance(top, Path('../reports/figures')); fig",
    ),
    (
        "## 2. Plot 24 — Beeswarm: direction + spread (top-15)",
        "fig = shap_plots.plot_beeswarm(shap_vals, X_proc, feat_names, top, Path('../reports/figures')); fig",
    ),
    (
        "## 3. Plot 25 — Dependence: how does SHAP vary with value? (top-4)",
        "fig = shap_plots.plot_dependence_top4(shap_vals, X_proc, feat_names, top, Path('../reports/figures')); fig",
    ),
    (
        "## 4. Per-feature signed mean (clinical sense check)",
        """signed = top.head(20).copy()
signed["pretty"] = signed["feature"].str.replace(r"^(num|bin|cat_low|cat_high)__", "", regex=True)
signed[["pretty", "mean_abs_shap", "mean_signed_shap"]].rename(
    columns={"pretty": "feature", "mean_abs_shap": "mean|SHAP|", "mean_signed_shap": "mean signed SHAP"}
).style.format({"mean|SHAP|": "{:.4f}", "mean signed SHAP": "{:+.4f}"}).bar(
    subset=["mean|SHAP|"], color="#A6BFFF").bar(
    subset=["mean signed SHAP"], color="#1FB89A", align="zero")""",
    ),
    (
        "## 5. Local explanation — a high-risk and a low-risk encounter",
        """row_pred = (shap_vals.sum(axis=1) + float(data['base_value']))
order = np.argsort(row_pred)
low_idx, high_idx = order[10], order[-10]  # avoid extreme tails
base = float(data['base_value'])

def explain_row(idx, label):
    row = shap_vals[idx]
    contrib = pd.DataFrame({
        "feature": [feat_names[i].replace("num__","").replace("bin__","").replace("cat_low__","").replace("cat_high__","") for i in range(len(row))],
        "value": X_proc[idx],
        "shap": row,
    }).reindex(np.argsort(np.abs(row))[::-1]).head(8)
    pred_logit = base + row.sum()
    pred_prob = 1 / (1 + np.exp(-pred_logit))
    print(f"\\n=== {label} encounter (idx={idx}, true label = {int(y_sample[idx])}) ===")
    print(f"  base log-odds = {base:+.3f}  →  sum SHAP = {row.sum():+.3f}  →  predicted P = {pred_prob:.4f}")
    return contrib.style.format({"value": "{:.3f}", "shap": "{:+.4f}"}).bar(subset=["shap"], color="#FFB39A", align="zero")

explain_row(high_idx, "HIGH-risk")""",
    ),
    (
        "### …and a contrasting LOW-risk encounter",
        "explain_row(low_idx, 'LOW-risk')",
    ),
]

NEXT_STEPS = """---
## Phase-8 outputs (committed)

| Path | Purpose |
|---|---|
| `reports/shap_values.npz`                      | shap_values, X_processed, feature_names, base_value, y_sample |
| `reports/shap_top_features.csv`                | per-feature mean(|SHAP|) + signed mean, ranked |
| `reports/figures/23_shap_global_importance.{png,pdf}` | top-20 global importance bar |
| `reports/figures/24_shap_beeswarm.{png,pdf}`          | top-15 beeswarm with feature-value coloring |
| `reports/figures/25_shap_dependence_top4.{png,pdf}`   | dependence scatters for top-4 features |
| `mlruns/` (experiment `readmit-bench-shap`)           | run with top1 importance + sample size logged |

## Next → Phase 9 (Fairness audit with Fairlearn)

Now that we know **what** the model uses, the next question is **does it use
it equally well across patient subgroups**. Phase 9 will slice the test
predictions by sex, age band, race-ethnicity proxy, and dual-eligibility
status, then report per-slice PR-AUC, recall@top-10%, false-negative rate,
and Fairlearn's standard parity gaps (DP, EO).
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
