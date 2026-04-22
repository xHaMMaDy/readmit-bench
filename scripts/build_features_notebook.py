"""Generate notebooks/02_features.ipynb from a Python source-of-truth."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_features.ipynb"


MD_INTRO = """# 02 — Feature Engineering & Data Splitting

**Project:** readmit-bench — 30-day hospital readmission prediction
**Phase:** 4 — Features + Split (V1 milestone)

---

This notebook documents the design decisions and shows the artefacts produced
by Phase 4:

1. **Derived features** — encounter-local fields added on top of the cohort
   (chronic_count, admit_month/dow, weekend admit flag, prior-admit indicator).
2. **Train / validation / test split** — *grouped by beneficiary*, *stratified
   on outcome*. The same patient never crosses splits, and every split has the
   same positive rate.
3. **Preprocessing pipeline** — an unfitted scikit-learn `ColumnTransformer`
   that downstream training code fits *inside each CV fold* (so target encoding
   stays leak-free).

### Why these choices

| Decision | Reason |
|---|---|
| Group by `beneficiary_id` | Patients have repeated encounters; rows-from-same-patient leakage inflates held-out scores. |
| Stratify on per-patient ever-positive | Keeps positive prevalence ≈9.6% in every split → metrics are comparable. |
| **Not** time-based split | Phase-3 EDA shows DE-SynPUF positive rate decays 14% → 4% across 2008→2010. A time hold-out would conflate model failure with synthetic-data drift. |
| Target-encode `admit_dx_code` (4 182 levels) and `drg_code` (740 levels) | One-hot would explode dimensionality with little gain for high-cardinality codes. |
| `OneHot(min_frequency=20)` for low-cardinality categoricals | Collapses rare US state codes / dx chapters into "infrequent" instead of polluting the matrix. |
| Build but **do not fit** the preprocessor here | Fitting outside CV would leak target distribution; the factory returns an unfitted `ColumnTransformer`. |
"""

SETUP = """import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for lib in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
    logging.getLogger(lib).setLevel(logging.WARNING)

%matplotlib inline
import matplotlib.pyplot as plt
import polars as pl

from readmit_bench.viz import apply_style
from readmit_bench.features import (
    FeatureSpec,
    add_derived_features,
    assign_splits,
    build_preprocessor,
    summarise_split,
)
from readmit_bench.features import plots as feat_plots

apply_style()
df = pl.read_parquet("../data/processed/features.parquet")
print("features.parquet shape:", df.shape)
df.head(3)
"""

SECTIONS = [
    (
        "## 1. Split summary — sizes & stratification",
        "summarise_split(df).as_table()",
    ),
    (
        "## 2. Plot 11 — Train / Val / Test overview",
        "fig = feat_plots.plot_split_overview(df.to_pandas()); fig",
    ),
    (
        "## 3. Plot 12 — Numeric feature correlations (Spearman, train only)",
        "fig = feat_plots.plot_feature_correlation(df.to_pandas()); fig",
    ),
    (
        "## 4. Feature spec — what goes into the model",
        """spec = FeatureSpec()
print("Numeric         :", spec.numeric)
print("Binary          :", spec.binary)
print("Cat low-card    :", spec.cat_lowcard)
print("Cat high-card   :", spec.cat_highcard)
print("Identifiers (drop):", spec.id_cols)
print("Total feature cols:", len(spec.all_features()))""",
    ),
    (
        "## 5. Preprocessor sanity check (50k train rows)",
        """import time
spec = FeatureSpec()
train = df.filter(pl.col("split") == "train").sample(n=50_000, seed=0)
Xtr = train.select(spec.all_features()).to_pandas()
ytr = train["y"].to_numpy()

pre = build_preprocessor()
t0 = time.time(); pre.fit(Xtr, ytr); t1 = time.time()
Z = pre.transform(Xtr); t2 = time.time()
print(f"fit: {t1-t0:.2f}s | transform: {t2-t1:.2f}s | output shape: {Z.shape}")
print(f"sparsity: {Z.nnz / (Z.shape[0]*Z.shape[1]):.1%} non-zero")
print("first 6 columns :", pre.get_feature_names_out()[:6].tolist())
print("last 4 columns  :", pre.get_feature_names_out()[-4:].tolist())""",
    ),
]

NEXT_STEPS = """---
## Phase-4 outputs (committed)

| Path | Purpose |
|---|---|
| `data/processed/features.parquet` | cohort + derived features + `split` column |
| `data/processed/splits.parquet`   | thin (claim_id, beneficiary_id, split) for cheap joins |
| `reports/split_summary.csv`       | counts + positive rates per split |
| `reports/figures/11_split_overview.{png,pdf}` | split visualisation |
| `reports/figures/12_feature_correlation.{png,pdf}` | numeric feature heatmap |

## Next → Phase 5 (Baselines)

Train 6 baselines (LR, RF, ExtraTrees, HistGB, XGBoost, LightGBM, CatBoost)
on the train split, evaluate on val, log everything to MLflow. Each model
gets the **same** preprocessor instance, fitted *inside* each CV fold.
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
