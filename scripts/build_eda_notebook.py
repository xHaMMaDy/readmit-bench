"""Generate notebooks/01_eda.ipynb from a Python source-of-truth."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_eda.ipynb"


MD_INTRO = """# 01 — Exploratory Data Analysis

**Project:** readmit-bench — 30-day hospital readmission prediction
**Dataset:** CMS DE-SynPUF synthetic Medicare claims, 20 samples (2007–2010)
**Cohort size:** ~1.33M inpatient encounters across ~752K beneficiaries

---

This notebook walks through the cohort and label produced in Phase 2
(`readmit_bench.data.cohort`) and surfaces the structure that will drive
Phase-4 feature engineering and model choice.

All plots are rendered with the project's publication-grade style
(`readmit_bench.viz`) and saved as PNG (300 dpi) + vector PDF in
`reports/figures/`.

### Key findings (TL;DR)
1. **Label balance** — ~9.6% positive (30-day readmits): heavily imbalanced, justifies PR-AUC as the primary metric.
2. **Age** — small, mostly flat readmit gradient; age alone is weak.
3. **Sex / race** — small but measurable gaps → mandatory subgroups for the Phase-9 fairness audit.
4. **Length of stay** — strong signal: readmitted patients have notably longer index stays.
5. **Diagnosis chapter** — Respiratory, Infectious, Blood disorders top the rate ranking.
6. **Prior utilisation** — readmit rate triples from ~8% (no prior) to ~24% (≥6 prior stays). Strongest single feature.
7. **Chronic conditions** — Stroke, COPD and Cancer flags carry the highest conditional rate (~13–14%).
8. **Temporal pattern** — both volume and rate decline through 2009–2010 (known DE-SynPUF synthesis property). Random/group split (by beneficiary) preferred over time-based split.
9. **Missingness** — only the day-since-last-discharge engineered field has meaningful NA share (first-encounter patients); raw fields essentially complete.
"""

SETUP = """import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for lib in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
    logging.getLogger(lib).setLevel(logging.WARNING)

%matplotlib inline
import matplotlib.pyplot as plt

from readmit_bench.viz import apply_style
from readmit_bench.eda import plots as eda

apply_style()
df = eda.load_cohort("../data/processed/cohort.parquet")
print("shape:", df.shape)
df.head(3)
"""

SECTIONS = [
    ("## 1. Label balance", "fig = eda.plot_label_balance(df); fig"),
    ("## 2. Age distribution & rate by age", "fig = eda.plot_age_distribution(df); fig"),
    ("## 3. Readmission rate by sex", "fig = eda.plot_sex_rate(df); fig"),
    ("## 4. Readmission rate by race (fairness preview)", "fig = eda.plot_race_rate(df); fig"),
    ("## 5. Length of stay by class", "fig = eda.plot_los_distribution(df); fig"),
    ("## 6. Top admit-diagnosis chapters", "fig = eda.plot_dx_chapter_rates(df); fig"),
    ("## 7. Prior 6-month inpatient utilisation", "fig = eda.plot_prior_count_vs_rate(df); fig"),
    ("## 8. Chronic-condition flags", "fig = eda.plot_chronic_conditions(df); fig"),
    ("## 9. Volume & rate over time", "fig = eda.plot_monthly_volume_and_rate(df); fig"),
    ("## 10. Missingness", "fig = eda.plot_missingness(df); fig"),
]

NEXT_STEPS = """---
## Next steps → Phase 4 (Features)
1. **Split strategy:** stratified by `y`, **grouped by `beneficiary_id`** (no leakage between train/val/test).
2. **Feature pipeline:** numeric (impute median + standard-scale), categorical (impute mode + target encode for high-cardinality, one-hot for low).
3. **Engineered features to keep:** `los_days`, `prior_6mo_inpatient_count`, `num_diagnoses`, `num_procedures`, `age_at_admit`, all chronic flags, `admit_dx_chapter`, `sex`, `race`, `state_code`, `days_since_last_discharge`.
4. **Drop / quarantine for fairness audit only:** raw `race` (audit), raw `sex` (audit) — but keep encoded versions for model.
5. **Hold race / sex columns out of the API contract** (V1 schema, see `CONTRACT.md`).
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
