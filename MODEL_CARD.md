# Model Card — readmit-bench v1.0

Responsible-AI documentation for the deployed 30-day inpatient readmission risk
model shipped in **readmit-bench v1.0**. Follows the structure of
[Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019)](https://arxiv.org/abs/1810.03993).

---

## 1. Model Details

| Field | Value |
|---|---|
| **Name** | `readmit-bench-xgboost-v1` |
| **Version** | 1.0 |
| **Type** | Binary classifier — probability of unplanned 30-day inpatient readmission |
| **Architecture** | Gradient-boosted decision trees (XGBoost) inside a scikit-learn `Pipeline(ColumnTransformer + XGBClassifier)` |
| **Hyperparameters (tuned via Optuna, 20 trials)** | `n_estimators=800, max_depth=4, learning_rate=0.0204, min_child_weight=16, subsample=0.988, colsample_bytree=0.913, reg_lambda=9.64, reg_alpha=0.234` |
| **Calibrator** | None (identity wrapper). XGBoost's raw probabilities had the lowest Brier on the held-out calib split; isotonic & Platt slightly worsened it. |
| **Operating threshold** | `t* = 0.0320`, selected to minimize expected cost under the cost model below. |
| **Inputs** | 29 features per encounter (9 numeric, 13 binary including 11 chronic-condition flags, 5 low-cardinality categorical, 2 high-cardinality categorical). Full schema: `src/readmit_bench/api/schemas.py::Encounter`. |
| **Outputs** | `probability ∈ [0, 1]`, `decision ∈ {flag, no_flag}` at `t*`, `risk_band ∈ {low, moderate, high, very_high}`. |
| **Training framework** | scikit-learn 1.5, XGBoost 2.x, Optuna 3.x, MLflow 2.x |
| **Random seed** | 42 (data splits, model init, Optuna sampler) |
| **License** | MIT (code) — model weights are derived from public CMS DE-SynPUF data (public domain). |
| **Author / contact** | HaMMaDy — github.com/xHaMMaDy |
| **Date** | 2026-04 |

Saved artifacts (in `models/`):
- `tuned/xgboost.joblib` — full fitted Pipeline (preprocessor + booster)
- `winner_calibrator.joblib` — identity calibrator wrapper
- `winner_threshold.json` — chosen threshold + cost metadata + test-set confusion

---

## 2. Intended Use

### Primary intended use

Demonstration / portfolio model showing an end-to-end production-grade workflow:
hospital readmission risk scoring on synthetic Medicare claims, deployed as a
FastAPI service with reproducible training, calibration, fairness audit, and CI.

### Intended users

- Hiring managers and engineers reviewing the author's senior-ML-engineer portfolio.
- ML practitioners studying a healthcare-shaped end-to-end pipeline.
- Researchers wanting a worked example of cost-based threshold selection +
  Fairlearn audit on a public synthetic dataset.

### Out-of-scope use (do **not** do this)

- **Any clinical decision-making.** This model is trained on synthetic claims
  (CMS DE-SynPUF) and has not been validated against any real patient outcomes,
  any institutional standard of care, or any regulatory requirement (FDA, MDR,
  HIPAA, etc.). It must not be used to triage, treat, or otherwise inform care
  for real patients.
- **Resource allocation, insurance underwriting, or eligibility decisions.**
- **Transfer to a real-world dataset without re-training and re-validation.**
  The synthetic distribution differs materially from real Medicare claims —
  see "Honest read" notes throughout the README.

---

## 3. Training Data

| Field | Value |
|---|---|
| **Source** | [CMS DE-SynPUF](https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs) — synthetic Medicare claims released by CMS for safe public use. |
| **Sample selection** | All 20 sample files of inpatient claims (2008–2010), joined with beneficiary summary for demographics. |
| **Cohort** | Inpatient encounters with valid admit/discharge dates, age ≥ 65 at admission, LOS ≥ 1 day, with a 30-day post-discharge follow-up window observable in the data. |
| **Cohort size** | 1,326,074 encounters across 752,289 beneficiaries. |
| **Label** | Binary — 1 if the patient has any subsequent inpatient admission within 30 days of discharge (excluding planned/scheduled readmissions where flagged); else 0. |
| **Positive rate** | 9.61% overall (cohort-wide). |
| **Splits** | Patient-grouped (no patient appears in more than one split), stratified on the label. Train 70% / val 15% / test 15% by encounters; **train = 927,893 encounters**, **val = 198,597**, **test = 199,584**. Calibration phase further splits val 50/50 into `calib` (n = 99,043) and `calib_test` (n = 99,554) — same patient-grouped, stratified protocol. |
| **Class imbalance handling** | None at the loss level — XGBoost trained with default log-loss. The cost-based threshold (Section 5) handles the imbalance at decision time. |

### Known data caveats (DE-SynPUF)

DE-SynPUF is **synthetic**. CMS generated it with privacy-preserving
perturbation specifically to allow public release; signal that exists in real
Medicare claims has been deliberately attenuated, and some artifacts have been
introduced. Concretely we observed:

- Top-1 model feature is `chronic_count`, a sum of binary chronic flags — its
  discriminative power may be inflated by the synthetic generation process.
- A single state code (`state_code_9`) appears in the top-3 SHAP features —
  almost certainly a synthesis artifact, not a real geographic signal.
- The sign on `prior_6mo_inpatient_count` is negative (more prior inpatient →
  lower predicted risk), the opposite of the clinical expectation. Possibly a
  confound (heavy utilizers under existing follow-up) or a synthesis artifact.
- All 6 boosted models saturate near **PR-AUC ≈ 0.171**; tuning bought
  essentially zero val lift. The dataset appears to be **capacity-limited, not
  optimization-limited** — there is a hard ceiling on extractable signal.

---

## 4. Evaluation

### Held-out test set (n = 199,584 encounters, 9.6% positive)

For the calibration & threshold phase the val split is split 50/50 — what we
call "test" here is the **calib_test** half (n = 99,554), the half on which
the operating threshold was *not* selected.

| Metric | Value |
|---|---|
| **PR-AUC** | 0.1717 (1.78× over the random baseline of 0.0964) |
| **ROC-AUC** | 0.6849 |
| **Brier score** | 0.0835 |
| **Log-loss** | 0.2980 |
| **Recall @ top-10%** | 0.211 |
| **Recall at deployed threshold** | 0.971 |
| **Precision at deployed threshold** | 0.107 |
| **Confusion at threshold** | TP = 9,282 · FP = 77,391 · FN = 273 · TN = 12,608 |

### Calibration

The deployed model is the **uncalibrated XGBoost output**. On the calib_test
split, isotonic regression and Platt scaling were both evaluated and found to
*slightly worsen* Brier — XGBoost trained with log-loss is already well
calibrated on this cohort. See `reports/figures/20_reliability_before_after.png`.

---

## 5. Cost Model & Threshold Selection

We do not deploy at the default `0.5` threshold. Instead the threshold minimizes
expected cost under an explicit, documented cost model.

| Cost component | Value | Rationale |
|---|---|---|
| **Cost of a missed readmission (FN)** | **$15,000** | Order-of-magnitude estimate of the unreimbursed cost of an avoidable inpatient readmission stay (Medicare HRRP penalty + uncovered care). |
| **Cost of an intervention (TP + FP)** | **$500** | Order-of-magnitude estimate of a post-discharge follow-up call + nurse review. |
| **Cost ratio** | **30 : 1** (FN : intervention) | |

Under this cost model the optimal threshold is **`t* = 0.0320`**, far below
0.5. At that threshold the model flags **87% of encounters** but catches **97%
of true readmissions**, producing:

| Quantity | Value |
|---|---|
| Total cost on test (n = 99,554) | **$47,431,500** |
| Per-encounter cost | **$476.44** |
| Cost of *always-treat* baseline | $49,777,000 (model **saves $2.35M**) |
| Cost of *never-treat* baseline | $143,325,000 (model **saves $95.9M**) |

> A 30:1 cost ratio mathematically pushes toward broad screening with cheap
> interventions. **Operators with a different cost ratio must re-pick the
> threshold** — `models/winner_threshold.json` carries the cost metadata, and
> the cost surface in `reports/figures/21_cost_surface.png` shows how total
> cost changes with the threshold.

---

## 6. Fairness Analysis

Per-slice metrics + Fairlearn aggregate parity gaps at the deployed threshold,
computed on the test split (n = 99,554).

| Attribute | DP diff | EO diff | FNR gap | Worst FNR slice | Best FNR slice |
|---|---|---|---|---|---|
| **sex** | 0.0008 | 0.0019 | 0.0019 | Male (0.030) | Female (0.028) |
| **race** | 0.0455 | 0.0473 | 0.0299 | Other (0.039, n=3,313) | Hispanic (0.010, n=1,996) |
| **age_bin** | 0.0510 | 0.0530 | 0.0168 | 65-74 (0.035) | 85+ (0.018) |

**Honest read.** Sex parity is essentially perfect (DP-diff < 0.001). Race
shows a ~3 pp FNR gap concentrated on the small "Other" slice (n = 3,313); a
real deployment should surface this in monitoring rather than tune the
threshold per slice on a low-n group. Age shows the 65–74 group is
under-flagged despite the lowest absolute prevalence — a feature-engineering
follow-up (age × chronic-combination interactions) is the natural next step.

Artifacts: `reports/fairness_summary.csv`, `reports/fairness_gaps.json`,
`reports/fairness_predictions.parquet` (full row-level scored predictions, for
auditors to recompute).

---

## 7. Limitations

1. **Synthetic training data.** The model has only ever seen CMS DE-SynPUF
   claims. Generalization to real Medicare claims is **unknown and untested**;
   generalization to non-Medicare populations is **expected to be poor** (the
   cohort is age ≥ 65 inpatient-only).
2. **Capacity-limited signal.** All boosted models saturate near PR-AUC 0.172.
   The ceiling on this dataset is low; further model complexity is unlikely to
   help.
3. **Calibration is global, not per-slice.** Per-slice reliability diagrams
   (plot 28) show the model is well calibrated for the largest groups but
   under-confident for some smaller ones. Per-slice calibration is left as
   V2 work.
4. **No temporal validation.** Splits are patient-grouped but not time-ordered.
   A real deployment must validate on a held-out *future* time window.
5. **No drift monitoring in V1.** Evidently drift reports are V2 (Phase 15).
6. **No prospective validation.** No clinician or patient outcomes have been
   used to validate the model's clinical utility.

---

## 8. Ethical Considerations

- **Risk of harm if misused clinically.** Section 2 explicitly forbids clinical
  use. Even with a perfect model the cost-based threshold flags 87% of
  encounters — this is a *screening* decision rule, not a *triage* rule, and
  treating it as triage would over-treat low-risk patients and waste resources.
- **Fairness gap on the "Other" race slice.** Documented in Section 6. A real
  deployment must monitor this slice and define an action threshold *before*
  shipping.
- **Synthetic data ≠ population.** Patterns the model has learned (e.g., a
  single state code in the top-3 features) may not generalize and could
  encode CMS's synthesis biases rather than real population structure.

---

## 9. How to Use

### Python
```python
from readmit_bench.api.predictor import Predictor
from readmit_bench.api.schemas import SAMPLE_ENCOUNTER

predictor = Predictor.load_default()
prob = predictor.predict_proba(SAMPLE_ENCOUNTER)
decision = predictor.predict(SAMPLE_ENCOUNTER)
print(prob, decision)  # e.g. 0.151, 'flag'
```

### HTTP (FastAPI)
```bash
uvicorn readmit_bench.api.main:app --host 0.0.0.0 --port 7860
curl -s -X POST http://127.0.0.1:7860/predict \
     -H "content-type: application/json" \
     -d @scripts/sample_payload.json
```

### Docker
```bash
docker build -t readmit-bench-api .
docker run --rm -p 7860:7860 readmit-bench-api
```

---

## 10. Citation

```
@software{hammady_readmit_bench_2026,
  author  = {HaMMaDy},
  title   = {readmit-bench: end-to-end 30-day inpatient readmission benchmark on CMS DE-SynPUF},
  year    = {2026},
  version = {2.0},
  url     = {https://github.com/xHaMMaDy/readmit-bench}
}
```

---

## 11. V2 Addenda — Honest negative results

V2 (`v2.0`) added neural and AutoML models, ensembles, LIME, and an Evidently
drift report. The deployed FastAPI model did **not** change. The V2 work is
included as a transparent record of what was tried and what did not help.

### 11.1 Neural & AutoML do not beat tuned GBMs (Phase 13)

| Family | Best model | Val PR-AUC | vs deployed XGBoost |
|---|---|---|---|
| Neural net (this work) | PyTorch MLP (4×256 + dropout) | 0.157 | −0.015 |
| Neural net (this work) | FT-Transformer (6 layers) | 0.153 | −0.019 |
| Neural net (this work) | TabNet | 0.153 | −0.019 |
| AutoML | FLAML (300 s budget) | 0.149 | −0.023 |
| Tuned GBM (deployed) | **XGBoost** | **0.172** | — |

*Read*: this is a tabular dataset with ~100 engineered features and ~10%
prevalence — exactly the regime where boosted trees dominate. Reporting the
NN underperformance avoids a "we tried fancy and shipped fancy" framing.

### 11.2 Ensembles add no lift (Phase 14)

Voting and stacking over the tuned XGBoost / CatBoost / HistGB triple, on the
same patient-grouped meta-split as Phase 7 calibration, recover **+0.00007
PR-AUC** over the best base — i.e. zero. Pairwise base-learner correlations
are ρ ≈ 0.99; the stacking LR coefficients land at ~3.6 each (essentially
equal weighting). The deployed single-model XGBoost is the right choice.

### 11.3 LIME local explanations (Phase 15)

Per-encounter LIME explanations are emitted to `reports/lime/*.html` for the
3 highest-risk and 3 lowest-risk val encounters. They corroborate the SHAP
global findings (chronic count, days-since-discharge, prior 6-mo encounter
count, specific state code) at the local level.

### 11.4 Drift baseline (Phase 15)

`reports/drift_report.html` is an Evidently 0.7 `DataDriftPreset` of train vs
val. It is intended as a **monitoring baseline** for production deployment —
no synthetic shift was injected, so it is not a model finding.

### 11.5 Flask dashboard (Phase 16)

`flask_app.py` reuses the deployed `Predictor` for an interactive demo
(form-based scoring + JSON `/api/predict` endpoint + embedded leaderboards,
fairness tables, and Evidently drift report).
**Same model, same threshold, same caveats** as the FastAPI service.
