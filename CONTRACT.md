# 📜 CONTRACT — Locked Decisions Before Building

> This file freezes the design decisions that drive every downstream phase.
> Changing anything here forces re-validation of all later work. Treat as immutable after Phase 0.

---

## 1. Problem Definition

**Task**: Binary classification — predict whether an inpatient hospital admission will be followed by an **unplanned readmission within 30 days**.

**Unit of analysis**: One **index inpatient encounter** (one row = one admission).

**Population**: Adult Medicare beneficiaries with at least one inpatient admission during 2008–2010, drawn from CMS DE-SynPUF synthetic claims (5 samples).

---

## 2. Label Rule (`y`)

For each index inpatient claim with admission date `T_admit` and discharge date `T_discharge`:

```
y = 1  IF EXISTS another inpatient claim for the same beneficiary with
         T_admit_next  IN  (T_discharge, T_discharge + 30 days]
         AND  it is "unplanned" (see exclusions)
y = 0  OTHERWISE
```

**Excluded from positive label** (counted as y=0, not as positive):
- Planned admissions (best-effort: same primary diagnosis chapter + same day-of-month suggesting routine; documented heuristic)
- Transfers (admit date == discharge date of prior, same facility code)

**Excluded from cohort entirely** (dropped):
- Beneficiary died during the index admission (`BENE_DEATH_DT` between `T_admit` and `T_discharge`)
- Beneficiary died within 30 days of discharge with no readmission (censored — cannot observe outcome)
- Index admission length-of-stay > 365 days (data error)
- Beneficiary younger than 18 (DE-SynPUF should be 65+ but defensive)

---

## 3. Leakage Policy

**Hard rule**: At training/inference time, only features known **at or before `T_admit`** of the index encounter are allowed.

**Forbidden features**:
- Anything from the index admission itself except admit-time fields (admit date, admit dx, admit source, age at admit, sex, race, prior chronic condition flags)
- Length of stay of the index admission (only known post-discharge)
- Discharge disposition of the index admission
- Any future claim date or amount

**Allowed historical features** (lookback window = 6 months prior to `T_admit`):
- Count of prior inpatient admissions
- Count of prior outpatient encounters
- Count of distinct prior diagnosis chapters (ICD-9)
- Total prior inpatient days
- Days since last inpatient discharge (or `NaN` if first)
- Chronic condition flags from beneficiary summary (these are annual flags — use the year of `T_admit`)

---

## 4. Splits

- **Stratified 70 / 15 / 15** (train / val / test) on `y`
- Split by **beneficiary ID** (no beneficiary appears in two splits) to avoid leakage across encounters of the same person
- Subgroup columns preserved through split: `age_bin`, `sex`, `race`

---

## 5. Subgroups for Fairness Audit

| Column | Bins / Values |
|---|---|
| `age_bin` | `<65`, `65–74`, `75–84`, `85+` |
| `sex` | `Male`, `Female` |
| `race` | `White`, `Black`, `Hispanic`, `Other`, `Unknown` (DE-SynPUF code mapping) |

---

## 6. Metrics

**Primary (model selection)**: **PR-AUC** (Average Precision)
**Secondary**: ROC-AUC, Recall@Precision=0.30, Brier score, Expected Calibration Error (ECE), Log loss
**Operational**: Recall, Precision, F1 at the cost-optimal threshold (see §7)

---

## 7. Cost Matrix & Threshold

Approximate published costs:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **True Positive (will readmit)** | `-$500` (intervention cost, prevents readmit) | `+$15,000` (readmit happens) |
| **True Negative (won't readmit)** | `+$500` (wasted intervention) | `$0` |

Threshold chosen to **minimize expected cost** on the validation set, then frozen for the test set.

---

## 8. API Contract (frozen Pydantic schema)

### `POST /predict`
**Request body** (`PredictionRequest`):
```json
{
  "beneficiary_id": "string",
  "admission_date": "YYYY-MM-DD",
  "age": 72,
  "sex": "Male",
  "race": "White",
  "admit_dx_code": "428",
  "admit_source": "EMERGENCY_ROOM",
  "chronic_conditions": {
    "diabetes": true,
    "chf": true,
    "copd": false,
    "ckd": false,
    "cancer": false
  },
  "prior_6mo_inpatient_count": 2,
  "prior_6mo_outpatient_count": 8,
  "prior_6mo_distinct_dx_chapters": 5,
  "days_since_last_discharge": 45
}
```

**Response body** (`PredictionResponse`):
```json
{
  "readmit_probability": 0.273,
  "predicted_readmit": true,
  "decision_threshold": 0.18,
  "expected_cost_savings_usd": 3870.0,
  "model_version": "v1.0",
  "warnings": []
}
```

### `GET /health`
Returns `{"status": "ok", "model_version": "v1.0"}`.

### `GET /` (HTML)
Embedded single-file demo form.

---

## 9. Frozen Tooling Choices

| Concern | Choice |
|---|---|
| Data engine | **Polars (lazy)** for raw → parquet; **pandas** for downstream ML |
| Tracking | **MLflow local file store** (`./mlruns`) — no server |
| Tuning | **Optuna** TPE sampler, 25 trials × top 3 models |
| Fairness | **Fairlearn** (NOT AIF360 — Windows-friendly) |
| AutoML | **FLAML** (NOT AutoGluon) |
| API | **FastAPI** + Pydantic v2 |
| Demo host | **HuggingFace Spaces (Docker SDK)** |
| Drift | **Evidently AI** |
| Tests | **pytest** (smoke + unit) |
| CI | **GitHub Actions** (lint + test on Ubuntu 3.11) |

---

## 10. Versioning

- `v1.0` tag = end of Phase 12 (V1 ships)
- `v2.0` tag = end of Phase 17 (full project)
- Every model in MLflow tagged with phase + git SHA

---

**Signed off**: Phase 0 — frozen.
