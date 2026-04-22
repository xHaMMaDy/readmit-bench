# readmit-bench — Architecture

End-to-end system architecture for the 30-day readmission risk pipeline.

## High-level flow

```mermaid
flowchart LR
  classDef src   fill:#eef4ff,stroke:#1f6feb,color:#1e3a8a;
  classDef proc  fill:#fffbeb,stroke:#d97706,color:#92400e;
  classDef model fill:#ecfdf5,stroke:#16a34a,color:#14532d;
  classDef serve fill:#fdf2f8,stroke:#be185d,color:#831843;
  classDef store fill:#f1f5f9,stroke:#475569,color:#0f172a;

  CMS[("CMS DE-SynPUF<br/>20 samples · ~1.33M rows")]:::src
  Acquire[["src/data/acquire<br/>Polars ingest"]]:::proc
  Cohort[["src/data/cohort<br/>encounter selection<br/>+ 30-day label"]]:::proc
  Features[["src/features<br/>ColumnTransformer<br/>numeric · cat · chronic"]]:::proc
  Split{{"Beneficiary-disjoint<br/>train / holdout split"}}:::proc

  Bench[["src/models/benchmark<br/>6 baselines"]]:::model
  Tune[["src/models/tune<br/>Optuna TPE × top 3"]]:::model
  V2NN[["src/models/v2_nn<br/>MLP · TabNet · FT-T · FLAML"]]:::model
  V2Ens[["src/models/v2_ensembles<br/>Stacking · Voting"]]:::model
  Calib[["calibration<br/>isotonic + cost threshold"]]:::model

  Eval[["src/evaluation<br/>PR · ROC · Brier · Recall@k"]]:::proc
  Explain[["src/explainability<br/>SHAP · LIME · permutation"]]:::proc
  Fair[["src/fairness<br/>Fairlearn audit"]]:::proc
  Drift[["src/drift<br/>Evidently train vs holdout"]]:::proc

  Artifacts[("models/*.pkl<br/>reports/*.csv · *.png · *.html")]:::store
  MLflow[("mlruns/<br/>tracked runs")]:::store

  API[/"FastAPI service<br/>POST /predict<br/>Dockerfile"/]:::serve
  Dash[/"Flask dashboard<br/>SaaS UI · scoring · leaderboards<br/>Dockerfile.flask"/]:::serve
  HF[/"HuggingFace Spaces"/]:::serve
  CI[/"GitHub Actions CI<br/>ruff · black · mypy · pytest"/]:::serve

  CMS --> Acquire --> Cohort --> Features --> Split
  Split --> Bench --> Tune --> Calib
  Split --> V2NN
  Tune  --> V2Ens
  Calib --> Eval
  Calib --> Explain
  Calib --> Fair
  Split --> Drift

  Bench --> Artifacts
  Tune  --> Artifacts
  V2NN  --> Artifacts
  V2Ens --> Artifacts
  Eval  --> Artifacts
  Explain --> Artifacts
  Fair  --> Artifacts
  Drift --> Artifacts
  Bench -. log .-> MLflow
  Tune  -. log .-> MLflow

  Artifacts --> API
  Artifacts --> Dash
  API --> HF
  Dash --> HF

  CI -. gates .-> Bench
  CI -. gates .-> API
  CI -. gates .-> Dash
```

## Runtime topology — Flask dashboard

```mermaid
sequenceDiagram
  autonumber
  participant U as Clinician
  participant B as Browser (SaaS UI)
  participant F as Flask app (flask_app.py)
  participant P as Predictor (joblib)
  participant A as Artifacts (reports/*)

  U->>B: open http://host:5050
  B->>F: GET /
  F->>A: load leaderboards, fairness, meta
  F-->>B: render index.html (sidebar shell)
  U->>B: fill encounter form, submit
  B->>F: POST / (form data)
  F->>F: build encounter dict + chronic_count + age_bin
  F->>P: predict_proba(encounter)
  P-->>F: {probability, decision, risk_band, threshold}
  F-->>B: render result card (gauge + pills + payload)
```

## Module map

| Layer            | Module                              | Responsibility                                             |
| ---------------- | ----------------------------------- | ---------------------------------------------------------- |
| Data             | `src/readmit_bench/data/`           | CMS download, cohort selection, label engineering          |
| Features         | `src/readmit_bench/features/`       | ColumnTransformer, frequency encoders                      |
| Models           | `src/readmit_bench/models/`         | Benchmark, tuning, V2 NN/AutoML, ensembles, calibration    |
| Evaluation       | `src/readmit_bench/evaluation/`     | PR/ROC/Brier curves, Recall@k, cost surface                |
| Explainability   | `src/readmit_bench/explainability/` | SHAP global + LIME local + permutation                     |
| Fairness         | `src/readmit_bench/fairness/`       | Fairlearn slicing, gaps, mitigation experiments            |
| Drift            | `src/readmit_bench/drift/`          | Evidently report (train vs holdout)                        |
| Serving — API    | `src/readmit_bench/api/` + `Dockerfile`        | FastAPI `/predict` + OpenAPI                    |
| Serving — UI     | `flask_app.py` + `templates/` + `static/` + `Dockerfile.flask` | Clinician dashboard            |
| CI               | `.github/workflows/ci.yml`          | Lint, type-check, tests on every push                      |
| Docs             | `README.md`, `MODEL_CARD.md`, `CONTRACT.md`, `reports/report.html` | What was built and why         |
