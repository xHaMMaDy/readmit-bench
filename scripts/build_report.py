"""Generate reports/report.html — a self-contained blog-style project report.

Embeds figures as base64-encoded PNGs and renders all leaderboards/fairness
tables from the CSV artifacts under reports/.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
OUT = REPORTS / "report.html"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def img(name: str, alt: str, caption: str) -> str:
    src = f"data:image/png;base64,{b64(FIGS / name)}"
    return f"""
    <figure>
      <img src="{src}" alt="{alt}" loading="lazy" />
      <figcaption>{caption}</figcaption>
    </figure>
    """


def df_table(df: pd.DataFrame, classes: str = "data") -> str:
    return df.to_html(index=False, classes=classes, border=0, float_format=lambda x: f"{x:.4f}")


def main() -> None:
    baselines = pd.read_csv(REPORTS / "baselines.csv")
    tuned = pd.read_csv(REPORTS / "tuned_summary.csv")
    v2 = pd.read_csv(REPORTS / "v2_leaderboard.csv")
    ens = pd.read_csv(REPORTS / "ensembles_summary.csv")
    fair = pd.read_csv(REPORTS / "fairness_summary.csv")
    fair_gaps = json.loads((REPORTS / "fairness_gaps.json").read_text())
    drift = json.loads((REPORTS / "drift_summary.json").read_text())

    # ---------- Build leaderboard view ----------
    base_view = baselines[
        ["display_name", "pr_auc", "roc_auc", "brier", "recall_at_top10", "fit_time_s"]
    ].rename(
        columns={
            "display_name": "Model",
            "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC",
            "brier": "Brier",
            "recall_at_top10": "Recall@10%",
            "fit_time_s": "Fit (s)",
        }
    )

    tuned_view = tuned[
        [
            "display_name",
            "best_cv_pr_auc",
            "pr_auc",
            "roc_auc",
            "brier",
            "tuning_seconds",
        ]
    ].rename(
        columns={
            "display_name": "Model",
            "best_cv_pr_auc": "CV PR-AUC",
            "pr_auc": "Holdout PR-AUC",
            "roc_auc": "ROC-AUC",
            "brier": "Brier",
            "tuning_seconds": "Tuning (s)",
        }
    )

    v2_view = v2[
        ["display_name", "pr_auc", "roc_auc", "brier", "fit_time_s", "predict_time_s"]
    ].rename(
        columns={
            "display_name": "Model",
            "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC",
            "brier": "Brier",
            "fit_time_s": "Fit (s)",
            "predict_time_s": "Predict (s)",
        }
    )

    ens_view = ens[["name", "pr_auc", "roc_auc", "brier", "recall_at_top10"]].rename(
        columns={
            "name": "Pipeline",
            "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC",
            "brier": "Brier",
            "recall_at_top10": "Recall@10%",
        }
    )

    fair_view = fair[fair["attribute"] == "race"][
        ["slice", "n", "prevalence", "pr_auc", "fnr_at_t", "fpr_at_t"]
    ].rename(
        columns={
            "slice": "Group",
            "n": "n",
            "prevalence": "Prevalence",
            "pr_auc": "PR-AUC",
            "fnr_at_t": "FNR@t",
            "fpr_at_t": "FPR@t",
        }
    )

    # ---------- Drift summary ----------
    drift_metrics = drift["metrics"]
    drifted = next(m for m in drift_metrics if "DriftedColumnsCount" in m["metric_name"])
    drifted_share = drifted["value"]["share"] * 100
    drifted_count = int(drifted["value"]["count"])
    drift_rows = []
    for m in drift_metrics:
        name = m["metric_name"]
        if not name.startswith("ValueDrift"):
            continue
        col = m["config"].get("column", "")
        method = m["config"].get("method", "")
        threshold = m["config"].get("threshold", 0.1)
        val = m["value"]
        drift_rows.append(
            {
                "Column": col,
                "Method": method.replace(" distance", "").replace(" (normed)", ""),
                "Distance": val,
                "Threshold": threshold,
                "Drift?": "⚠️ yes" if val > threshold else "ok",
            }
        )
    drift_df = pd.DataFrame(drift_rows).sort_values("Distance", ascending=False).head(15)

    # ---------- Figures (group → list[(file, caption)]) ----------
    cohort_figs = [
        ("01_label_balance.png", "30-day readmission base rate (~9.6%)."),
        ("02_age_distribution.png", "Age distribution at admission — Medicare population skew."),
        ("05_los_distribution.png", "Length of stay distribution (right-skewed)."),
        ("08_chronic_conditions.png", "Chronic-condition prevalence in the cohort."),
        ("11_split_overview.png", "Beneficiary-disjoint train/holdout split."),
    ]
    baseline_figs = [
        ("13_baselines_leaderboard.png", "V1 baseline leaderboard sorted by PR-AUC."),
        ("14_baselines_pr_curves.png", "PR curves — boosted trees dominate."),
        ("15_baselines_roc_curves.png", "ROC curves — much less informative on imbalanced data."),
        (
            "16_baselines_calibration.png",
            "Reliability diagram — XGBoost/CatBoost over-confident; calibration warranted.",
        ),
    ]
    tuning_figs = [
        ("17_tuning_history.png", "Optuna search history for the top 3 models."),
        ("18_tuning_param_importance.png", "Hyperparameter importance per model."),
        ("19_tuned_vs_baseline.png", "Tuned vs baseline — small gains, plateau in sight."),
        ("20_reliability_before_after.png", "Calibration before vs after isotonic correction."),
        ("21_cost_surface.png", "Cost surface — operating threshold chosen at the elbow."),
        ("22_confusion_at_threshold.png", "Confusion matrix at deployed threshold."),
    ]
    explain_figs = [
        ("23_shap_global_importance.png", "Global SHAP feature importance."),
        ("24_shap_beeswarm.png", "SHAP beeswarm — directionality + magnitude per feature."),
        ("25_shap_dependence_top4.png", "SHAP dependence for the top 4 drivers."),
    ]
    fairness_figs = [
        ("26_fairness_slice_pr_auc.png", "Per-group PR-AUC across sex / race / age."),
        ("27_fairness_fnr_by_group.png", "False-negative rate gaps by group."),
        ("28_fairness_reliability_by_group.png", "Calibration parity across protected attributes."),
    ]
    v2_figs = [
        (
            "29_v2_nn_automl_leaderboard.png",
            "V2 — neural + AutoML models vs the tuned XGBoost benchmark.",
        ),
        ("30_ensembles_vs_base.png", "V2 — Stacking / Voting ensembles vs single learners."),
    ]

    def fig_block(items):
        return "\n".join(img(n, n, c) for n, c in items)

    # ---------- Compose HTML ----------
    css = """
    :root{
      --bg:#f7f8fb; --ink:#0f1626; --ink-2:#3a4660; --muted:#6b7790;
      --line:#e4e8ef; --card:#ffffff; --brand:#1f6feb; --teal:#14b8a6;
      --green:#16a34a; --amber:#d97706; --red:#dc2626;
    }
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{
      margin:0; background:var(--bg); color:var(--ink);
      font-family:'Plus Jakarta Sans','Inter',system-ui,sans-serif;
      line-height:1.65; font-feature-settings:"ss01","cv11";
    }
    .wrap{max-width:1080px; margin:0 auto; padding:0 28px;}
    header.masthead{
      padding:64px 0 48px; border-bottom:1px solid var(--line);
      background:linear-gradient(180deg, #ffffff 0%, var(--bg) 100%);
    }
    .eyebrow{
      letter-spacing:.18em; text-transform:uppercase; font-size:12px;
      font-weight:700; color:var(--brand);
    }
    h1{
      font-size:clamp(40px, 6vw, 64px); line-height:1.05;
      margin:14px 0 12px; letter-spacing:-.02em; font-weight:800;
    }
    h1 em{font-style:normal; background:linear-gradient(90deg,var(--brand),var(--teal)); -webkit-background-clip:text; background-clip:text; color:transparent;}
    .deck{font-size:19px; color:var(--ink-2); max-width:760px;}
    .meta{
      display:flex; gap:18px; flex-wrap:wrap; margin-top:24px;
      font-size:13px; color:var(--muted);
    }
    .meta b{color:var(--ink); font-weight:600}
    nav.toc{
      position:sticky; top:0; background:rgba(255,255,255,.92);
      backdrop-filter:blur(8px); border-bottom:1px solid var(--line);
      padding:12px 0; z-index:5; font-size:13px;
    }
    nav.toc ul{display:flex; gap:18px; flex-wrap:wrap; padding:0; margin:0; list-style:none}
    nav.toc a{color:var(--ink-2); text-decoration:none; padding:4px 0}
    nav.toc a:hover{color:var(--brand)}
    section{padding:56px 0; border-bottom:1px solid var(--line);}
    section h2{
      font-size:32px; letter-spacing:-.01em; margin:0 0 8px;
    }
    section .sec-eyebrow{
      letter-spacing:.18em; text-transform:uppercase; font-size:11px;
      font-weight:700; color:var(--teal); margin-bottom:6px;
    }
    section .lede{font-size:17px; color:var(--ink-2); max-width:780px; margin:0 0 28px}
    p{margin:0 0 16px}
    figure{margin:32px 0; padding:0}
    figure img{
      width:100%; height:auto; border-radius:12px;
      box-shadow:0 1px 2px rgba(15,22,38,.04), 0 8px 28px rgba(15,22,38,.06);
      border:1px solid var(--line); background:white;
    }
    figcaption{
      font-size:13px; color:var(--muted); margin-top:10px;
      font-style:italic; text-align:center;
    }
    .grid-2{display:grid; grid-template-columns:1fr 1fr; gap:20px}
    @media (max-width:780px){ .grid-2{grid-template-columns:1fr} }
    table.data{
      border-collapse:collapse; width:100%; font-size:14px; margin:16px 0 24px;
      background:var(--card); border-radius:10px; overflow:hidden;
      border:1px solid var(--line);
    }
    table.data thead th{
      background:#f1f4f9; color:var(--ink); text-align:left;
      padding:10px 14px; font-weight:600; font-size:12px;
      letter-spacing:.04em; text-transform:uppercase;
      border-bottom:1px solid var(--line);
    }
    table.data tbody td{
      padding:10px 14px; border-bottom:1px solid var(--line);
      font-variant-numeric:tabular-nums;
      font-family:'JetBrains Mono','SFMono-Regular',Consolas,monospace;
      font-size:13px;
    }
    table.data tbody tr:last-child td{border-bottom:0}
    table.data tbody tr:hover{background:#f7f9fc}
    .callout{
      border-left:3px solid var(--brand); background:#eef4ff;
      padding:14px 18px; border-radius:6px; color:#1e3a8a;
      margin:20px 0; font-size:15px;
    }
    .callout.warn{border-color:var(--amber); background:#fef6e7; color:#92400e}
    .callout.ok{border-color:var(--green); background:#ecfdf5; color:#14532d}
    .stat-row{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:24px 0;}
    .stat{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px;}
    .stat .v{font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:700; color:var(--ink);}
    .stat .k{font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-top:4px;}
    code{background:#eef0f4; padding:2px 6px; border-radius:4px; font-size:.92em}
    pre{
      background:#0f1626; color:#e6edf7; padding:16px 20px;
      border-radius:10px; overflow:auto; font-size:13px; line-height:1.55;
    }
    footer{padding:48px 0; color:var(--muted); font-size:13px; text-align:center;}
    .pill{display:inline-block; padding:3px 10px; border-radius:999px; background:#eef4ff; color:var(--brand); font-size:12px; font-weight:600;}
    .pill.warn{background:#fef6e7; color:#92400e}
    .pill.green{background:#ecfdf5; color:#14532d}
    """

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<title>readmit-bench — Project Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>{css}</style>
</head><body>

<header class="masthead">
  <div class="wrap">
    <div class="eyebrow">Senior ML portfolio · Healthcare</div>
    <h1>30-day hospital <em>readmission</em><br/>risk on Medicare claims.</h1>
    <p class="deck">An end-to-end machine learning system on CMS DE-SynPUF. From a 1.33M-row claims warehouse to a calibrated, fairness-audited, monitored model — served by a clinician-facing dashboard.</p>
    <div class="meta">
      <span><b>Cohort</b> 992,985 encounters · 9.6% positive</span>
      <span><b>Models</b> 12+ benchmarked</span>
      <span><b>Winner</b> Tuned XGBoost (PR-AUC 0.172)</span>
      <span><b>Serving</b> Flask + FastAPI in Docker</span>
    </div>
  </div>
</header>

<nav class="toc"><div class="wrap"><ul>
  <li><a href="#problem">Problem</a></li>
  <li><a href="#data">Data &amp; cohort</a></li>
  <li><a href="#baselines">Baselines</a></li>
  <li><a href="#tuning">Tuning &amp; calibration</a></li>
  <li><a href="#explain">Explainability</a></li>
  <li><a href="#fairness">Fairness</a></li>
  <li><a href="#v2">V2 — NN, AutoML, ensembles</a></li>
  <li><a href="#drift">Drift</a></li>
  <li><a href="#deploy">Deployment</a></li>
  <li><a href="#takeaways">Takeaways</a></li>
</ul></div></nav>

<main>

<section id="problem"><div class="wrap">
  <div class="sec-eyebrow">Problem</div>
  <h2>What we predict, and why it's hard.</h2>
  <p class="lede">Unplanned hospital readmissions within 30 days are a federally-tracked quality metric and a major cost driver. The signal is weak (~9.6% base rate), the data is messy longitudinal claims, and the action — flag a patient for follow-up — has real downstream cost.</p>

  <div class="stat-row">
    <div class="stat"><div class="v">992,985</div><div class="k">encounters in cohort</div></div>
    <div class="stat"><div class="v">9.6%</div><div class="k">30-day readmission rate</div></div>
    <div class="stat"><div class="v">31</div><div class="k">model features</div></div>
    <div class="stat"><div class="v">12+</div><div class="k">models benchmarked</div></div>
  </div>

  <p>The deployed model is a <b>tuned XGBoost</b> with isotonic calibration, operated at a cost-tuned threshold. The companion dashboard puts it in front of a clinician.</p>
</div></section>

<section id="data"><div class="wrap">
  <div class="sec-eyebrow">Data &amp; cohort</div>
  <h2>From raw CMS claims to a leakage-safe encounter table.</h2>
  <p class="lede">Source: CMS Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF, 20 samples). Inpatient + beneficiary tables, processed with Polars into a single encounter-level table with engineered prior-history, chronic-burden, and admit-context features.</p>

  {fig_block(cohort_figs)}

  <div class="callout">Splits are <b>beneficiary-disjoint</b> — no patient appears in both train and holdout. The label is constructed by looking forward 30 days from each discharge using only that patient's own future claims.</div>
</div></section>

<section id="baselines"><div class="wrap">
  <div class="sec-eyebrow">V1 Baselines</div>
  <h2>Six models on identical pre-processing.</h2>
  <p class="lede">Same ColumnTransformer, same train/holdout, single seed. Boosted trees lead by a wide margin; logistic regression is a strong floor.</p>

  {df_table(base_view)}
  {fig_block(baseline_figs)}
</div></section>

<section id="tuning"><div class="wrap">
  <div class="sec-eyebrow">Tuning &amp; calibration</div>
  <h2>Optuna search and a cost-aware threshold.</h2>
  <p class="lede">Top-3 models tuned with Optuna's TPE over a CV-PR-AUC objective; predictions then isotonically calibrated and the operating threshold chosen on a 1:5 (FN:FP) cost surface.</p>

  {df_table(tuned_view)}
  {fig_block(tuning_figs)}

  <div class="callout warn">Honest result: tuning lifted CV PR-AUC by ~1 point — gains plateau quickly on this cohort. The bigger wins came from <b>calibration + threshold selection</b>, not from heroic hyperparameter search.</div>
</div></section>

<section id="explain"><div class="wrap">
  <div class="sec-eyebrow">Explainability</div>
  <h2>Why the model does what it does.</h2>
  <p class="lede">Global SHAP, beeswarm, dependence on top-4 drivers, plus 6 LIME local explanations (3 high-risk, 3 low-risk) under <code>reports/lime/</code>.</p>
  {fig_block(explain_figs)}
</div></section>

<section id="fairness"><div class="wrap">
  <div class="sec-eyebrow">Fairness audit</div>
  <h2>Per-group PR-AUC, FNR, and calibration parity.</h2>
  <p class="lede">Sliced by sex, race, and age band. Demographic-parity and equalized-odds gaps reported in <code>reports/fairness_gaps.json</code>.</p>

  <div class="stat-row">
    <div class="stat"><div class="v">{fair_gaps['race']['demographic_parity_difference']:.3f}</div><div class="k">DP gap (race)</div></div>
    <div class="stat"><div class="v">{fair_gaps['race']['equalized_odds_difference']:.3f}</div><div class="k">EO gap (race)</div></div>
    <div class="stat"><div class="v">{fair_gaps['age_bin']['demographic_parity_difference']:.3f}</div><div class="k">DP gap (age)</div></div>
    <div class="stat"><div class="v">{fair_gaps['sex']['demographic_parity_difference']:.4f}</div><div class="k">DP gap (sex)</div></div>
  </div>

  <h3>Race slice — holdout PR-AUC and error rates</h3>
  {df_table(fair_view)}

  {fig_block(fairness_figs)}

  <div class="callout">The largest disparities are by <b>race</b> (DP gap ≈ 0.045) — small in absolute terms but documented in the model card with mitigation experiments.</div>
</div></section>

<section id="v2"><div class="wrap">
  <div class="sec-eyebrow">V2 extensions</div>
  <h2>Neural networks, AutoML, and ensembles — did they win?</h2>
  <p class="lede">PyTorch MLP, FT-Transformer, TabNet, and FLAML AutoML benchmarked under the same protocol. Then Stacking + soft-Voting ensembles over the top boosted trees.</p>

  <h3>NN + AutoML leaderboard</h3>
  {df_table(v2_view)}
  {fig_block([v2_figs[0]])}

  <h3>Ensembles vs single learners</h3>
  {df_table(ens_view)}
  {fig_block([v2_figs[1]])}

  <div class="callout warn">Honest negative result: deep tabular models <b>did not beat</b> tuned XGBoost on this cohort. Soft-voting and stacking added ~0.0001 PR-AUC over the best base learner — within noise. The deployed model remains tuned XGBoost.</div>
</div></section>

<section id="drift"><div class="wrap">
  <div class="sec-eyebrow">Monitoring</div>
  <h2>Evidently drift report — train vs holdout.</h2>
  <p class="lede">{drifted_count} of 30 columns flagged as drifted ({drifted_share:.1f}% share). Top-15 features by drift distance shown below. Full interactive report at <code>reports/drift_report.html</code>.</p>

  {df_table(drift_df.assign(**{
      'Distance': drift_df['Distance'].map(lambda x: f"{x:.4f}"),
      'Threshold': drift_df['Threshold'].map(lambda x: f"{x:.2f}")
  }))}

  <div class="callout ok">The two flagged columns (<code>admit_dx_code</code>, <code>drg_code</code>) are very high-cardinality categoricals — drift in their distribution is expected and absorbed by the model's frequency encoding.</div>
</div></section>

<section id="deploy"><div class="wrap">
  <div class="sec-eyebrow">Deployment</div>
  <h2>Two surfaces — REST API and a clinician dashboard.</h2>
  <p>The model is served two ways from the same artifact:</p>
  <ul>
    <li><b>FastAPI</b> at <code>/predict</code> — JSON in, JSON out, OpenAPI docs at <code>/docs</code>. Containerized via <code>Dockerfile</code>.</li>
    <li><b>Flask dashboard</b> at <code>/</code> — full UI: form-based scoring, leaderboards, fairness tables, drift report iframe, method writeup. Containerized via <code>Dockerfile.flask</code>.</li>
  </ul>

  <pre>curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"age_at_admit":73,"los_days":7,"chronic_chf":1,...}}'</pre>

  <p>Both run on HuggingFace Spaces; CI on every push (ruff, black, mypy, pytest with 45 tests).</p>
</div></section>

<section id="takeaways"><div class="wrap">
  <div class="sec-eyebrow">Takeaways</div>
  <h2>What this project actually shows.</h2>
  <ul>
    <li>Built and shipped an end-to-end ML system on real-world-shaped healthcare data.</li>
    <li>Held to honest evaluation: beneficiary-disjoint splits, PR-AUC primary, no leakage.</li>
    <li>Reported negative results (V2 NN/AutoML/ensembles did not beat XGBoost) instead of cherry-picking.</li>
    <li>Calibrated + cost-tuned + fairness-audited the deployed model, with a documented model card.</li>
    <li>Two production surfaces (REST + dashboard), 45 unit tests, CI pipeline, drift monitoring.</li>
  </ul>
</div></section>

</main>

<footer class="wrap">
  Generated by <code>scripts/build_report.py</code> ·
  See <a href="../README.md">README</a> · <a href="../MODEL_CARD.md">Model Card</a> · <a href="../CONTRACT.md">Contract</a>
</footer>

</body></html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
