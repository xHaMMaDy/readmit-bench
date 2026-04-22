"""Flask dashboard for readmit-bench (Phase 16).

Single-file Flask app meant to ship on Render / Hugging Face Spaces /
PythonAnywhere alongside the FastAPI service. Reuses the production
``Predictor`` class so the Web UI shows the exact same probability,
threshold and decision the API returns.

Run locally:
    python flask_app.py
    # → http://127.0.0.1:5050
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from readmit_bench.api.predictor import Predictor  # noqa: E402

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False

# --------------------------------------------------------------------------- #
# Predictor singleton — loaded once at startup, same instance per process.    #
# --------------------------------------------------------------------------- #
_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor()
    return _PREDICTOR


# --------------------------------------------------------------------------- #
# Form schema — every column the production pipeline expects.                 #
# Mirrors readmit_bench.features.pipeline.{NUMERIC_COLS, BINARY_COLS, ...}    #
# --------------------------------------------------------------------------- #
CHRONIC_FLAGS = (
    "chronic_alzheimer",
    "chronic_chf",
    "chronic_ckd",
    "chronic_cancer",
    "chronic_copd",
    "chronic_depression",
    "chronic_diabetes",
    "chronic_ihd",
    "chronic_osteoporosis",
    "chronic_ra_oa",
    "chronic_stroke",
)

NUMERIC_FIELDS: tuple[tuple[str, str, float, float, float], ...] = (
    # (name, label, min, max, default)
    ("age_at_admit", "Age at admit", 18, 100, 72),
    ("los_days", "Length of stay (days)", 1, 60, 5),
    ("num_diagnoses", "# diagnoses on claim", 1, 25, 6),
    ("num_procedures", "# procedures on claim", 0, 25, 1),
    ("prior_6mo_inpatient_count", "Prior 6-mo inpatient encounters", 0, 10, 1),
    ("days_since_last_discharge_imputed", "Days since last discharge", 0, 365, 90),
    ("admit_month", "Admit month", 1, 12, 6),
    ("admit_dow", "Admit day-of-week (0=Mon)", 0, 6, 2),
)

LOWCARD_FIELDS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("sex", "Sex", ("F", "M"), "M"),
    ("race", "Race", ("White", "Black", "Hispanic", "Asian", "Other"), "White"),
    (
        "admit_dx_chapter",
        "Admit-dx ICD chapter",
        (
            "Circulatory",
            "Respiratory",
            "Digestive",
            "Neoplasms",
            "Endocrine",
            "Genitourinary",
            "Injury",
            "Other",
        ),
        "Circulatory",
    ),
)

TEXT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("state_code", "State code", "36"),
    ("admit_dx_code", "Admit dx code (ICD-9)", "4280"),
    ("drg_code", "DRG code", "291"),
)


def _age_bin(age: float) -> str:
    if age < 55:
        return "<55"
    if age < 65:
        return "55-64"
    if age < 75:
        return "65-74"
    if age < 85:
        return "75-84"
    return "85+"


def _build_encounter(form: dict[str, Any]) -> dict[str, Any]:
    enc: dict[str, Any] = {}
    for name, _label, _lo, _hi, _default in NUMERIC_FIELDS:
        enc[name] = float(form.get(name, _default))
    enc["is_weekend_admit"] = form.get("is_weekend_admit") in ("on", "true", "1", True)
    enc["has_prior_admit"] = form.get("has_prior_admit") in ("on", "true", "1", True)
    chronic_count = 0
    for flag in CHRONIC_FLAGS:
        v = form.get(flag) in ("on", "true", "1", True)
        enc[flag] = v
        chronic_count += int(v)
    enc["chronic_count"] = chronic_count
    for name, _label, _options, _default in LOWCARD_FIELDS:
        enc[name] = str(form.get(name, _default))
    enc["age_bin"] = _age_bin(enc["age_at_admit"])
    for name, _label, _default in TEXT_FIELDS:
        enc[name] = str(form.get(name, _default))
    return enc


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _leaderboards() -> list[dict[str, Any]]:
    files = [
        ("V1 baselines", ROOT / "reports" / "baselines_summary.csv"),
        ("V1 tuned (Optuna)", ROOT / "reports" / "tuned_summary.csv"),
        ("V1 calibration", ROOT / "reports" / "calibration_summary.csv"),
        ("V2 NN + AutoML", ROOT / "reports" / "v2_leaderboard.csv"),
        ("V2 ensembles", ROOT / "reports" / "ensembles_summary.csv"),
    ]
    boards: list[dict[str, Any]] = []
    for label, path in files:
        df = _read_csv(path)
        if df is None:
            continue
        boards.append(
            {
                "label": label,
                "rel_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "table_html": df.to_html(
                    classes="leaderboard",
                    index=False,
                    border=0,
                    float_format=lambda v: f"{v:.4f}",
                ),
            }
        )
    return boards


def _fairness() -> dict[str, Any] | None:
    df = _read_csv(ROOT / "reports" / "fairness_summary.csv")
    if df is None:
        return None
    attrs = sorted(df["attribute"].unique().tolist()) if "attribute" in df.columns else []
    by_attr: dict[str, str] = {}
    for a in attrs:
        sub = df[df["attribute"] == a]
        by_attr[a] = sub.to_html(
            classes="leaderboard",
            index=False,
            border=0,
            float_format=lambda v: f"{v:.4f}",
        )
    return {"attributes": attrs, "tables": by_attr}


def _drift_available() -> bool:
    return (ROOT / "reports" / "drift_report.html").exists()


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #


@app.route("/", methods=["GET", "POST"])
def home():
    predictor = get_predictor()
    meta = predictor.meta
    result = None
    error = None
    encounter: dict[str, Any] | None = None

    if request.method == "POST":
        try:
            encounter = _build_encounter(request.form)
            result = predictor.predict([encounter])[0]
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

    return render_template(
        "index.html",
        meta=meta,
        numeric_fields=NUMERIC_FIELDS,
        lowcard_fields=LOWCARD_FIELDS,
        text_fields=TEXT_FIELDS,
        chronic_flags=CHRONIC_FLAGS,
        result=result,
        error=error,
        encounter=encounter,
        encounter_json=json.dumps(encounter, indent=2) if encounter else None,
        leaderboards=_leaderboards(),
        fairness=_fairness(),
        drift_available=_drift_available(),
    )


@app.route("/drift")
def drift():
    path = ROOT / "reports" / "drift_report.html"
    if not path.exists():
        return (
            "Drift report not generated yet. Run "
            "<code>python -m readmit_bench.drift.evidently_report</code>.",
            404,
        )
    return path.read_text(encoding="utf-8")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        encounter = _build_encounter(payload)
        result = get_predictor().predict([encounter])[0]
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify({"input": encounter, **result})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": get_predictor().meta.winner_model})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False)
