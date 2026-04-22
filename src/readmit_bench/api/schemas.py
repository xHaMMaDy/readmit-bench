"""Pydantic schemas for the readmit-bench prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SAMPLE_ENCOUNTER: dict = {
    "los_days": 7,
    "age_at_admit": 83,
    "num_diagnoses": 9,
    "num_procedures": 4,
    "prior_6mo_inpatient_count": 0,
    "days_since_last_discharge_imputed": 9999,
    "chronic_count": 7,
    "admit_month": 5,
    "admit_dow": 6,
    "is_weekend_admit": True,
    "has_prior_admit": False,
    "chronic_alzheimer": False,
    "chronic_chf": True,
    "chronic_ckd": True,
    "chronic_cancer": False,
    "chronic_copd": True,
    "chronic_depression": True,
    "chronic_diabetes": True,
    "chronic_ihd": True,
    "chronic_osteoporosis": True,
    "chronic_ra_oa": False,
    "chronic_stroke": False,
    "sex": "Female",
    "race": "Hispanic",
    "age_bin": "75-84",
    "admit_dx_chapter": "Infectious",
    "state_code": 54,
    "admit_dx_code": "0389",
    "drg_code": "854",
}


class Encounter(BaseModel):
    """One inpatient encounter — features required to score 30-day readmission risk."""

    model_config = ConfigDict(json_schema_extra={"example": SAMPLE_ENCOUNTER})

    # ---- numeric ----
    los_days: float = Field(..., ge=0, description="Length of stay in days.")
    age_at_admit: float = Field(..., ge=0, le=120, description="Patient age on admission.")
    num_diagnoses: float = Field(
        ..., ge=0, description="Distinct diagnosis codes on the index claim."
    )
    num_procedures: float = Field(..., ge=0, description="Procedure codes on the index claim.")
    prior_6mo_inpatient_count: float = Field(
        ..., ge=0, description="Inpatient stays in the prior 180 days."
    )
    days_since_last_discharge_imputed: float = Field(
        ...,
        ge=0,
        description="Days since most recent prior discharge (9999 when no prior stay).",
    )
    chronic_count: float = Field(
        ..., ge=0, le=11, description="Sum of CMS chronic-condition flags."
    )
    admit_month: int = Field(..., ge=1, le=12, description="Calendar month of admission (1-12).")
    admit_dow: int = Field(
        ..., ge=1, le=7, description="ISO day of week of admission (1=Mon, 7=Sun)."
    )

    # ---- binary flags ----
    is_weekend_admit: bool
    has_prior_admit: bool
    chronic_alzheimer: bool
    chronic_chf: bool
    chronic_ckd: bool
    chronic_cancer: bool
    chronic_copd: bool
    chronic_depression: bool
    chronic_diabetes: bool
    chronic_ihd: bool
    chronic_osteoporosis: bool
    chronic_ra_oa: bool
    chronic_stroke: bool

    # ---- low-cardinality categoricals ----
    sex: str = Field(..., description='e.g. "Male", "Female", "Unknown".')
    race: str = Field(..., description='e.g. "White", "Black", "Hispanic", "Asian", "Other".')
    age_bin: str = Field(..., description='e.g. "<65", "65-74", "75-84", "85+".')
    admit_dx_chapter: str = Field(..., description="ICD-9 chapter of the admit diagnosis.")
    state_code: int = Field(..., description="CMS-encoded state of residence (integer code).")

    # ---- high-cardinality categoricals (target-encoded) ----
    admit_dx_code: str = Field(..., description="Raw ICD-9 admit diagnosis code.")
    drg_code: str = Field(..., description="DRG code for the index encounter.")


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Calibrated probability of 30-day readmission.")
    threshold: float = Field(..., description="Cost-optimal operating threshold from Phase 7.")
    decision: Literal["flag", "skip"] = Field(
        ...,
        description='"flag" → enrol in transition-of-care program; "skip" → no intervention.',
    )
    risk_band: Literal["low", "moderate", "high", "very_high"]


class BatchPredictRequest(BaseModel):
    encounters: list[Encounter] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    n: int
    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    name: str
    version: str
    winner_model: str
    calibrator: str
    threshold: float
    cost_fn_usd: float
    cost_fp_usd: float
    test_n: int
    test_n_pos: int
    feature_count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    calibrator_loaded: bool
