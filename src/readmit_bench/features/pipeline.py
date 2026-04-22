"""sklearn ColumnTransformer factory for readmit-bench.

The transformer is **not fitted here**. It is constructed and returned so the
caller (training loop) can fit it inside each CV fold — this prevents target
encoding and median imputation from leaking the validation distribution.

Column groupings (post `add_derived_features`):
    NUMERIC          — continuous, impute median + StandardScaler
    BINARY           — bool flags, cast to int (no scaling needed)
    CAT_LOWCARD      — small cardinality, OneHot (handle_unknown=ignore)
    CAT_HIGHCARD     — large cardinality, sklearn TargetEncoder
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

from .derive import CHRONIC_COLS

NUMERIC_COLS: tuple[str, ...] = (
    "los_days",
    "age_at_admit",
    "num_diagnoses",
    "num_procedures",
    "prior_6mo_inpatient_count",
    "days_since_last_discharge_imputed",
    "chronic_count",
    "admit_month",
    "admit_dow",
)

BINARY_COLS: tuple[str, ...] = (
    "is_weekend_admit",
    "has_prior_admit",
    *CHRONIC_COLS,
)

CAT_LOWCARD_COLS: tuple[str, ...] = (
    "sex",
    "race",
    "age_bin",
    "admit_dx_chapter",
    "state_code",
)

CAT_HIGHCARD_COLS: tuple[str, ...] = (
    "admit_dx_code",
    "drg_code",
)

ID_COLS: tuple[str, ...] = (
    "beneficiary_id",
    "claim_id",
    "sample",
    "admit_date",
    "discharge_date",
    "admit_year",
)

LABEL_COL = "y"


@dataclass(frozen=True)
class FeatureSpec:
    numeric: tuple[str, ...] = NUMERIC_COLS
    binary: tuple[str, ...] = BINARY_COLS
    cat_lowcard: tuple[str, ...] = CAT_LOWCARD_COLS
    cat_highcard: tuple[str, ...] = CAT_HIGHCARD_COLS
    id_cols: tuple[str, ...] = ID_COLS
    label: str = LABEL_COL

    def all_features(self) -> list[str]:
        return [
            *self.numeric,
            *self.binary,
            *self.cat_lowcard,
            *self.cat_highcard,
        ]


def _bool_to_int(x):
    # FunctionTransformer needs a top-level callable for pickling
    return x.astype("int8")


def build_preprocessor(
    spec: FeatureSpec | None = None,
    *,
    target_encoder_smooth: str | float = "auto",
    target_encoder_cv: int = 5,
    random_state: int = 42,
) -> ColumnTransformer:
    """Return an *unfitted* ColumnTransformer.

    Fit it inside each CV fold so leak-free target encoding stays intact.
    """
    spec = spec or FeatureSpec()

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    binary_pipe = Pipeline(
        steps=[
            (
                "to_int",
                FunctionTransformer(_bool_to_int, validate=False, feature_names_out="one-to-one"),
            ),
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )

    cat_low_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype="float32",
                    min_frequency=20,
                ),
            ),
        ]
    )

    cat_high_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "target_enc",
                TargetEncoder(
                    smooth=target_encoder_smooth,
                    cv=target_encoder_cv,
                    target_type="binary",
                    random_state=random_state,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(spec.numeric)),
            ("bin", binary_pipe, list(spec.binary)),
            ("cat_low", cat_low_pipe, list(spec.cat_lowcard)),
            ("cat_high", cat_high_pipe, list(spec.cat_highcard)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
