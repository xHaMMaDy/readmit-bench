"""Feature engineering + grouped/stratified split for readmit-bench."""

from .derive import CHRONIC_COLS, add_derived_features
from .pipeline import (
    BINARY_COLS,
    CAT_HIGHCARD_COLS,
    CAT_LOWCARD_COLS,
    NUMERIC_COLS,
    FeatureSpec,
    build_preprocessor,
)
from .split import (
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    SplitReport,
    assign_splits,
    summarise_split,
    write_split_assignments,
)

__all__ = [
    "CHRONIC_COLS",
    "add_derived_features",
    "BINARY_COLS",
    "CAT_HIGHCARD_COLS",
    "CAT_LOWCARD_COLS",
    "NUMERIC_COLS",
    "FeatureSpec",
    "build_preprocessor",
    "DEFAULT_SEED",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_VAL_SIZE",
    "SplitReport",
    "assign_splits",
    "summarise_split",
    "write_split_assignments",
]
