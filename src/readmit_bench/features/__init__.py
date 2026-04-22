"""Feature engineering + grouped/stratified split for readmit-bench."""

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

# Lazy imports for polars-dependent modules (training-only)
def __getattr__(name: str):
    if name == "CHRONIC_COLS":
        from .derive import CHRONIC_COLS
        return CHRONIC_COLS
    elif name == "add_derived_features":
        from .derive import add_derived_features
        return add_derived_features
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
