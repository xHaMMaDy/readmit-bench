"""Feature engineering + grouped/stratified split for readmit-bench."""

from .pipeline import (
    BINARY_COLS,
    CAT_HIGHCARD_COLS,
    CAT_LOWCARD_COLS,
    NUMERIC_COLS,
    FeatureSpec,
    build_preprocessor,
)

# Lazy imports for polars-dependent modules (training-only)
def __getattr__(name: str):
    if name == "CHRONIC_COLS":
        from .derive import CHRONIC_COLS
        return CHRONIC_COLS
    elif name == "add_derived_features":
        from .derive import add_derived_features
        return add_derived_features
    elif name == "DEFAULT_SEED":
        from .split import DEFAULT_SEED
        return DEFAULT_SEED
    elif name == "DEFAULT_TEST_SIZE":
        from .split import DEFAULT_TEST_SIZE
        return DEFAULT_TEST_SIZE
    elif name == "DEFAULT_VAL_SIZE":
        from .split import DEFAULT_VAL_SIZE
        return DEFAULT_VAL_SIZE
    elif name == "SplitReport":
        from .split import SplitReport
        return SplitReport
    elif name == "assign_splits":
        from .split import assign_splits
        return assign_splits
    elif name == "summarise_split":
        from .split import summarise_split
        return summarise_split
    elif name == "write_split_assignments":
        from .split import write_split_assignments
        return write_split_assignments
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
