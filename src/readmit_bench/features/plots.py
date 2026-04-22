"""Phase-4 feature plots for readmit-bench.

Two plots:
  11. Split overview        -- encounters / beneficiaries / positive rate per split
  12. Feature correlation   -- Spearman heatmap on numeric features (train only)

Run:
    python -m readmit_bench.features.plots
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from readmit_bench.eda.plots import (
    COLOR_NEG,
    COLOR_POS,
    _add_caption,
    _format_int,
    _suptitle,
)
from readmit_bench.features.pipeline import NUMERIC_COLS
from readmit_bench.viz import apply_style, save_fig

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = Path("data/processed/features.parquet")
DEFAULT_OUT_DIR = Path("reports/figures")

SPLIT_ORDER = ("train", "val", "test")
SPLIT_COLORS = {
    "train": COLOR_NEG,
    "val": "#7B61FF",
    "test": COLOR_POS,
}


# ---------------------------------------------------------------------------
# Plot 11 — split overview
# ---------------------------------------------------------------------------


def plot_split_overview(df: pd.DataFrame) -> plt.Figure:
    """Encounters / beneficiaries / positive rate per split."""
    summary = (
        df.groupby("split")
        .agg(
            n_encounters=("y", "size"),
            n_benef=("beneficiary_id", "nunique"),
            pos_rate=("y", "mean"),
        )
        .reindex(list(SPLIT_ORDER))
    )
    global_rate = float(df["y"].mean())

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.06, right=0.985, wspace=0.30)

    bar_colors = [SPLIT_COLORS[s] for s in SPLIT_ORDER]

    # panel 1: encounters
    ax = axes[0]
    bars = ax.bar(
        SPLIT_ORDER, summary["n_encounters"].to_numpy(), color=bar_colors, edgecolor="white"
    )
    for bar, val in zip(bars, summary["n_encounters"].to_numpy(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f" {_format_int(int(val))}\n({val / summary['n_encounters'].sum() * 100:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2937",
        )
    ax.set_title("Encounters", fontsize=11, color="#374151", loc="left", pad=10)
    ax.set_ylim(0, summary["n_encounters"].max() * 1.18)
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x / 1000):,}K")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35)

    # panel 2: beneficiaries
    ax = axes[1]
    bars = ax.bar(SPLIT_ORDER, summary["n_benef"].to_numpy(), color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, summary["n_benef"].to_numpy(), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f" {_format_int(int(val))}\n({val / summary['n_benef'].sum() * 100:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2937",
        )
    ax.set_title(
        "Beneficiaries (no patient appears in two splits)",
        fontsize=11,
        color="#374151",
        loc="left",
        pad=10,
    )
    ax.set_ylim(0, summary["n_benef"].max() * 1.18)
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x / 1000):,}K")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35)

    # panel 3: positive rate
    ax = axes[2]
    rates = summary["pos_rate"].to_numpy() * 100
    bars = ax.bar(SPLIT_ORDER, rates, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, rates, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2937",
        )
    ax.axhline(global_rate * 100, color="#9CA3AF", linestyle="--", linewidth=1, zorder=0)
    ax.text(
        0.99,
        0.97,
        f"global {global_rate * 100:.2f}%",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#6B7280",
        style="italic",
    )
    ax.set_title(
        "30-day readmit rate (stratified)", fontsize=11, color="#374151", loc="left", pad=10
    )
    ax.set_ylim(0, max(rates.max(), global_rate * 100) * 1.25)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.1f}%")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35)

    _suptitle(
        fig,
        "Train / validation / test split — grouped by beneficiary, stratified on outcome",
        "70 / 15 / 15 by patient. Same patient never crosses splits → no in-patient leakage. "
        "Positive rate matches across splits → unbiased held-out evaluation.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot 12 — feature correlation heatmap (train only, Spearman)
# ---------------------------------------------------------------------------


def plot_feature_correlation(df: pd.DataFrame) -> plt.Figure:
    """Spearman correlation of numeric features on the training set."""
    numeric = list(NUMERIC_COLS)
    train = df.loc[df["split"] == "train", numeric + ["y"]].copy()
    sample = train.sample(n=min(200_000, len(train)), random_state=0)
    corr = sample.corr(method="spearman", numeric_only=True)

    short_names = {
        "los_days": "LOS (days)",
        "age_at_admit": "Age",
        "num_diagnoses": "# diagnoses",
        "num_procedures": "# procedures",
        "prior_6mo_inpatient_count": "Prior 6mo admits",
        "days_since_last_discharge_imputed": "Days since last d/c",
        "chronic_count": "Chronic count",
        "admit_month": "Admit month",
        "admit_dow": "Admit weekday",
        "y": "y (readmit)",
    }
    corr.index = [short_names.get(c, c) for c in corr.index]
    corr.columns = [short_names.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    fig.subplots_adjust(top=0.85, bottom=0.16, left=0.22, right=0.92)

    vmax = float(np.nanmax(np.abs(corr.values - np.eye(len(corr)))))
    vmax = max(vmax, 0.05)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right", fontsize=9.5)
    ax.set_yticklabels(corr.index, fontsize=9.5)

    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "#1F2937"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=9.5, color="#4B5563")
    cbar.ax.tick_params(labelsize=8.5)

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)

    _suptitle(
        fig,
        "Numeric feature correlations (Spearman, train only, n=200k sample)",
        "Bottom row = correlation with the outcome y. Days-since-last-discharge and prior 6-month "
        "admits show the strongest direct signal.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


_PLOTS = (
    ("11_split_overview", plot_split_overview),
    ("12_feature_correlation", plot_feature_correlation),
)


def run_all(features_path: Path = DEFAULT_FEATURES, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    apply_style()
    if not features_path.exists():
        raise FileNotFoundError(
            f"features parquet not found: {features_path}. "
            "Run `python -m readmit_bench.features.build` first."
        )
    logger.info("loading features from %s", features_path)
    pdf = pl.read_parquet(features_path).to_pandas()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in _PLOTS:
        logger.info("→ %s", name)
        fig = fn(pdf)
        save_fig(fig, out_dir / f"{name}.png")
        save_fig(fig, out_dir / f"{name}.pdf")
        plt.close(fig)
    logger.info("wrote %d figures to %s", len(_PLOTS), out_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
        logging.getLogger(name).setLevel(logging.WARNING)
    run_all()


if __name__ == "__main__":
    main()
