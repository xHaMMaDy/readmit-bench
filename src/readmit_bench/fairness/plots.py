"""Phase-9 fairness plots: 26 per-slice PR-AUC, 27 FNR by group at t*, 28 reliability by group.

Reads:
    reports/fairness_summary.csv
    reports/fairness_predictions.parquet  (y, p, yhat, sensitive cols)
    reports/fairness_gaps.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from readmit_bench.eda.plots import SOURCE_TXT, _add_caption, _suptitle
from readmit_bench.fairness.audit import SENSITIVE_ATTRS
from readmit_bench.viz import apply_style, palette, save_fig

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY = Path("reports/fairness_summary.csv")
DEFAULT_PRED = Path("reports/fairness_predictions.parquet")
DEFAULT_GAPS = Path("reports/fairness_gaps.json")
DEFAULT_FIG_DIR = Path("reports/figures")
PHASE9_SOURCE = SOURCE_TXT

ATTR_LABELS = {"sex": "Sex", "race": "Race / ethnicity", "age_bin": "Age band"}


def _palette_map():
    p = palette()
    return {
        "primary": p[0],
        "accent": p[1],
        "positive": p[2],
        "negative": p[3],
        "muted": p[7],
        "grid": "#D7DCE3",
    }


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _slice_order(table: pd.DataFrame, attr: str) -> list[str]:
    """Stable, semantic ordering of slices for an attribute."""
    if attr == "age_bin":
        order = ["<65", "65-74", "75-84", "85+"]
    elif attr == "sex":
        order = ["Female", "Male"]
    else:
        order = (
            table[(table["attribute"] == attr) & (table["slice"] != "OVERALL")]
            .sort_values("n", ascending=False)["slice"]
            .tolist()
        )
    return order


# ---------- Plot 26: per-slice PR-AUC ----------


def plot_slice_pr_auc(table: pd.DataFrame, fig_dir: Path) -> Path:
    pal = _palette_map()
    fig, axes = plt.subplots(1, len(SENSITIVE_ATTRS), figsize=(13, 4.6), sharey=True)

    for ax, attr in zip(axes, SENSITIVE_ATTRS, strict=False):
        sub = table[table["attribute"] == attr].copy()
        overall = sub.loc[sub["slice"] == "OVERALL", "pr_auc"].iloc[0]
        slices = _slice_order(table, attr)
        vals = sub.set_index("slice").loc[slices, "pr_auc"].to_numpy()
        n = sub.set_index("slice").loc[slices, "n"].to_numpy()

        x = np.arange(len(slices))
        bars = ax.bar(x, vals, color=pal["primary"], width=0.62, edgecolor="white", linewidth=1.0)
        ax.axhline(overall, color=pal["accent"], lw=1.6, ls="--", label=f"Overall = {overall:.3f}")
        for b, v, ni in zip(bars, vals, n, strict=False):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.004,
                f"{v:.3f}\n(n={ni:,})",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#33405A",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(slices, rotation=0)
        ax.set_title(ATTR_LABELS[attr], fontsize=11, weight="semibold")
        ax.set_ylim(0, max(vals.max(), overall) * 1.18)
        ax.grid(axis="y", color=pal["grid"], lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
        _despine(ax)

    axes[0].set_ylabel("PR-AUC")
    _suptitle(
        fig,
        "Per-slice PR-AUC vs. overall",
        "Ranking quality holds across slices; gaps are within sampling noise on smaller groups.",
    )
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.07, right=0.98, wspace=0.18)
    _add_caption(fig, PHASE9_SOURCE)
    return save_fig(fig, fig_dir / "26_fairness_slice_pr_auc")


# ---------- Plot 27: FNR by group at t* ----------


def plot_fnr_by_group(table: pd.DataFrame, gaps: dict, fig_dir: Path) -> Path:
    pal = _palette_map()
    fig, axes = plt.subplots(1, len(SENSITIVE_ATTRS), figsize=(13, 4.8), sharey=True)

    for ax, attr in zip(axes, SENSITIVE_ATTRS, strict=False):
        sub = table[table["attribute"] == attr].copy()
        overall_fnr = sub.loc[sub["slice"] == "OVERALL", "fnr_at_t"].iloc[0]
        slices = _slice_order(table, attr)
        vals = sub.set_index("slice").loc[slices, "fnr_at_t"].to_numpy()
        n_pos = sub.set_index("slice").loc[slices, "n_pos"].to_numpy()
        worst = gaps[attr]["worst_fnr_slice"]

        colors = [pal["negative"] if s == worst else pal["primary"] for s in slices]
        x = np.arange(len(slices))
        bars = ax.bar(x, vals, color=colors, width=0.62, edgecolor="white", linewidth=1.0)
        ax.axhline(
            overall_fnr,
            color=pal["accent"],
            lw=1.6,
            ls="--",
            label=f"Overall FNR = {overall_fnr:.3f}",
        )
        for b, v, np_i in zip(bars, vals, n_pos, strict=False):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.02,
                f"{v:.3f}\n({int(np_i):,} pos)",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#33405A",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(slices, rotation=0)
        ax.set_title(
            f"{ATTR_LABELS[attr]}  ·  gap = {gaps[attr]['fnr_gap']:.3f}",
            fontsize=11,
            weight="semibold",
        )
        ax.set_ylim(0, max(vals.max() * 1.30, overall_fnr * 1.30, 0.01))
        ax.grid(axis="y", color=pal["grid"], lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        _despine(ax)

    axes[0].set_ylabel("False-negative rate @ t* = 0.0320")
    _suptitle(
        fig,
        "False-negative rate by group at deployment threshold",
        "Worst-slice (red) vs. overall (dashed). Misses are the costly clinical outcome.",
    )
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.07, right=0.98, wspace=0.18)
    _add_caption(fig, PHASE9_SOURCE)
    return save_fig(fig, fig_dir / "27_fairness_fnr_by_group")


# ---------- Plot 28: reliability by group ----------


def plot_reliability_by_group(pred: pd.DataFrame, fig_dir: Path, n_bins: int = 10) -> Path:
    pal = _palette_map()
    fig, axes = plt.subplots(1, len(SENSITIVE_ATTRS), figsize=(13, 4.8), sharey=True, sharex=True)

    cmap = plt.get_cmap("viridis")

    for ax, attr in zip(axes, SENSITIVE_ATTRS, strict=False):
        slices = (
            ["<65", "65-74", "75-84", "85+"]
            if attr == "age_bin"
            else ["Female", "Male"] if attr == "sex" else pred[attr].value_counts().index.tolist()
        )
        for i, val in enumerate(slices):
            mask = pred[attr].to_numpy() == val
            if mask.sum() < 200:
                continue
            y_g, p_g = pred.loc[mask, "y"].to_numpy(), pred.loc[mask, "p"].to_numpy()
            if y_g.sum() == 0 or y_g.sum() == len(y_g):
                continue
            frac_pos, mean_pred = calibration_curve(y_g, p_g, n_bins=n_bins, strategy="quantile")
            color = cmap(0.15 + 0.7 * i / max(1, len(slices) - 1))
            ax.plot(
                mean_pred,
                frac_pos,
                "o-",
                color=color,
                lw=1.6,
                ms=5,
                label=f"{val} (n={int(mask.sum()):,})",
            )
        lo, hi = 0, max(pred["p"].quantile(0.995), 0.4)
        ax.plot([lo, hi], [lo, hi], color=pal["muted"], lw=1.0, ls=":", label="perfect")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_title(ATTR_LABELS[attr], fontsize=11, weight="semibold")
        ax.set_xlabel("Predicted probability")
        ax.grid(color=pal["grid"], lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=8.0)
        _despine(ax)

    axes[0].set_ylabel("Empirical fraction positive")
    _suptitle(
        fig,
        "Reliability curves by group",
        "If a curve sits below the diagonal the model overestimates risk for that group; above → underestimates.",
    )
    fig.subplots_adjust(top=0.83, bottom=0.16, left=0.07, right=0.98, wspace=0.18)
    _add_caption(fig, PHASE9_SOURCE)
    return save_fig(fig, fig_dir / "28_fairness_reliability_by_group")


# ---------- runner ----------


def run(
    summary: Path = DEFAULT_SUMMARY,
    pred: Path = DEFAULT_PRED,
    gaps: Path = DEFAULT_GAPS,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> list[Path]:
    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(summary)
    pred_df = pd.read_parquet(pred)
    gaps_d = json.loads(gaps.read_text())

    out = [
        plot_slice_pr_auc(table, fig_dir),
        plot_fnr_by_group(table, gaps_d, fig_dir),
        plot_reliability_by_group(pred_df, fig_dir),
    ]
    for p in out:
        logger.info("wrote %s", p)
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Phase-9 fairness plots 26–28")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--gaps", type=Path, default=DEFAULT_GAPS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(args.summary, args.pred, args.gaps, args.fig_dir)


if __name__ == "__main__":
    _cli()
