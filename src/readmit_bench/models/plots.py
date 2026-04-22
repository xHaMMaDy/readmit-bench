"""Phase-5 baseline-comparison plots (13–16).

Reads:
    reports/baselines.csv          -- leaderboard
    reports/baselines_curves.npz   -- PR/ROC/calibration arrays per model

Writes (PNG 300 dpi + PDF) to ``reports/figures/``:
    13_baselines_leaderboard
    14_baselines_pr_curves
    15_baselines_roc_curves
    16_baselines_calibration
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from readmit_bench.eda.plots import _add_caption, _suptitle
from readmit_bench.viz import apply_style, palette, save_fig

logger = logging.getLogger(__name__)

DEFAULT_LEADERBOARD = Path("reports/baselines.csv")
DEFAULT_CURVES = Path("reports/baselines_curves.npz")
DEFAULT_FIG_DIR = Path("reports/figures")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _model_colors(names: list[str]) -> dict[str, str]:
    pal = palette("qual")
    while len(pal) < len(names):
        pal = pal + pal
    return {name: pal[i] for i, name in enumerate(names)}


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# 13. Leaderboard
# ---------------------------------------------------------------------------


def plot_leaderboard(df: pd.DataFrame) -> plt.Figure:
    """Four-panel ranked bar chart: PR-AUC, ROC-AUC, Brier (lower=better), recall@top10%."""
    df = df.sort_values("pr_auc", ascending=True).reset_index(drop=True)
    colors = _model_colors(df["display_name"].tolist())

    panels = [
        ("pr_auc", "PR-AUC", "higher is better", False),
        ("roc_auc", "ROC-AUC", "higher is better", False),
        ("brier", "Brier score", "lower is better", True),
        ("recall_at_top10", "Recall @ top-10% risk", "higher is better", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    axes = axes.flatten()

    for ax, (col, label, hint, lower_better) in zip(axes, panels, strict=False):
        order = df.sort_values(col, ascending=not lower_better).reset_index(drop=True)
        bar_colors = [colors[n] for n in order["display_name"]]
        bars = ax.barh(
            order["display_name"], order[col], color=bar_colors,
            edgecolor="white", linewidth=0.6,
        )
        for bar, value in zip(bars, order[col], strict=False):
            ax.text(
                bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {value:.3f}", va="center", ha="left",
                fontsize=9, color="#111827",
            )
        ax.set_title(f"{label}  ·  {hint}", fontsize=11, color="#111827", loc="left", pad=6)
        ax.set_xlim(0, max(order[col]) * 1.18)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.tick_params(axis="x", labelsize=8.5, colors="#6B7280")
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        _despine(ax)

    fig.subplots_adjust(top=0.85, bottom=0.10, left=0.16, right=0.97, hspace=0.55, wspace=0.55)
    _suptitle(
        fig,
        "Baseline model leaderboard — validation set",
        "All models trained on the full train split (927,893 rows) and scored on val (198,597 rows). "
        "Sorted within each panel by that metric.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# 14. PR curves overlay
# ---------------------------------------------------------------------------


def plot_pr_curves(df: pd.DataFrame, curves: dict[str, np.ndarray], prevalence: float) -> plt.Figure:
    df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    colors = _model_colors(df["display_name"].tolist())

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for _, row in df.iterrows():
        name, dn = row["name"], row["display_name"]
        recall = curves[f"{name}__pr_recall"]
        precision = curves[f"{name}__pr_precision"]
        ax.plot(
            recall, precision,
            color=colors[dn], linewidth=2.0, alpha=0.92,
            label=f"{dn}  (PR-AUC = {row['pr_auc']:.3f})",
        )

    ax.axhline(prevalence, color="#9CA3AF", linestyle="--", linewidth=1.0,
               label=f"Random (prevalence = {prevalence:.3f})")
    ax.set_xlabel("Recall", fontsize=10.5)
    ax.set_ylabel("Precision", fontsize=10.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, df["pr_auc"].max() * 2.6))
    ax.grid(True, alpha=0.25, linewidth=0.6)
    _despine(ax)

    handles, labels = ax.get_legend_handles_labels()
    n_items = len(labels)
    ncol = 4 if n_items > 4 else n_items
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.90),
        ncol=ncol, frameon=False, fontsize=9.5,
        columnspacing=1.8, handlelength=2.0, handletextpad=0.6,
    )

    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.10, right=0.97)
    _suptitle(
        fig,
        "Precision–Recall curves — validation set",
        "PR is the right view for an imbalanced positive class (~9.6%). Higher curve = better separation at every operating point.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# 15. ROC curves overlay
# ---------------------------------------------------------------------------


def plot_roc_curves(df: pd.DataFrame, curves: dict[str, np.ndarray]) -> plt.Figure:
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    colors = _model_colors(df["display_name"].tolist())

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for _, row in df.iterrows():
        name, dn = row["name"], row["display_name"]
        fpr = curves[f"{name}__roc_fpr"]
        tpr = curves[f"{name}__roc_tpr"]
        ax.plot(
            fpr, tpr,
            color=colors[dn], linewidth=2.0, alpha=0.92,
            label=f"{dn}  (ROC-AUC = {row['roc_auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.0, label="Random")
    ax.set_xlabel("False positive rate", fontsize=10.5)
    ax.set_ylabel("True positive rate", fontsize=10.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    _despine(ax)

    handles, labels = ax.get_legend_handles_labels()
    n_items = len(labels)
    ncol = 4 if n_items > 4 else n_items
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.90),
        ncol=ncol, frameon=False, fontsize=9.5,
        columnspacing=1.8, handlelength=2.0, handletextpad=0.6,
    )

    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.10, right=0.97)
    _suptitle(
        fig,
        "ROC curves — validation set",
        "Closer to the top-left corner = better. Diagonal line is a random classifier.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# 16. Calibration / reliability
# ---------------------------------------------------------------------------


def plot_calibration(df: pd.DataFrame, curves: dict[str, np.ndarray]) -> plt.Figure:
    df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    n = len(df)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 3.4 * nrows + 1.2), squeeze=False)
    colors = _model_colors(df["display_name"].tolist())

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i // ncols][i % ncols]
        name, dn = row["name"], row["display_name"]
        pred = curves[f"{name}__rel_pred"]
        pos = curves[f"{name}__rel_pos"]
        count = curves[f"{name}__rel_count"]

        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=0.9)
        sizes = 12 + 90 * (count / max(count.max(), 1))
        ax.scatter(pred, pos, s=sizes, color=colors[dn], alpha=0.85,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.plot(pred, pos, color=colors[dn], linewidth=1.3, alpha=0.7)
        upper = float(max(pred.max(), pos.max(), 0.05)) * 1.15
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_title(f"{dn}\nBrier = {row['brier']:.4f}", fontsize=10, color="#111827", loc="left", pad=8)
        ax.set_xlabel("Predicted probability", fontsize=9)
        ax.set_ylabel("Observed rate", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        _despine(ax)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.subplots_adjust(top=0.85, bottom=0.08, left=0.06, right=0.98, hspace=0.55, wspace=0.40)
    _suptitle(
        fig,
        "Calibration / reliability — validation set",
        "Diagonal = perfectly calibrated. Marker size = bin count. Models above the diagonal under-predict, below over-predict.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all(
    leaderboard_csv: Path = DEFAULT_LEADERBOARD,
    curves_npz: Path = DEFAULT_CURVES,
    out_dir: Path = DEFAULT_FIG_DIR,
    prevalence: float = 0.0964,
) -> list[Path]:
    apply_style()
    df = pd.read_csv(leaderboard_csv)
    curves = dict(np.load(curves_npz))
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    plots = [
        ("13_baselines_leaderboard", plot_leaderboard(df)),
        ("14_baselines_pr_curves", plot_pr_curves(df, curves, prevalence=prevalence)),
        ("15_baselines_roc_curves", plot_roc_curves(df, curves)),
        ("16_baselines_calibration", plot_calibration(df, curves)),
    ]
    for name, fig in plots:
        path = save_fig(fig, out_dir / f"{name}.png")
        plt.close(fig)
        results.append(path)
        logger.info("saved %s", path)
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    p.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--prevalence", type=float, default=0.0964)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    args = _parse_args()
    run_all(args.leaderboard, args.curves, args.out_dir, args.prevalence)


if __name__ == "__main__":
    main()
