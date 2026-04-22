"""Phase-6 tuning visualisation: Plots 17–19.

Reads:
    reports/tuned_summary.csv             -- per-model best params + val metrics
    reports/tuned_curves.npz              -- PR/ROC/cal arrays
    reports/tuning/<name>_trials.csv      -- per-trial history
    reports/tuning/<name>_study.pkl       -- full Optuna study (for param importances)
    reports/baselines.csv                 -- previous (untuned) leaderboard

Writes (PNG 300 dpi + PDF) to ``reports/figures/``:
    17_tuning_history       -- best-so-far PR-AUC vs trial, per model
    18_tuning_param_importance -- Optuna param-importance bars per model
    19_tuned_vs_baseline    -- side-by-side bars + delta annotations on val PR-AUC
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from readmit_bench.eda.plots import _add_caption, _suptitle
from readmit_bench.viz import apply_style, palette, save_fig

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY = Path("reports/tuned_summary.csv")
DEFAULT_BASELINES = Path("reports/baselines.csv")
DEFAULT_TUNING_DIR = Path("reports/tuning")
DEFAULT_FIG_DIR = Path("reports/figures")


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _model_colors(names: list[str]) -> dict[str, str]:
    pal = palette("qual")
    while len(pal) < len(names):
        pal = pal + pal
    return {n: pal[i] for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# 17. Trial history
# ---------------------------------------------------------------------------


def plot_trial_history(summary: pd.DataFrame, tuning_dir: Path) -> plt.Figure:
    n = len(summary)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6), squeeze=False, sharey=True)
    axes = axes[0]
    colors = _model_colors(summary["display_name"].tolist())

    for ax, (_, row) in zip(axes, summary.iterrows(), strict=False):
        name = row["name"]
        trials = pd.read_csv(tuning_dir / f"{name}_trials.csv")
        trials = trials.sort_values("number").reset_index(drop=True)
        completed = trials[trials["state"] == "COMPLETE"].copy()
        completed["best_so_far"] = completed["value"].cummax()

        c = colors[row["display_name"]]
        ax.scatter(
            completed["number"],
            completed["value"],
            color=c,
            alpha=0.45,
            s=36,
            edgecolor="white",
            linewidth=0.5,
            label="trial value",
            zorder=2,
        )
        ax.plot(
            completed["number"],
            completed["best_so_far"],
            color=c,
            linewidth=2.2,
            label="best so far",
            zorder=3,
        )

        best_idx = completed["value"].idxmax()
        bx, by = completed.loc[best_idx, "number"], completed.loc[best_idx, "value"]
        ax.scatter(
            [bx],
            [by],
            s=140,
            marker="*",
            color=c,
            edgecolor="#111827",
            linewidth=1.0,
            zorder=4,
            label=f"best (#{int(bx)})",
        )
        ax.annotate(
            f"{by:.4f}",
            xy=(bx, by),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color="#111827",
            fontweight="semibold",
        )

        ax.set_title(
            f"{row['display_name']}\nbest cv-PR-AUC = {row['best_cv_pr_auc']:.4f}",
            fontsize=11,
            color="#111827",
            loc="left",
        )
        ax.set_xlabel("Trial #", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("CV mean PR-AUC", fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        _despine(ax)
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.98, wspace=0.18)
    _suptitle(
        fig,
        "Optuna search history — TPE sampler with median pruner",
        "3-fold StratifiedGroupKFold on a 300K-row patient subsample. "
        "Star marks the best trial; the line shows best-so-far across the search.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# 18. Parameter importance
# ---------------------------------------------------------------------------


def plot_param_importance(summary: pd.DataFrame, tuning_dir: Path) -> plt.Figure:
    n = len(summary)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), squeeze=False)
    axes = axes[0]
    colors = _model_colors(summary["display_name"].tolist())

    for ax, (_, row) in zip(axes, summary.iterrows(), strict=False):
        name = row["name"]
        with open(tuning_dir / f"{name}_study.pkl", "rb") as f:
            study: optuna.Study = pickle.load(f)
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] param-importance failed: %s", name, exc)
            ax.text(
                0.5,
                0.5,
                "param-importance unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6B7280",
            )
            _despine(ax)
            continue

        imp = pd.Series(importances).sort_values(ascending=True)
        c = colors[row["display_name"]]
        bars = ax.barh(imp.index, imp.values, color=c, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, imp.values, strict=False):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  {v:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#111827",
            )

        ax.set_title(row["display_name"], fontsize=11, color="#111827", loc="left", pad=4)
        ax.set_xlim(0, max(imp.values) * 1.20 if len(imp) else 1.0)
        ax.set_xlabel("Importance (fANOVA, normalised)", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.tick_params(axis="x", labelsize=8.5, colors="#6B7280")
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        _despine(ax)

    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.13, right=0.98, wspace=0.42)
    _suptitle(
        fig,
        "Hyperparameter importance — which knobs actually moved PR-AUC?",
        "fANOVA decomposition of the Optuna search; values normalised so they sum to 1 per model.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# 19. Tuned vs baseline
# ---------------------------------------------------------------------------


def plot_tuned_vs_baseline(summary: pd.DataFrame, baselines: pd.DataFrame) -> plt.Figure:
    merged = (
        summary.merge(
            baselines[["name", "pr_auc", "roc_auc", "brier", "recall_at_top10"]].rename(
                columns={
                    "pr_auc": "pr_auc_base",
                    "roc_auc": "roc_auc_base",
                    "brier": "brier_base",
                    "recall_at_top10": "recall_at_top10_base",
                }
            ),
            on="name",
            how="left",
        )
        .sort_values("pr_auc", ascending=True)
        .reset_index(drop=True)
    )

    panels = [
        ("pr_auc", "pr_auc_base", "PR-AUC", False),
        ("roc_auc", "roc_auc_base", "ROC-AUC", False),
        ("brier", "brier_base", "Brier (↓ better)", True),
        ("recall_at_top10", "recall_at_top10_base", "Recall @ top-10%", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6))
    axes = axes.flatten()

    color_base = "#9CA3AF"
    color_tuned = "#2E5BFF"

    for ax, (tcol, bcol, label, lower_better) in zip(axes, panels, strict=False):
        y = np.arange(len(merged))
        h = 0.38
        ax.barh(
            y - h / 2,
            merged[bcol],
            height=h,
            color=color_base,
            edgecolor="white",
            linewidth=0.6,
            label="Baseline (untuned)",
        )
        ax.barh(
            y + h / 2,
            merged[tcol],
            height=h,
            color=color_tuned,
            edgecolor="white",
            linewidth=0.6,
            label="Tuned (Optuna)",
        )

        for i, (b, t) in enumerate(zip(merged[bcol], merged[tcol], strict=False)):
            delta = t - b
            sign = "+" if delta >= 0 else ""
            improved = (delta > 0) ^ lower_better  # XOR: lower_better flips
            color = "#16A34A" if improved else "#DC2626"
            ax.text(
                max(b, t) * 1.01,
                i,
                f"  {sign}{delta:+.4f}".replace("++", "+"),
                va="center",
                ha="left",
                fontsize=9,
                color=color,
                fontweight="semibold",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(merged["display_name"], fontsize=10)
        ax.set_title(label, fontsize=11, color="#111827", loc="left", pad=4)
        ax.set_xlim(0, max(merged[tcol].max(), merged[bcol].max()) * 1.18)
        ax.tick_params(axis="x", labelsize=8.5, colors="#6B7280")
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        _despine(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        fontsize=9.5,
        frameon=False,
    )
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.16, right=0.97, hspace=0.50, wspace=0.55)
    _suptitle(
        fig,
        "Tuned vs untuned — what did Optuna actually buy us?",
        "Same train/val split, same preprocessor. Δ shown in green if tuning improved that metric, red if it hurt.",
    )
    _add_caption(fig)
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all(
    summary_csv: Path = DEFAULT_SUMMARY,
    baselines_csv: Path = DEFAULT_BASELINES,
    tuning_dir: Path = DEFAULT_TUNING_DIR,
    out_dir: Path = DEFAULT_FIG_DIR,
) -> list[Path]:
    apply_style()
    summary = pd.read_csv(summary_csv)
    baselines = pd.read_csv(baselines_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("17_tuning_history", plot_trial_history(summary, tuning_dir)),
        ("18_tuning_param_importance", plot_param_importance(summary, tuning_dir)),
        ("19_tuned_vs_baseline", plot_tuned_vs_baseline(summary, baselines)),
    ]
    written: list[Path] = []
    for name, fig in plots:
        path = save_fig(fig, out_dir / f"{name}.png")
        plt.close(fig)
        written.append(path)
        logger.info("saved %s", path)
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--baselines-csv", type=Path, default=DEFAULT_BASELINES)
    p.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    args = _parse_args()
    run_all(args.summary_csv, args.baselines_csv, args.tuning_dir, args.out_dir)


if __name__ == "__main__":
    main()
