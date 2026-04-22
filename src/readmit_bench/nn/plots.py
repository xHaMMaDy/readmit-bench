"""Phase-13/14 V2 model plots (29–30).

29: V2 NN + AutoML leaderboard vs V1 GBM tuned winner
30: Stacking / Voting ensemble vs best base learner (zoomed PR-AUC)

Reads:
    reports/v2_leaderboard.csv
    reports/tuned_summary.csv
    reports/ensembles_summary.csv

Writes (PNG 300 dpi + PDF) to ``reports/figures/``:
    29_v2_nn_automl_leaderboard
    30_ensembles_vs_base
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from readmit_bench.eda.plots import _add_caption, _suptitle
from readmit_bench.viz import apply_style, palette, save_fig

logger = logging.getLogger(__name__)

DEFAULT_FIG_DIR = Path("reports/figures")


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_v2_leaderboard(v2_df: pd.DataFrame, v1_best: float, v1_label: str) -> plt.Figure:
    """Compare V2 NN + AutoML PR-AUC against the V1 tuned-GBM winner."""
    df = v2_df.sort_values("pr_auc", ascending=True).reset_index(drop=True)
    pal = palette("qual")
    colors = [pal[i % len(pal)] for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    bars = ax.barh(
        df["display_name"],
        df["pr_auc"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, value in zip(bars, df["pr_auc"], strict=False):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#111827",
        )

    ax.axvline(
        v1_best,
        color="#DC2626",
        linestyle="--",
        linewidth=1.4,
        label=f"V1 tuned winner — {v1_label} (PR-AUC = {v1_best:.4f})",
    )

    ax.set_xlabel("PR-AUC (validation)", fontsize=10.5)
    ax.set_xlim(0, max(df["pr_auc"].max(), v1_best) * 1.18)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9, colors="#6B7280")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    _despine(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.22, right=0.97)
    _suptitle(
        fig,
        "Phase 13 — Neural & AutoML benchmark",
        "Tabular MLP, TabNet, FT-Transformer and FLAML AutoML evaluated on val (198,597 rows). "
        "Honest finding: NNs do not beat tuned GBMs on this saturated cohort.",
    )
    _add_caption(fig)
    return fig


def plot_ensembles_vs_base(ens_df: pd.DataFrame) -> plt.Figure:
    """Zoomed bar chart: best base vs voting/stacking on the same calib/test split."""
    df = ens_df.copy()
    df["kind"] = df["name"].str.split("/").str[0]
    df["short"] = df["name"].str.split("/").str[1]
    pretty = {
        "voting_mean": "Voting (mean)",
        "stacking_lr": "Stacking (LR meta)",
        "xgboost": "XGBoost (tuned)",
        "catboost": "CatBoost (tuned)",
        "hist_gradient_boosting": "HistGradientBoosting (tuned)",
    }
    df["display_name"] = df["short"].map(pretty).fillna(df["short"])
    df = df.sort_values("pr_auc", ascending=True).reset_index(drop=True)
    pal = palette("qual")

    def _color(kind: str) -> str:
        if kind == "ensemble":
            return "#DC2626"
        return pal[0]

    colors = [_color(k) for k in df["kind"]]

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    bars = ax.barh(
        df["display_name"],
        df["pr_auc"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, value in zip(bars, df["pr_auc"], strict=False):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.5f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#111827",
        )

    base_mask = df["kind"] == "base"
    if base_mask.any():
        best_base = float(df.loc[base_mask, "pr_auc"].max())
        ax.axvline(
            best_base,
            color="#6B7280",
            linestyle="--",
            linewidth=1.2,
            label=f"Best base learner (PR-AUC = {best_base:.5f})",
        )
        ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    lo = float(df["pr_auc"].min())
    hi = float(df["pr_auc"].max())
    span = max(hi - lo, 1e-4)
    ax.set_xlim(lo - span * 0.6, hi + span * 1.2)
    ax.set_xlabel("PR-AUC (held-out test split)", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9, colors="#6B7280")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    _despine(ax)

    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.32, right=0.97)
    _suptitle(
        fig,
        "Phase 14 — Ensembles vs best base learner (zoomed)",
        "Voting (mean) and stacking (LR meta) over tuned XGBoost/CatBoost/HistGB. "
        "Honest finding: lift over best base is essentially zero — base learners are too correlated (ρ ≈ 0.99).",
    )
    _add_caption(fig)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-leaderboard", type=Path, default=Path("reports/v2_leaderboard.csv"))
    parser.add_argument("--tuned-summary", type=Path, default=Path("reports/tuned_summary.csv"))
    parser.add_argument(
        "--ensembles-summary", type=Path, default=Path("reports/ensembles_summary.csv")
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.v2_leaderboard.exists() and args.tuned_summary.exists():
        v2 = pd.read_csv(args.v2_leaderboard)
        tuned = pd.read_csv(args.tuned_summary)
        winner_row = tuned.sort_values("pr_auc", ascending=False).iloc[0]
        fig = plot_v2_leaderboard(
            v2,
            v1_best=float(winner_row["pr_auc"]),
            v1_label=str(winner_row.get("display_name", winner_row.get("name", "GBM"))),
        )
        save_fig(fig, args.out_dir / "29_v2_nn_automl_leaderboard")
        plt.close(fig)
        logger.info("wrote 29_v2_nn_automl_leaderboard")

    if args.ensembles_summary.exists():
        ens = pd.read_csv(args.ensembles_summary)
        fig = plot_ensembles_vs_base(ens)
        save_fig(fig, args.out_dir / "30_ensembles_vs_base")
        plt.close(fig)
        logger.info("wrote 30_ensembles_vs_base")


if __name__ == "__main__":
    main()
