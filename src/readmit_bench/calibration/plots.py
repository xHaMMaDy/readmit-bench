"""Phase-7 calibration visualisation: Plots 20–22.

Reads:
    reports/calibration_curves.npz   -- reliability + cost-curve arrays
    reports/calibration_summary.csv  -- per-method Brier / PR-AUC table
    models/winner_threshold.json     -- chosen threshold + cost + confusion

Writes (PNG 300 dpi + PDF) to ``reports/figures/``:
    20_reliability_before_after  -- uncal vs chosen calibrator on TEST set
    21_cost_surface              -- total cost($) vs threshold, mark optimum
    22_confusion_at_threshold    -- 2x2 confusion + cost breakdown panel
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from readmit_bench.eda.plots import SOURCE_TXT, _add_caption, _suptitle
from readmit_bench.viz import apply_style, palette, save_fig

logger = logging.getLogger(__name__)

DEFAULT_CURVES = Path("reports/calibration_curves.npz")
DEFAULT_SUMMARY = Path("reports/calibration_summary.csv")
DEFAULT_THRESHOLD = Path("models/winner_threshold.json")
DEFAULT_FIG_DIR = Path("reports/figures")

PHASE7_SOURCE = SOURCE_TXT


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


def _fmt_money(x: float, _pos=None) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


# ---------- Plot 20 ----------
def plot_reliability_before_after(
    curves, summary: pd.DataFrame, winner: str, fig_dir: Path
) -> Path:
    apply_style()
    pal = _palette_map()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    for ax, label, mp_key, fp_key, c_key, color in (
        (
            axes[0],
            "Uncalibrated",
            "rel_uncal_mean_pred",
            "rel_uncal_frac_pos",
            "rel_uncal_counts",
            pal["muted"],
        ),
        (
            axes[1],
            (
                "After calibration (identity — no change)"
                if winner.lower() == "uncalibrated"
                else f"After {winner.capitalize()}"
            ),
            "rel_cal_mean_pred",
            "rel_cal_frac_pos",
            "rel_cal_counts",
            pal["primary"],
        ),
    ):
        mp = curves[mp_key]
        fp = curves[fp_key]
        cnt = curves[c_key]
        ax.plot([0, 1], [0, 1], color=pal["grid"], lw=1.2, ls="--", label="perfect")
        sizes = 20 + 240 * (cnt / cnt.max())
        ax.scatter(
            mp, fp, s=sizes, color=color, alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3
        )
        ax.plot(mp, fp, color=color, lw=1.6, alpha=0.9, zorder=2)
        # Brier annotation
        method = "uncalibrated" if "Uncal" in label else winner
        brier = float(summary.loc[summary["method"] == method, "brier"].iloc[0])
        ax.text(
            0.04,
            0.94,
            f"Brier = {brier:.5f}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="600",
            va="top",
            bbox=dict(facecolor="white", edgecolor=pal["grid"], boxstyle="round,pad=0.35"),
        )
        ax.set_xlim(0, max(0.6, float(mp.max()) * 1.1))
        ax.set_ylim(0, max(0.6, float(fp.max()) * 1.1))
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive rate")
        ax.set_title(label, fontsize=13, fontweight="600", pad=8)
        _despine(ax)
        ax.grid(True, color=pal["grid"], alpha=0.4, lw=0.6)

    _suptitle(
        fig,
        "Reliability: uncalibrated vs chosen calibrator (test split)",
        "Bubbles sized by bin count · diagonal = perfect calibration",
    )
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.06, right=0.98, wspace=0.20)
    _add_caption(fig, PHASE7_SOURCE)
    out = fig_dir / "20_reliability_before_after.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


# ---------- Plot 21 ----------
def plot_cost_surface(curves, threshold_payload: dict, fig_dir: Path) -> Path:
    apply_style()
    pal = _palette_map()
    grid = curves["threshold_grid"]
    costs = curves["cost_grid"]
    t_star = float(threshold_payload["threshold"])
    cost_star = float(threshold_payload["test_total_cost_usd"])
    cost_always = float(threshold_payload["test_cost_always_treat_usd"])
    cost_never = float(threshold_payload["test_cost_never_treat_usd"])

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.plot(grid, costs / 1e6, color=pal["primary"], lw=2.2, label="Total cost($) at threshold")
    ax.fill_between(grid, costs / 1e6, costs.max() / 1e6, color=pal["primary"], alpha=0.07)

    ax.axhline(
        cost_always / 1e6,
        color=pal["accent"],
        lw=1.4,
        ls="--",
        alpha=0.8,
        label=f"Always-treat baseline ({_fmt_money(cost_always)})",
    )
    ax.axhline(
        cost_never / 1e6,
        color=pal["muted"],
        lw=1.4,
        ls="--",
        alpha=0.8,
        label=f"Never-treat baseline ({_fmt_money(cost_never)})",
    )

    ax.axvline(t_star, color=pal["positive"], lw=1.6, ls=":", alpha=0.9)
    ax.scatter(
        [t_star],
        [cost_star / 1e6],
        s=180,
        color=pal["positive"],
        edgecolor="white",
        linewidth=1.6,
        zorder=5,
        label=f"Optimum: t={t_star:.4f} → {_fmt_money(cost_star)}",
    )

    ax.set_xlabel("Decision threshold (calibrated probability)")
    ax.set_ylabel("Total expected cost on test split (USD, millions)")
    ax.set_title("", pad=2)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    _despine(ax)
    ax.grid(True, color=pal["grid"], alpha=0.4, lw=0.6)
    ax.set_xlim(0, 1)

    cost_fn = float(threshold_payload["cost_fn"])
    cost_fp = float(threshold_payload["cost_fp"])
    saved = cost_always - cost_star
    _suptitle(
        fig,
        "Cost vs threshold: pick the operating point that minimises clinical USD",
        f"FN cost = ${cost_fn:,.0f}/miss · FP cost = ${cost_fp:,.0f}/intervention · "
        f"chosen point saves {_fmt_money(saved)} vs always-treat",
    )
    _add_caption(fig, PHASE7_SOURCE)
    out = fig_dir / "21_cost_surface.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


# ---------- Plot 22 ----------
def plot_confusion_at_threshold(threshold_payload: dict, fig_dir: Path) -> Path:
    apply_style()
    pal = _palette_map()
    cm = threshold_payload["confusion"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    n = tn + fp + fn + tp
    cost_fn_unit = float(threshold_payload["cost_fn"])
    cost_fp_unit = float(threshold_payload["cost_fp"])
    cost_total = float(threshold_payload["test_total_cost_usd"])
    t_star = float(threshold_payload["threshold"])
    cal = threshold_payload["calibrator"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.0, 1.05]})

    # ---- left: confusion heat-map ----
    ax = axes[0]
    mat = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_axisbelow(False)
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            pct = v / n * 100
            color = "white" if v > mat.max() / 2 else "#1a1a1a"
            ax.text(
                j,
                i,
                f"{v:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=13,
                fontweight="600",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"], fontsize=11)
    ax.set_title(f"Confusion @ t={t_star:.4f} ({cal})", fontsize=12, fontweight="600", pad=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    # ---- right: cost breakdown ----
    ax = axes[1]
    fn_cost = fn * cost_fn_unit
    fp_cost = fp * cost_fp_unit
    tp_cost = tp * cost_fp_unit  # interventions on true pos still cost the unit
    bars = [
        "FN — missed\nreadmits",
        "FP — unneeded\ninterventions",
        "TP — justified\ninterventions",
    ]
    vals = [fn_cost, fp_cost, tp_cost]
    colors = [pal["negative"], pal["accent"], pal["positive"]]
    y = np.arange(len(bars))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=1.5, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(bars, fontsize=10)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(
            v + max(vals) * 0.01,
            i,
            _fmt_money(v),
            va="center",
            fontsize=10,
            fontweight="600",
            color="#1a1a1a",
        )
    ax.set_xlabel("Cost on test split (USD)")
    ax.set_title(
        f"Cost decomposition · total = {_fmt_money(cost_total)}",
        fontsize=12,
        fontweight="600",
        pad=8,
    )
    _despine(ax)
    ax.grid(True, axis="x", color=pal["grid"], alpha=0.4, lw=0.6)
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_money))
    ax.set_xlim(0, max(vals) * 1.18)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    _suptitle(
        fig,
        "Operating point confusion + cost decomposition",
        f"Recall = {recall:.3f} · Precision = {precision:.3f} · "
        f"flagged = {(tp + fp) / n * 100:.1f}% of encounters",
    )
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.10, right=0.97, wspace=0.50)
    _add_caption(fig, PHASE7_SOURCE)
    out = fig_dir / "22_confusion_at_threshold.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


def run_all(
    curves_path: Path = DEFAULT_CURVES,
    summary_path: Path = DEFAULT_SUMMARY,
    threshold_path: Path = DEFAULT_THRESHOLD,
    fig_dir: Path = DEFAULT_FIG_DIR,
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    curves = np.load(curves_path)
    summary = pd.read_csv(summary_path)
    threshold_payload = json.loads(threshold_path.read_text())
    winner = threshold_payload["calibrator"]

    outputs = []
    outputs.append(plot_reliability_before_after(curves, summary, winner, fig_dir))
    outputs.append(plot_cost_surface(curves, threshold_payload, fig_dir))
    outputs.append(plot_confusion_at_threshold(threshold_payload, fig_dir))
    return outputs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    p = argparse.ArgumentParser(description="Render Phase-7 calibration plots 20-22.")
    p.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = p.parse_args()
    outs = run_all(args.curves, args.summary, args.threshold, args.fig_dir)
    for o in outs:
        logger.info("wrote %s", o)


if __name__ == "__main__":
    main()
