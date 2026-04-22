"""Phase-8 SHAP visualisation: Plots 23–25.

Reads:
    reports/shap_values.npz         -- shap_values, X_processed, feature_names, base_value, y_sample
    reports/shap_top_features.csv   -- mean|SHAP| ranking

Writes (PNG 300 dpi + PDF) to ``reports/figures/``:
    23_shap_global_importance   -- top-20 mean(|SHAP|) bar chart
    24_shap_beeswarm            -- top-15 beeswarm with feature-value coloring
    25_shap_dependence_top4     -- 2x2 dependence scatters for top-4 features
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from readmit_bench.eda.plots import SOURCE_TXT, _add_caption, _suptitle
from readmit_bench.viz import apply_style, palette, save_fig

logger = logging.getLogger(__name__)

DEFAULT_NPZ = Path("reports/shap_values.npz")
DEFAULT_TOP = Path("reports/shap_top_features.csv")
DEFAULT_FIG_DIR = Path("reports/figures")
PHASE8_SOURCE = SOURCE_TXT

# Diverging blue→red colormap (low feature value → blue, high → red) — SHAP convention.
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "readmit_shap", ["#2E5BFF", "#A6BFFF", "#F4F4F4", "#FFB39A", "#E63946"]
)


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


def _pretty(name: str) -> str:
    """Strip ColumnTransformer prefixes (`num__`, `bin__`, `cat_low__`, `cat_high__`)."""
    for pre in ("num__", "bin__", "cat_low__", "cat_high__"):
        if name.startswith(pre):
            return name[len(pre) :]
    return name


# ---------- Plot 23 ----------
def plot_global_importance(top_df: pd.DataFrame, fig_dir: Path, n_top: int = 20) -> Path:
    apply_style()
    pal = _palette_map()
    df = top_df.head(n_top).iloc[::-1].reset_index(drop=True)
    colors = [pal["positive"] if s >= 0 else pal["negative"] for s in df["mean_signed_shap"]]

    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    y = np.arange(len(df))
    ax.barh(y, df["mean_abs_shap"], color=colors, edgecolor="white", linewidth=1.0, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty(f) for f in df["feature"]], fontsize=10)
    for i, (m, s) in enumerate(zip(df["mean_abs_shap"], df["mean_signed_shap"], strict=False)):
        ax.text(
            m + df["mean_abs_shap"].max() * 0.01,
            i,
            f"{m:.3f}  (signed {s:+.3f})",
            va="center",
            fontsize=9,
            color="#1a1a1a",
        )
    ax.set_xlabel("mean(|SHAP value|)  —  average impact on log-odds of readmission")
    ax.set_xlim(0, df["mean_abs_shap"].max() * 1.30)
    _despine(ax)
    ax.grid(True, axis="x", color=pal["grid"], alpha=0.4, lw=0.6)

    # Color-key legend
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=pal["positive"],
            edgecolor="white",
            label="Mean signed SHAP ≥ 0  (raises risk)",
        ),
        Patch(
            facecolor=pal["negative"],
            edgecolor="white",
            label="Mean signed SHAP < 0  (lowers risk)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95)

    _suptitle(
        fig,
        f"Global feature importance (top-{n_top} by mean |SHAP|)",
        "Bar length = average magnitude of impact · color = average direction (positive ↔ negative)",
    )
    _add_caption(fig, PHASE8_SOURCE)
    out = fig_dir / "23_shap_global_importance.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


# ---------- Plot 24 ----------
def plot_beeswarm(
    shap_vals: np.ndarray,
    X_proc: np.ndarray,
    feature_names: list[str],
    top_df: pd.DataFrame,
    fig_dir: Path,
    n_top: int = 15,
    max_points_per_feature: int = 1500,
) -> Path:
    apply_style()
    pal = _palette_map()
    rng = np.random.default_rng(0)
    top_features = top_df.head(n_top)["feature"].tolist()[::-1]  # least important on top
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    for i, fname in enumerate(top_features):
        col = name_to_idx[fname]
        sv = shap_vals[:, col]
        fv = X_proc[:, col]
        if len(sv) > max_points_per_feature:
            keep = rng.choice(len(sv), max_points_per_feature, replace=False)
            sv = sv[keep]
            fv = fv[keep]
        # Normalise feature values to 5–95 percentile for stable color mapping.
        lo, hi = np.percentile(fv, [5, 95])
        if hi <= lo:
            hi = lo + 1e-6
        norm = Normalize(vmin=lo, vmax=hi, clip=True)
        # Y jitter
        y_jit = i + (rng.random(len(sv)) - 0.5) * 0.65
        ax.scatter(sv, y_jit, s=14, c=fv, cmap=SHAP_CMAP, norm=norm, alpha=0.65, edgecolors="none")

    ax.axvline(0, color=pal["grid"], lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels([_pretty(f) for f in top_features], fontsize=10)
    ax.set_xlabel("SHAP value (impact on model output, log-odds)")
    _despine(ax)
    ax.grid(True, axis="x", color=pal["grid"], alpha=0.4, lw=0.6)

    # Single colorbar covering the conceptual low→high feature-value gradient.
    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0.05, 0.95])
    cbar.set_ticklabels(["low", "high"])
    cbar.set_label("Feature value", rotation=270, labelpad=14)

    _suptitle(
        fig,
        f"SHAP beeswarm (top-{n_top} features)",
        "Each dot = one encounter · x = SHAP value · color = feature value (blue low → red high)",
    )
    _add_caption(fig, PHASE8_SOURCE)
    out = fig_dir / "24_shap_beeswarm.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


# ---------- Plot 25 ----------
def plot_dependence_top4(
    shap_vals: np.ndarray,
    X_proc: np.ndarray,
    feature_names: list[str],
    top_df: pd.DataFrame,
    fig_dir: Path,
) -> Path:
    apply_style()
    pal = _palette_map()
    top4 = top_df.head(4)["feature"].tolist()
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    for ax, fname in zip(axes.ravel(), top4, strict=False):
        col = name_to_idx[fname]
        x = X_proc[:, col]
        y = shap_vals[:, col]
        # If discrete-looking (≤10 unique vals), jitter x for readability.
        uniq = np.unique(x)
        x_plot = (
            x + (np.random.default_rng(col).random(len(x)) - 0.5) * 0.15 if len(uniq) <= 10 else x
        )
        ax.scatter(x_plot, y, s=10, color=pal["primary"], alpha=0.35, edgecolors="none")
        ax.axhline(0, color=pal["grid"], lw=1.0, ls="--", alpha=0.7)

        # Smoothed trend via percentile bins (only meaningful for continuous-ish features).
        if len(uniq) > 10:
            try:
                bins = np.quantile(x, np.linspace(0, 1, 21))
                bins = np.unique(bins)
                mids, means = [], []
                for lo, hi in zip(bins[:-1], bins[1:], strict=False):
                    m = (x >= lo) & (x <= hi)
                    if m.sum() >= 20:
                        mids.append((lo + hi) / 2)
                        means.append(y[m].mean())
                if mids:
                    ax.plot(mids, means, color=pal["accent"], lw=2.2, label="binned mean SHAP")
                    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
            except Exception:
                pass

        ax.set_title(_pretty(fname), fontsize=11, fontweight="600", pad=6)
        ax.set_xlabel("Feature value (after preprocessing)")
        ax.set_ylabel("SHAP value")
        _despine(ax)
        ax.grid(True, color=pal["grid"], alpha=0.4, lw=0.6)

    _suptitle(
        fig,
        "Dependence plots for top-4 features",
        "x = feature value · y = SHAP impact · orange line = binned mean for continuous features",
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.98, hspace=0.42, wspace=0.25)
    _add_caption(fig, PHASE8_SOURCE)
    out = fig_dir / "25_shap_dependence_top4.png"
    save_fig(fig, out)
    plt.close(fig)
    return out


def run_all(
    npz_path: Path = DEFAULT_NPZ, top_path: Path = DEFAULT_TOP, fig_dir: Path = DEFAULT_FIG_DIR
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=False)
    shap_vals = data["shap_values"]
    X_proc = data["X_processed"]
    feature_names = list(data["feature_names"])
    top_df = pd.read_csv(top_path)
    outputs = []
    outputs.append(plot_global_importance(top_df, fig_dir))
    outputs.append(plot_beeswarm(shap_vals, X_proc, feature_names, top_df, fig_dir))
    outputs.append(plot_dependence_top4(shap_vals, X_proc, feature_names, top_df, fig_dir))
    return outputs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    p = argparse.ArgumentParser(description="Render Phase-8 SHAP plots 23-25.")
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--top", type=Path, default=DEFAULT_TOP)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = p.parse_args()
    for o in run_all(args.npz, args.top, args.fig_dir):
        logger.info("wrote %s", o)


if __name__ == "__main__":
    main()
