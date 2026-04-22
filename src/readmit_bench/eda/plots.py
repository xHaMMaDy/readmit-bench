"""Phase-3 EDA plots for the readmit-bench project.

Each ``plot_*`` function takes a pandas DataFrame and returns a matplotlib
Figure. ``run_all`` loads the cohort parquet once and writes every plot to
``reports/figures/`` as PNG (300 dpi) + vector PDF.

Design rules (every figure must satisfy):
* Headline title (left-aligned) + descriptive subtitle.
* Color-blind-safe palette, semantic colors (positive class = warm orange,
  negative = cool blue, totals = slate).
* Direct value annotations where they aid the reader.
* Source caption in the bottom-left of every figure.
* No top/right spines, light y-grid only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from readmit_bench.viz import apply_style, palette, save_fig

logger = logging.getLogger(__name__)

# --- semantic colors ---------------------------------------------------------
COLOR_POS = "#FF7847"  # readmitted (positive class)
COLOR_NEG = "#2E5BFF"  # not readmitted (negative class)
COLOR_NEUTRAL = "#5F6B7A"  # totals / neutral
COLOR_RATE = "#E63946"  # readmit rate line
COLOR_VOLUME = "#2E5BFF"  # volume bar

SOURCE_TXT = (
    "Source: CMS DE-SynPUF synthetic Medicare claims (20 samples, 2007–2010) · readmit-bench"
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _add_caption(fig, text: str = SOURCE_TXT) -> None:
    fig.text(
        0.005,
        0.005,
        text,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#6B7280",
        style="italic",
    )


def _suptitle(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(
        0.01, 0.985, title, ha="left", va="top", fontsize=15, fontweight="semibold", color="#111827"
    )
    if subtitle:
        fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=10.5, color="#4B5563")
    # Reserve enough headroom so subplot titles don't collide with the (sub)title,
    # but never loosen tighter top margins set by the caller (e.g. plot 17 needs
    # top=0.80 for two-line subplot titles).
    target = 0.84 if subtitle else 0.91
    if fig.subplotpars.top > target:
        fig.subplots_adjust(top=target)


def _wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    denom = 1.0 + z * z / np.where(n > 0, n, 1)
    centre = (p + z * z / (2.0 * np.where(n > 0, n, 1))) / denom
    half = (
        z
        * np.sqrt(p * (1 - p) / np.where(n > 0, n, 1) + z * z / (4.0 * np.where(n > 0, n, 1) ** 2))
    ) / denom
    low = np.clip(centre - half, 0, 1)
    high = np.clip(centre + half, 0, 1)
    return low, high


def _format_int(n: int) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------------------
# 1. Label balance
# ---------------------------------------------------------------------------


def plot_label_balance(df: pd.DataFrame) -> plt.Figure:
    counts = df["y"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    pct = counts / counts.sum() * 100

    fig, (ax_bar, ax_donut) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # --- bar (left) ---
    bars = ax_bar.bar(
        ["Not readmitted", "Readmitted ≤ 30d"],
        counts.values,
        color=[COLOR_NEG, COLOR_POS],
        width=0.55,
        edgecolor="white",
    )
    ax_bar.set_ylabel("Encounters")
    ax_bar.set_title("Class counts", loc="left", fontsize=12, color="#374151")
    ax_bar.set_ylim(0, counts.max() * 1.18)
    for bar, n, p in zip(bars, counts.values, pct.values, strict=False):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.025,
            f"{_format_int(int(n))}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="medium",
            color="#1F2937",
        )
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # --- donut (right) ---
    wedges, _ = ax_donut.pie(
        counts.values,
        colors=[COLOR_NEG, COLOR_POS],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2),
    )
    ax_donut.set_aspect("equal")
    ax_donut.set_title("Class share", loc="center", fontsize=12, color="#374151", pad=8)
    ax_donut.text(
        0,
        0.06,
        f"{pct.iloc[1]:.1f}%",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="semibold",
        color=COLOR_POS,
    )
    ax_donut.text(
        0, -0.18, "positive class", ha="center", va="center", fontsize=10, color="#6B7280"
    )

    _suptitle(
        fig,
        "30-day readmission label balance",
        f"Imbalanced binary target — {_format_int(int(counts.sum()))} encounters total",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


# ---------------------------------------------------------------------------
# 2. Age distribution by label
# ---------------------------------------------------------------------------


def plot_age_distribution(df: pd.DataFrame) -> plt.Figure:
    age = df["age_at_admit"].dropna()
    age = age[(age >= 18) & (age <= 110)]
    pos = df.loc[df["y"] == 1, "age_at_admit"].dropna()
    neg = df.loc[df["y"] == 0, "age_at_admit"].dropna()

    bins = np.arange(20, 106, 2)

    fig, (ax_hist, ax_rate) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # left: overlapping density-normalised histograms
    ax_hist.hist(
        neg,
        bins=bins,
        density=True,
        color=COLOR_NEG,
        alpha=0.55,
        label="Not readmitted",
        edgecolor="white",
        linewidth=0.4,
    )
    ax_hist.hist(
        pos,
        bins=bins,
        density=True,
        color=COLOR_POS,
        alpha=0.65,
        label="Readmitted ≤ 30d",
        edgecolor="white",
        linewidth=0.4,
    )
    ax_hist.set_xlabel("Age at admission (years)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Age distribution by class", loc="left", fontsize=12, color="#374151", pad=8)
    ax_hist.set_xlim(20, 105)

    # right: readmit rate by age bin (with 95% Wilson CI)
    bins_r = np.arange(20, 106, 5)
    df_age = df[["age_at_admit", "y"]].dropna()
    df_age = df_age[(df_age["age_at_admit"] >= 20) & (df_age["age_at_admit"] <= 105)]
    df_age["bucket"] = pd.cut(df_age["age_at_admit"], bins=bins_r, right=False)
    grp = df_age.groupby("bucket", observed=True)["y"].agg(["sum", "count"]).reset_index()
    centres = np.array([(b.left + b.right) / 2 for b in grp["bucket"]])
    rate = grp["sum"] / grp["count"]
    low, high = _wilson_ci(grp["sum"].to_numpy(), grp["count"].to_numpy())

    ax_rate.fill_between(centres, low * 100, high * 100, color=COLOR_RATE, alpha=0.18, linewidth=0)
    ax_rate.plot(
        centres,
        rate * 100,
        color=COLOR_RATE,
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="Readmit rate (%)",
    )
    ax_rate.axhline(
        df["y"].mean() * 100,
        color=COLOR_NEUTRAL,
        linestyle="--",
        linewidth=1.0,
        label=f"Overall: {df['y'].mean()*100:.1f}%",
    )
    ax_rate.set_xlabel("Age at admission (years)")
    ax_rate.set_ylabel("30-day readmit rate (%)")
    ax_rate.set_title(
        "Readmission rate by age (95% CI)", loc="left", fontsize=12, color="#374151", pad=8
    )
    ax_rate.set_xlim(20, 105)

    # Consolidated figure-level legend (top-center, between subtitle and axes)
    h1, l1 = ax_hist.get_legend_handles_labels()
    h2, l2 = ax_rate.get_legend_handles_labels()
    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=False,
        fontsize=10,
        columnspacing=1.8,
        handlelength=2.0,
        handletextpad=0.6,
    )

    _suptitle(
        fig,
        "Age & 30-day readmission",
        "Older patients show only a mild rate gradient — age alone is a weak signal",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.82))
    return fig


# ---------------------------------------------------------------------------
# 3. Sex × readmit rate
# ---------------------------------------------------------------------------


def plot_sex_rate(df: pd.DataFrame) -> plt.Figure:
    grp = df.groupby("sex", observed=True)["y"].agg(["sum", "count"]).reset_index()
    grp = grp[grp["sex"].notna()]
    grp["rate"] = grp["sum"] / grp["count"]
    low, high = _wilson_ci(grp["sum"].to_numpy(), grp["count"].to_numpy())
    grp["err_low"] = grp["rate"] - low
    grp["err_high"] = high - grp["rate"]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = [COLOR_NEG if s == "Male" else "#A06CD5" for s in grp["sex"]]
    bars = ax.bar(
        grp["sex"],
        grp["rate"] * 100,
        yerr=[grp["err_low"] * 100, grp["err_high"] * 100],
        color=colors,
        width=0.5,
        capsize=4,
        edgecolor="white",
        error_kw=dict(lw=1.2, ecolor="#374151"),
    )
    overall = df["y"].mean() * 100
    ax.axhline(
        overall, color=COLOR_RATE, linestyle="--", linewidth=1.0, label=f"Overall: {overall:.2f}%"
    )
    ax.set_ylabel("30-day readmit rate (%)")
    ax.set_xlabel("")
    ax.legend(loc="upper right", frameon=False)
    for bar, n, r in zip(bars, grp["count"], grp["rate"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"{r*100:.2f}%\n(n={_format_int(int(n))})",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#1F2937",
        )
    ax.set_ylim(0, max(grp["rate"]) * 100 * 1.30)

    _suptitle(
        fig,
        "30-day readmission by sex",
        "Wilson 95% CI shown — small but measurable gap, useful for fairness audit",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


# ---------------------------------------------------------------------------
# 4. Race × readmit rate
# ---------------------------------------------------------------------------


def plot_race_rate(df: pd.DataFrame) -> plt.Figure:
    grp = df.groupby("race", observed=True)["y"].agg(["sum", "count"]).reset_index()
    grp = grp[grp["race"].notna() & (grp["count"] >= 100)]
    grp["rate"] = grp["sum"] / grp["count"]
    low, high = _wilson_ci(grp["sum"].to_numpy(), grp["count"].to_numpy())
    grp["err_low"] = grp["rate"] - low
    grp["err_high"] = high - grp["rate"]
    grp = grp.sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = palette("qual")[: len(grp)]
    y_pos = np.arange(len(grp))
    ax.barh(
        y_pos,
        grp["rate"] * 100,
        xerr=[grp["err_low"] * 100, grp["err_high"] * 100],
        color=colors,
        height=0.55,
        edgecolor="white",
        capsize=4,
        error_kw=dict(lw=1.2, ecolor="#374151"),
    )
    overall = df["y"].mean() * 100
    ax.axvline(
        overall, color=COLOR_RATE, linestyle="--", linewidth=1.0, label=f"Overall: {overall:.2f}%"
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(grp["race"])
    ax.set_xlabel("30-day readmit rate (%)")
    ax.set_ylabel("")
    ax.invert_yaxis()
    for i, (r, n) in enumerate(zip(grp["rate"], grp["count"], strict=False)):
        ax.text(
            r * 100 + 0.12,
            i,
            f"{r*100:.2f}%   n={_format_int(int(n))}",
            va="center",
            fontsize=10,
            color="#1F2937",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
        )
    ax.set_xlim(0, max(grp["rate"]) * 100 * 1.45)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.legend(loc="lower right", frameon=False)

    _suptitle(
        fig,
        "30-day readmission by race",
        "Mandatory subgroup view — informs Phase-9 fairness audit (Fairlearn)",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


# ---------------------------------------------------------------------------
# 5. Length-of-stay distribution
# ---------------------------------------------------------------------------


def plot_los_distribution(df: pd.DataFrame) -> plt.Figure:
    df["los_days"].clip(lower=0)
    los_neg = df.loc[df["y"] == 0, "los_days"].clip(lower=0)
    los_pos = df.loc[df["y"] == 1, "los_days"].clip(lower=0)

    bins = np.logspace(0, np.log10(60), 32)

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax_hist.hist(
        np.clip(los_neg + 1, 1, 60),
        bins=bins,
        color=COLOR_NEG,
        alpha=0.55,
        label="Not readmitted",
        edgecolor="white",
        linewidth=0.4,
        density=True,
    )
    ax_hist.hist(
        np.clip(los_pos + 1, 1, 60),
        bins=bins,
        color=COLOR_POS,
        alpha=0.65,
        label="Readmitted ≤ 30d",
        edgecolor="white",
        linewidth=0.4,
        density=True,
    )
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Length of stay (days, log scale, capped at 60)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper right", frameon=False)
    ax_hist.set_title("LOS distribution by class", loc="left", fontsize=12, color="#374151")

    # boxplot
    data = [los_neg.clip(upper=60), los_pos.clip(upper=60)]
    bp = ax_box.boxplot(
        data,
        vert=True,
        patch_artist=True,
        widths=0.45,
        showfliers=False,
        medianprops=dict(color="#1F2937", linewidth=1.5),
        whiskerprops=dict(color="#374151", linewidth=1.0),
        capprops=dict(color="#374151", linewidth=1.0),
    )
    for patch, c in zip(bp["boxes"], [COLOR_NEG, COLOR_POS], strict=False):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("white")
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(["Not readmitted", "Readmitted ≤ 30d"])
    ax_box.set_ylabel("Length of stay (days, capped at 60)")
    medians = [float(np.median(d)) for d in data]
    for i, m in enumerate(medians):
        ax_box.text(i + 1, m + 1.0, f"median={m:.0f}d", ha="center", fontsize=10, color="#1F2937")
    ax_box.set_title("LOS — central tendency by class", loc="left", fontsize=12, color="#374151")

    _suptitle(
        fig,
        "Length of stay vs 30-day readmission",
        "Readmitted patients have longer index stays — strong, expected signal",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


# ---------------------------------------------------------------------------
# 6. Top admit-dx chapters by readmit rate
# ---------------------------------------------------------------------------


def plot_dx_chapter_rates(df: pd.DataFrame, *, top_k: int = 12, min_n: int = 5000) -> plt.Figure:
    sub = df[df["admit_dx_chapter"].notna()]
    grp = sub.groupby("admit_dx_chapter", observed=True)["y"].agg(["sum", "count"]).reset_index()
    grp = grp[grp["count"] >= min_n]
    grp["rate"] = grp["sum"] / grp["count"]
    low, high = _wilson_ci(grp["sum"].to_numpy(), grp["count"].to_numpy())
    grp["err_low"] = grp["rate"] - low
    grp["err_high"] = high - grp["rate"]
    grp = grp.sort_values("rate", ascending=False).head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    overall = df["y"].mean() * 100
    norm = (grp["rate"] - grp["rate"].min()) / max(grp["rate"].max() - grp["rate"].min(), 1e-9)
    cmap = plt.cm.get_cmap("YlOrRd")
    colors = [cmap(0.30 + 0.65 * v) for v in norm]
    y_pos = np.arange(len(grp))

    ax.hlines(y_pos, 0, grp["rate"] * 100, color="#D1D5DB", linewidth=1.5)
    ax.scatter(
        grp["rate"] * 100, y_pos, color=colors, s=120, edgecolor="white", linewidth=1.2, zorder=3
    )
    ax.errorbar(
        grp["rate"] * 100,
        y_pos,
        xerr=[grp["err_low"] * 100, grp["err_high"] * 100],
        fmt="none",
        ecolor="#9CA3AF",
        capsize=3,
        lw=0.8,
        zorder=2,
    )
    ax.axvline(
        overall, color=COLOR_RATE, linestyle="--", linewidth=1.0, label=f"Overall: {overall:.2f}%"
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(grp["admit_dx_chapter"], fontsize=10)
    ax.set_xlabel("30-day readmit rate (%)")
    for i, (r, n) in enumerate(zip(grp["rate"], grp["count"], strict=False)):
        ax.text(
            r * 100 + 0.55,
            i,
            f"{r*100:.2f}%   n={_format_int(int(n))}",
            va="center",
            fontsize=9.5,
            color="#1F2937",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
        )
    ax.set_xlim(0, grp["rate"].max() * 100 * 1.42)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)

    _suptitle(
        fig,
        f"Top {top_k} admit-diagnosis chapters by readmission rate",
        f"ICD-9 chapter of the index admit; min {_format_int(min_n)} encounters per chapter; 95% CI shown",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


# ---------------------------------------------------------------------------
# 7. Prior 6-month inpatient count vs rate
# ---------------------------------------------------------------------------


def plot_prior_count_vs_rate(df: pd.DataFrame, *, max_count: int = 6) -> plt.Figure:
    sub = df[["prior_6mo_inpatient_count", "y"]].dropna().copy()
    sub["bucket"] = sub["prior_6mo_inpatient_count"].clip(upper=max_count).astype(int)
    grp = sub.groupby("bucket", observed=True)["y"].agg(["sum", "count"]).reset_index()
    # Filter buckets with too few samples (CI noise)
    grp = grp[grp["count"] >= 50]
    grp["rate"] = grp["sum"] / grp["count"]
    low, high = _wilson_ci(grp["sum"].to_numpy(), grp["count"].to_numpy())
    grp["err_low"] = grp["rate"] - low
    grp["err_high"] = high - grp["rate"]

    fig, ax_left = plt.subplots(figsize=(10.5, 5.0))
    ax_right = ax_left.twinx()
    ax_right.grid(False)

    ax_right.bar(
        grp["bucket"],
        grp["count"],
        color="#E5E7EB",
        width=0.65,
        label="# encounters",
        edgecolor="white",
        zorder=1,
    )
    ax_right.set_yscale("log")
    ax_right.set_ylabel("Number of encounters (log)", color="#6B7280")
    ax_right.tick_params(axis="y", colors="#6B7280")

    ax_left.fill_between(
        grp["bucket"],
        grp["err_low"] * 100,
        grp["err_high"] * 100,
        color=COLOR_RATE,
        alpha=0.18,
        linewidth=0,
    )
    ax_left.plot(
        grp["bucket"],
        grp["rate"] * 100,
        color=COLOR_RATE,
        marker="o",
        markersize=6,
        linewidth=2.2,
        zorder=3,
        label="Readmit rate (%)",
    )
    overall = df["y"].mean() * 100
    ax_left.axhline(
        overall,
        color=COLOR_NEUTRAL,
        linestyle="--",
        linewidth=1.0,
        label=f"Overall: {overall:.2f}%",
    )
    ax_left.set_xlabel(f"Prior 6-month inpatient encounters (capped at {max_count})")
    ax_left.set_ylabel("30-day readmit rate (%)", color=COLOR_RATE)
    ax_left.tick_params(axis="y", colors=COLOR_RATE)
    ax_left.set_xticks(range(0, int(grp["bucket"].max()) + 1))
    ax_left.set_zorder(ax_right.get_zorder() + 1)
    ax_left.patch.set_visible(False)

    handles_l, labels_l = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    fig.legend(
        handles_l + handles_r,
        labels_l + labels_r,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=2.0,
        handlelength=2.0,
        handletextpad=0.6,
    )

    _suptitle(
        fig,
        "Prior healthcare utilisation strongly predicts re-admission",
        "Readmit rate climbs steeply with the number of inpatient stays in the prior 6 months",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.82))
    return fig


# ---------------------------------------------------------------------------
# 8. Chronic condition flags vs readmit rate
# ---------------------------------------------------------------------------


def plot_chronic_conditions(df: pd.DataFrame) -> plt.Figure:
    cols = [c for c in df.columns if c.startswith("chronic_")]
    rows = []
    for c in cols:
        mask = df[c] == 1
        n = int(mask.sum())
        if n < 1000:
            continue
        k = int(df.loc[mask, "y"].sum())
        rate = k / n
        rows.append({"cond": c.replace("chronic_", "").upper(), "n": n, "rate": rate, "k": k})
    cond_df = pd.DataFrame(rows).sort_values("rate", ascending=True)
    low, high = _wilson_ci(cond_df["k"].to_numpy(), cond_df["n"].to_numpy())
    cond_df["err_low"] = cond_df["rate"] - low
    cond_df["err_high"] = high - cond_df["rate"]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    overall = df["y"].mean() * 100
    norm = (cond_df["rate"] - cond_df["rate"].min()) / max(
        cond_df["rate"].max() - cond_df["rate"].min(), 1e-9
    )
    cmap = plt.cm.get_cmap("YlOrRd")
    colors = [cmap(0.30 + 0.65 * v) for v in norm]
    y = np.arange(len(cond_df))
    ax.barh(
        y,
        cond_df["rate"] * 100,
        xerr=[cond_df["err_low"] * 100, cond_df["err_high"] * 100],
        color=colors,
        height=0.6,
        edgecolor="white",
        capsize=3,
        error_kw=dict(lw=1.0, ecolor="#374151"),
    )
    ax.axvline(
        overall, color=COLOR_RATE, linestyle="--", linewidth=1.0, label=f"Overall: {overall:.2f}%"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(cond_df["cond"])
    ax.set_xlabel("30-day readmit rate (%) | conditioned on flag = 1")
    ax.invert_yaxis()
    for i, (r, n) in enumerate(zip(cond_df["rate"], cond_df["n"], strict=False)):
        ax.text(
            r * 100 + 0.45,
            i,
            f"{r*100:.2f}%   n={_format_int(int(n))}",
            va="center",
            fontsize=9.5,
            color="#1F2937",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
        )
    ax.set_xlim(0, cond_df["rate"].max() * 100 * 1.42)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", frameon=False)

    _suptitle(
        fig,
        "Readmission rate by chronic-condition flag (CMS DE-SynPUF)",
        "Each bar = patients with that flag set; CHF, CKD and IHD dominate the risk gradient",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


# ---------------------------------------------------------------------------
# 9. Monthly volume + readmit rate over time
# ---------------------------------------------------------------------------


def plot_monthly_volume_and_rate(df: pd.DataFrame) -> plt.Figure:
    s = df[["admit_date", "y"]].dropna().copy()
    s["month"] = pd.to_datetime(s["admit_date"]).dt.to_period("M").dt.to_timestamp()
    grp = s.groupby("month").agg(volume=("y", "size"), rate=("y", "mean")).reset_index()
    grp = grp[(grp["month"] >= "2008-01-01") & (grp["month"] <= "2010-12-01")]

    fig, ax_vol = plt.subplots(figsize=(11.5, 5.0))
    ax_rate = ax_vol.twinx()
    ax_rate.grid(False)

    ax_vol.bar(
        grp["month"],
        grp["volume"],
        width=22,
        color=COLOR_VOLUME,
        alpha=0.55,
        edgecolor="white",
        label="Monthly admit volume",
    )
    ax_vol.set_ylabel("Inpatient admissions / month", color=COLOR_VOLUME)
    ax_vol.tick_params(axis="y", colors=COLOR_VOLUME)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax_rate.plot(
        grp["month"],
        grp["rate"] * 100,
        color=COLOR_RATE,
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="30-day readmit rate (%)",
    )
    ax_rate.set_ylabel("30-day readmit rate (%)", color=COLOR_RATE)
    ax_rate.tick_params(axis="y", colors=COLOR_RATE)
    overall = df["y"].mean() * 100
    ax_rate.axhline(overall, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.0)

    ax_vol.set_xlabel("Admission month")
    handles_l, labels_l = ax_vol.get_legend_handles_labels()
    handles_r, labels_r = ax_rate.get_legend_handles_labels()
    ax_vol.legend(handles_l + handles_r, labels_l + labels_r, loc="upper right", frameon=False)
    fig.autofmt_xdate(rotation=0, ha="center")

    _suptitle(
        fig,
        "Inpatient volume & 30-day readmission rate over time",
        "Both volume and rate decline in 2009–2010 — known CMS DE-SynPUF synthesis property; informs Phase-4 split strategy",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


# ---------------------------------------------------------------------------
# 10. Missingness
# ---------------------------------------------------------------------------


def plot_missingness(df: pd.DataFrame, *, top_k: int = 20) -> plt.Figure:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(top_k)
    if len(miss) == 0:
        miss = df.isna().mean().sort_values(ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.32 * len(miss) + 1.5)))
    cmap = plt.cm.get_cmap("Blues")
    norm = (miss.values - miss.values.min()) / max(miss.values.max() - miss.values.min(), 1e-9)
    colors = [cmap(0.30 + 0.55 * v) for v in norm]
    y = np.arange(len(miss))
    ax.barh(y, miss.values * 100, color=colors, height=0.65, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(miss.index)
    ax.set_xlabel("% missing")
    ax.invert_yaxis()
    for i, v in enumerate(miss.values):
        ax.text(v * 100 + 0.4, i, f"{v*100:.1f}%", va="center", fontsize=9.5, color="#1F2937")
    ax.set_xlim(0, max(miss.values.max() * 100, 5) * 1.18)
    ax.grid(axis="y", visible=False)

    _suptitle(
        fig,
        f"Missingness — top {len(miss)} columns",
        "Cohort-level NA rate after Phase-2 derivation; informs Phase-4 imputation strategy",
    )
    _add_caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

PLOTS: dict[str, callable] = {
    "01_label_balance": plot_label_balance,
    "02_age_distribution": plot_age_distribution,
    "03_sex_rate": plot_sex_rate,
    "04_race_rate": plot_race_rate,
    "05_los_distribution": plot_los_distribution,
    "06_dx_chapter_rates": plot_dx_chapter_rates,
    "07_prior_count_vs_rate": plot_prior_count_vs_rate,
    "08_chronic_conditions": plot_chronic_conditions,
    "09_monthly_volume_and_rate": plot_monthly_volume_and_rate,
    "10_missingness": plot_missingness,
}


def load_cohort(parquet_path: str | Path = "data/processed/cohort.parquet") -> pd.DataFrame:
    """Load cohort parquet as a pandas DataFrame (via Polars for speed)."""
    p = Path(parquet_path)
    logger.info("Loading cohort from %s ...", p)
    df = pl.read_parquet(p).to_pandas()
    logger.info("  → %s rows, %s cols", f"{len(df):,}", df.shape[1])
    return df


def run_all(
    parquet_path: str | Path = "data/processed/cohort.parquet",
    out_dir: str | Path = "reports/figures",
    only: Iterable[str] | None = None,
) -> list[Path]:
    """Generate every EDA figure to ``out_dir``. Returns list of PNG paths."""
    apply_style()
    df = load_cohort(parquet_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(only) if only else None
    written: list[Path] = []
    for name, fn in PLOTS.items():
        if keep and name not in keep:
            continue
        logger.info("→ %s", name)
        fig = fn(df)
        png = save_fig(fig, out_dir / f"{name}.png")
        written.append(png)
    logger.info("wrote %d figures to %s", len(written), out_dir)
    return written


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    # Silence noisy fonttools subsetter (matplotlib calls it for every PDF save)
    for lib in ("fontTools", "fontTools.subset", "matplotlib.font_manager"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
