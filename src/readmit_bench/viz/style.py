"""Modern, professional, scientific publication-grade plot style.

Use::

    from readmit_bench.viz import apply_style, palette, save_fig
    apply_style()                     # call once at notebook/script top

    fig, ax = plt.subplots()
    ax.plot(..., color=palette()[0])
    save_fig(fig, "reports/figures/my_plot.png")

Design choices
--------------
* **Font stack** — falls back gracefully across OSes:
  Inter → IBM Plex Sans → Source Sans 3 → Segoe UI → Helvetica Neue → Arial → DejaVu Sans.
* **Palette** — custom *readmit-bench Pro* palette: a curated 10-color qualitative
  ramp inspired by Tableau 10 + Okabe-Ito (color-blind safe), tuned for white background.
* **Axes** — bottom + left spines only (Tufte-style), light gridlines on y, no top/right.
* **Sizes** — figure 7×4.5 in @ 150 dpi default; 11pt body, 13pt title, 10pt ticks.
* **Saving** — `save_fig` writes PNG (300 dpi) + PDF (vector) side-by-side.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

#: Primary qualitative palette — 10 colors, color-blind-safe, white-bg-tuned.
PRO_PALETTE_QUAL: list[str] = [
    "#2E5BFF",  # blue       (primary)
    "#FF7847",  # orange     (accent)
    "#1FB89A",  # teal-green
    "#E63946",  # red        (warn / negative)
    "#A06CD5",  # violet
    "#F4B400",  # amber
    "#0F9D58",  # green
    "#5F6B7A",  # slate
    "#D81B60",  # magenta
    "#00ACC1",  # cyan
]

#: Sequential palette for ordered categories / heat-style ramps (low → high).
PRO_PALETTE_SEQ: list[str] = [
    "#EFF3FF",
    "#C6DBEF",
    "#9ECAE1",
    "#6BAED6",
    "#4292C6",
    "#2171B5",
    "#08519C",
    "#08306B",
]

#: Diverging palette for signed quantities (negative ← 0 → positive).
PRO_PALETTE_DIV: list[str] = [
    "#B2182B",
    "#D6604D",
    "#F4A582",
    "#FDDBC7",
    "#F7F7F7",
    "#D1E5F0",
    "#92C5DE",
    "#4393C3",
    "#2166AC",
]

#: Default name used when palette() is called with no argument.
PRO_PALETTE = PRO_PALETTE_QUAL


def palette(name: str = "qual") -> list[str]:
    """Return one of the project palettes by short name."""
    table = {"qual": PRO_PALETTE_QUAL, "seq": PRO_PALETTE_SEQ, "div": PRO_PALETTE_DIV}
    if name not in table:
        raise ValueError(f"unknown palette {name!r}; choose from {list(table)}")
    return list(table[name])


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_FONT_STACK = [
    "Inter",
    "IBM Plex Sans",
    "Source Sans 3",
    "Source Sans Pro",
    "Segoe UI",
    "Helvetica Neue",
    "Arial",
    "DejaVu Sans",
]


def apply_style(*, dpi: int = 150, figsize: tuple[float, float] = (7.0, 4.5)) -> None:
    """Install the project's matplotlib rcParams. Call once per notebook/script."""
    mpl.rcParams.update(
        {
            # ---------- font ----------
            "font.family": "sans-serif",
            "font.sans-serif": _FONT_STACK,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelweight": "regular",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.titleweight": "semibold",
            "mathtext.fontset": "stixsans",
            # ---------- figure ----------
            "figure.figsize": figsize,
            "figure.dpi": dpi,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            # ---------- axes ----------
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.titlepad": 12,
            "axes.labelpad": 6,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=PRO_PALETTE_QUAL),
            # ---------- grid ----------
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "grid.alpha": 1.0,
            # ---------- ticks ----------
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            # ---------- legend ----------
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.borderpad": 0.4,
            # ---------- lines / markers ----------
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "patch.linewidth": 0.5,
            "patch.edgecolor": "white",
            # ---------- pdf / svg ----------
            "pdf.fonttype": 42,  # embed as TrueType (selectable text)
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def finalize(
    ax,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_loc: str | None = None,
) -> None:
    """One-call cosmetic polish for an axes after plotting."""
    if title:
        ax.set_title(title, loc="left", pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend_loc and ax.get_legend_handles_labels()[1]:
        ax.legend(loc=legend_loc, frameon=False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
    ax.tick_params(colors="#333333", which="both")


def save_fig(fig, path: str | Path, *, also_pdf: bool = True) -> Path:
    """Save a figure to PNG (300dpi) and (optionally) a vector PDF beside it.

    Returns the PNG path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() != ".png":
        p = p.with_suffix(".png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    if also_pdf:
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p
