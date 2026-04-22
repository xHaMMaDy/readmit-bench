"""Exploratory Data Analysis plotting package.

All functions return a `matplotlib.figure.Figure` and (when called via
:func:`run_all`) save a 300dpi PNG + vector PDF to ``reports/figures/``.
"""

from .plots import (
    plot_age_distribution,
    plot_chronic_conditions,
    plot_dx_chapter_rates,
    plot_label_balance,
    plot_los_distribution,
    plot_missingness,
    plot_monthly_volume_and_rate,
    plot_prior_count_vs_rate,
    plot_race_rate,
    plot_sex_rate,
    run_all,
)

__all__ = [
    "plot_age_distribution",
    "plot_chronic_conditions",
    "plot_dx_chapter_rates",
    "plot_label_balance",
    "plot_los_distribution",
    "plot_missingness",
    "plot_monthly_volume_and_rate",
    "plot_prior_count_vs_rate",
    "plot_race_rate",
    "plot_sex_rate",
    "run_all",
]
