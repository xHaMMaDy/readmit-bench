"""Unified evaluation metrics + curve helpers for readmit-bench.

Single source of truth so every model in the benchmark is judged on identical
definitions. Heavily imbalanced (~9.6% positive) → PR-AUC is the primary
metric. Brier and log-loss capture calibration quality (matters for the
cost-based threshold step in Phase 7). recall@top-K and precision@top-K
mirror the clinical decision: "we can intervene on K% of patients per month".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class Metrics:
    pr_auc: float
    roc_auc: float
    brier: float
    log_loss: float
    recall_at_top10: float
    precision_at_top10: float
    n: int
    n_pos: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _at_top_k(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> tuple[float, float]:
    n = len(y_true)
    k = max(1, int(round(n * top_frac)))
    idx = np.argpartition(-y_score, k - 1)[:k]
    selected = y_true[idx]
    n_pos_total = int(y_true.sum())
    if n_pos_total == 0:
        return 0.0, 0.0
    recall = float(selected.sum() / n_pos_total)
    precision = float(selected.mean())
    return recall, precision


def compute_metrics(y_true, y_score) -> Metrics:
    """Compute the full readmit-bench metric panel."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_score={y_score.shape}")

    eps = 1e-7
    y_score_clipped = np.clip(y_score, eps, 1 - eps)
    recall_top10, precision_top10 = _at_top_k(y_true, y_score, 0.10)

    return Metrics(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, y_score)),
        log_loss=float(log_loss(y_true, y_score_clipped)),
        recall_at_top10=recall_top10,
        precision_at_top10=precision_top10,
        n=int(len(y_true)),
        n_pos=int(y_true.sum()),
    )


def pr_curve_points(y_true, y_score, max_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Return down-sampled (recall, precision) points suitable for plotting."""
    p, r, _ = precision_recall_curve(y_true, y_score)
    if len(p) > max_points:
        idx = np.linspace(0, len(p) - 1, max_points).astype(int)
        p, r = p[idx], r[idx]
    return r, p


def roc_curve_points(y_true, y_score, max_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr


def reliability_curve(
    y_true, y_score, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantile-binned reliability curve: (mean_pred, frac_pos, bin_count)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    quantiles = np.quantile(y_score, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    bins = np.digitize(y_score, quantiles[1:-1])
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        mean_pred.append(y_score[mask].mean())
        frac_pos.append(y_true[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)
