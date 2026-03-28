"""Validation metrics for PA-FGRS scores against known truth.

Computes: Pearson r, R², mean bias, AUC, and variance calibration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def compute_pafgrs_metrics(scores_df: pd.DataFrame) -> dict[str, float]:
    """Compute validation metrics for PA-FGRS scores.

    Parameters
    ----------
    scores_df : DataFrame with columns ``est``, ``var``, ``true_A``, ``affected``.

    Returns
    -------
    Dict with keys: r, r2, bias, auc, var_calibration, n_scored.
    """
    est = scores_df["est"].values
    true_a = scores_df["true_A"].values
    var_est = scores_df["var"].values
    affected = scores_df["affected"].values.astype(bool)

    # Only score individuals who actually got relatives (est != 0 or var != h2_max)
    # Use all individuals — even those with 0 relatives contribute to calibration
    n = len(est)

    # Pearson correlation and R²
    if n < 3 or np.std(est) < 1e-15 or np.std(true_a) < 1e-15:
        r_val, r2_val = 0.0, 0.0
    else:
        r_val, _ = pearsonr(est, true_a)
        r2_val = r_val**2

    # Mean bias
    bias = float(np.mean(est - true_a))

    # AUC: discriminative ability for affected vs unaffected
    n_pos, n_neg = int(affected.sum()), int((~affected).sum())
    if n_pos > 0 and n_neg > 0:
        auc = _fast_auc(est, affected)
    else:
        auc = float("nan")

    # Variance calibration: mean(var) vs mean((est - true_A)²)
    mean_reported_var = float(np.mean(var_est))
    mean_actual_mse = float(np.mean((est - true_a) ** 2))
    var_calibration = mean_reported_var / mean_actual_mse if mean_actual_mse > 1e-15 else float("nan")

    return {
        "r": round(float(r_val), 6),
        "r2": round(float(r2_val), 6),
        "bias": round(bias, 6),
        "auc": round(float(auc), 6),
        "var_calibration": round(var_calibration, 6),
        "mean_reported_var": round(mean_reported_var, 6),
        "mean_actual_mse": round(mean_actual_mse, 6),
        "n_scored": n,
        "n_affected": n_pos,
    }


def _fast_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC via the Mann-Whitney U statistic (no sklearn dependency)."""
    pos = scores[labels]
    neg = scores[~labels]
    # For each positive, count how many negatives it exceeds
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Handle ties: average ranks
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    n_pos = len(pos)
    n_neg = len(neg)
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def write_metrics_tsv(
    metrics: dict[str, float],
    path: str | Path,
    trait: str = "trait1",
    cip_source: str = "empirical",
    h2_source: str = "true",
) -> None:
    """Write metrics to a TSV file."""
    rows = []
    for metric_name, value in metrics.items():
        rows.append(
            {
                "trait": trait,
                "cip_source": cip_source,
                "h2_source": h2_source,
                "metric": metric_name,
                "value": value,
            }
        )
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    logger.info("Wrote PA-FGRS metrics to %s", path)


def read_and_combine_metrics(tsv_paths: list[str | Path]) -> pd.DataFrame:
    """Read and concatenate multiple metrics TSV files."""
    dfs = [pd.read_csv(p, sep="\t") for p in tsv_paths if Path(p).exists()]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
