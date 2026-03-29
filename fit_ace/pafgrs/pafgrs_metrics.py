"""Validation metrics for PA-FGRS scores against known truth.

Computes: Pearson r, R², mean bias, AUC, and variance calibration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sim_ace.core.utils import fast_pearsonr

logger = logging.getLogger(__name__)


def _compute_trait_metrics(
    est: np.ndarray,
    true_a: np.ndarray,
    var_est: np.ndarray,
    affected: np.ndarray,
) -> dict[str, float]:
    """Core metric computation for a single trait's PA-FGRS scores."""
    n = len(est)
    affected = np.asarray(affected, dtype=bool)

    if n < 3 or np.std(est) < 1e-15 or np.std(true_a) < 1e-15:
        r_val, r2_val = 0.0, 0.0
    else:
        r_val, _ = fast_pearsonr(est, true_a)
        r2_val = r_val**2

    bias = float(np.mean(est - true_a))

    n_pos = int(affected.sum())
    n_neg = int((~affected).sum())
    auc = _fast_auc(est, affected) if n_pos > 0 and n_neg > 0 else float("nan")

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


def compute_pafgrs_metrics(scores_df: pd.DataFrame) -> dict[str, float]:
    """Compute validation metrics for PA-FGRS scores.

    Parameters
    ----------
    scores_df : DataFrame with columns ``est``, ``var``, ``true_A``, ``affected``.

    Returns
    -------
    Dict with keys: r, r2, bias, auc, var_calibration, n_scored.
    """
    return _compute_trait_metrics(
        scores_df["est"].values,
        scores_df["true_A"].values,
        scores_df["var"].values,
        scores_df["affected"].values,
    )


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


# ---------------------------------------------------------------------------
# Bivariate metrics
# ---------------------------------------------------------------------------


def compute_bivariate_metrics(
    scores_df: pd.DataFrame,
    univariate_r1: float | None = None,
    univariate_r2: float | None = None,
) -> dict[str, float]:
    """Compute validation metrics for bivariate PA-FGRS scores.

    Parameters
    ----------
    scores_df : DataFrame with columns ``est_1``, ``est_2``, ``var_1``,
        ``var_2``, ``cov_12``, ``true_A1``, ``true_A2``,
        ``affected1``, ``affected2``.
    univariate_r1, univariate_r2 : optional univariate Pearson r for
        computing improvement (delta_r).

    Returns
    -------
    Dict with per-trait marginal metrics and joint metrics.
    """
    n = len(scores_df)

    result: dict[str, float] = {"n_scored": n}

    # Per-trait marginal metrics (reuse shared helper)
    for t in (1, 2):
        trait_m = _compute_trait_metrics(
            scores_df[f"est_{t}"].values,
            scores_df[f"true_A{t}"].values,
            scores_df[f"var_{t}"].values,
            scores_df[f"affected{t}"].values,
        )
        result[f"r_{t}"] = trait_m["r"]
        result[f"r2_{t}"] = trait_m["r2"]
        result[f"bias_{t}"] = trait_m["bias"]
        result[f"auc_{t}"] = trait_m["auc"]
        result[f"var_calibration_{t}"] = trait_m["var_calibration"]
        result[f"n_affected_{t}"] = trait_m["n_affected"]

    # Improvement over univariate (delta_r)
    if univariate_r1 is not None:
        result["delta_r_1"] = round(result["r_1"] - univariate_r1, 6)
    if univariate_r2 is not None:
        result["delta_r_2"] = round(result["r_2"] - univariate_r2, 6)

    # Cross-trait posterior covariance calibration
    cov_12 = scores_df["cov_12"].values
    true_cross = (scores_df["true_A1"].values - scores_df["est_1"].values) * (
        scores_df["true_A2"].values - scores_df["est_2"].values
    )
    mean_cov = float(np.mean(cov_12))
    mean_cross_err = float(np.mean(true_cross))
    result["cov_calibration"] = round(mean_cov / mean_cross_err, 6) if abs(mean_cross_err) > 1e-15 else float("nan")

    return result
