"""Pairwise frailty correlation wrappers for phenotype statistics.

Higher-level convenience functions that dispatch to the core Weibull MLE
estimators in fit_ace.frailty.weibull_mle.
"""

from __future__ import annotations

__all__ = [
    "compute_pair_corr",
    "compute_weibull_pair_corr",
    "cross_trait_corr_se",
    "km_censoring_weights",
]

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from fit_ace.frailty.weibull_mle import cross_trait_weibull_corr_se, pairwise_weibull_corr_se
from sim_ace.core.utils import PAIR_TYPES

logger = logging.getLogger(__name__)


def km_censoring_weights(
    t_obs: np.ndarray,
    event: np.ndarray,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """Compute IPCW weights from the Kaplan-Meier censoring survival function.

    For the censoring KM, roles are swapped: censoring (delta=0) is the "event"
    and disease onset (delta=1) is the "censoring of censoring". G(t) = P(not
    censored by time t).

    Args:
        t_obs: observed times, shape (n,)
        event: event indicators (1=observed event, 0=censored), shape (n,)
        clip_percentile: percentile at which to cap raw weights (default 99)

    Returns:
        weights array, shape (n,), normalized to sum to n.
    """
    t_obs = np.asarray(t_obs, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    n = len(t_obs)

    # For censoring KM: "event" = censored (1 - delta)
    censor_event = 1.0 - event

    # Sort by time
    order = np.argsort(t_obs)
    c_sorted = censor_event[order]

    # KM estimator: G(t_i) = prod_{j: t_j <= t_i} (1 - d_j / n_j)
    at_risk = np.arange(n, 0, -1, dtype=np.float64)
    hazard = c_sorted / at_risk
    surv = np.cumprod(1.0 - hazard)

    # Map back to original order
    g_values = np.empty(n, dtype=np.float64)
    g_values[order] = surv

    # Clip G away from 0 and compute raw weights
    g_values = np.maximum(g_values, 1e-10)
    raw_weights = 1.0 / g_values

    # Clip extreme weights at percentile
    if clip_percentile < 100:
        cap = np.percentile(raw_weights, clip_percentile)
        raw_weights = np.minimum(raw_weights, cap)

    # Normalize so weights sum to n
    raw_weights *= n / raw_weights.sum()

    return raw_weights


def cross_trait_corr_se(
    t1: np.ndarray,
    delta1: np.ndarray,
    t2: np.ndarray,
    delta2: np.ndarray,
    beta1: float,
    beta2: float,
    hazard_model_1: str,
    hazard_params_1: dict,
    hazard_model_2: str,
    hazard_params_2: dict,
    n_quad: int = 20,
    ipcw_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Cross-trait frailty correlation with generalized baseline hazard.

    Currently only Weibull baselines are supported; other models raise
    NotImplementedError (frailty correlation for non-Weibull is future work).
    """
    if hazard_model_1 != "weibull" or hazard_model_2 != "weibull":
        raise NotImplementedError(
            f"cross_trait_corr_se only supports Weibull baselines, got '{hazard_model_1}' and '{hazard_model_2}'"
        )
    return cross_trait_weibull_corr_se(
        t1,
        delta1,
        t2,
        delta2,
        scale1=hazard_params_1["scale"],
        rho1=hazard_params_1["rho"],
        beta1=beta1,
        scale2=hazard_params_2["scale"],
        rho2=hazard_params_2["rho"],
        beta2=beta2,
        n_quad=n_quad,
        ipcw_weights=ipcw_weights,
    )


def compute_pair_corr(
    df: pd.DataFrame,
    trait_num: int,
    beta: float,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    hazard_model: str,
    hazard_params: dict,
    n_quad: int = 20,
    min_pairs: int = 10,
    max_pairs: int = 100_000,
    seed: int = 42,
    use_raw: bool = False,
) -> dict[str, dict[str, Any]]:
    """Pairwise frailty correlation with generalized baseline hazard.

    Currently only Weibull baselines are supported; other models raise
    NotImplementedError.
    """
    if hazard_model != "weibull":
        raise NotImplementedError(f"compute_pair_corr only supports Weibull baseline, got '{hazard_model}'")
    return compute_weibull_pair_corr(
        df=df,
        trait_num=trait_num,
        scale=hazard_params["scale"],
        rho=hazard_params["rho"],
        beta=beta,
        pairs=pairs,
        n_quad=n_quad,
        min_pairs=min_pairs,
        max_pairs=max_pairs,
        seed=seed,
        use_raw=use_raw,
    )


def compute_weibull_pair_corr(
    df: pd.DataFrame,
    trait_num: int,
    scale: float,
    rho: float,
    beta: float,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    n_quad: int = 20,
    min_pairs: int = 10,
    max_pairs: int = 100_000,
    seed: int = 42,
    use_raw: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute pairwise Weibull correlation for all relationship types, one trait.

    Args:
        df: Phenotype DataFrame with t_observed{trait_num} and affected{trait_num}.
        trait_num: 1 or 2.
        scale: Weibull scale parameter (lifelines convention).
        rho: Weibull shape parameter (lifelines convention).
        beta: Log-hazard-ratio coefficient linking liability to hazard.
        pairs: Pre-extracted relationship pair indices.
        n_quad: quadrature nodes per dimension
        min_pairs: skip relationship types with fewer pairs
        max_pairs: subsample if more pairs than this
        seed: random seed for subsampling
        use_raw: if True, use uncensored t{trait_num} with delta=1 for all individuals

    Returns:
        Dict keyed by pair type, each {"r": float|None, "se": float|None, "n_pairs": int}
    """
    if use_raw:
        t_obs = df[f"t{trait_num}"].values.astype(np.float64)
        delta = np.ones(len(df), dtype=np.float64)
    else:
        t_obs = df[f"t_observed{trait_num}"].values.astype(np.float64)
        delta = df[f"affected{trait_num}"].values.astype(np.float64)
    rng = np.random.default_rng(seed)

    result: dict[str, dict[str, float | int | None]] = {}
    for ptype in PAIR_TYPES:
        idx1, idx2 = pairs[ptype]
        n_p = len(idx1)
        if n_p < min_pairs:
            result[ptype] = {"r": None, "se": None, "n_pairs": int(n_p)}
            continue

        # Subsample if too many pairs
        if n_p > max_pairs:
            sel = rng.choice(n_p, max_pairs, replace=False)
            idx1 = idx1[sel]
            idx2 = idx2[sel]
            logger.info(
                "Weibull corr %s trait%d: subsampled %d -> %d pairs",
                ptype,
                trait_num,
                n_p,
                max_pairs,
            )

        r, se = pairwise_weibull_corr_se(
            t_obs[idx1],
            delta[idx1],
            t_obs[idx2],
            delta[idx2],
            scale,
            rho,
            beta,
            n_quad=n_quad,
        )

        result[ptype] = {
            "r": float(r) if not np.isnan(r) else None,
            "se": float(se) if not np.isnan(se) else None,
            "n_pairs": int(n_p),
        }

    return result
