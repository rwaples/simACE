"""
Pairwise Weibull frailty likelihood estimator for liability correlation.

For a pair (i, j) with correlated liabilities (L_i, L_j) ~ BVN(0, 0, 1, 1, r),
computes the MLE of r from paired survival data (t_observed, affected).

Model (lifelines scale/shape convention):
    h(t | L) = (rho/scale) * (t/scale)^(rho-1) * exp(beta * L)
    S(t | L) = exp(-(t/scale)^rho * exp(beta * L))
    g(t, delta, L) = [h(t|L)]^delta * S(t|L)

Integration via Gauss-Hermite quadrature (probabilist's convention).

Note on censoring: all non-affected individuals are treated as right-censored
at their observed time (delta=0 contributes S(t_obs|L)). This is exact for
right-censored and death-censored cases. For left-truncated individuals
(onset before observation window), t_observed is set to the left boundary,
so S(left|L) ~ 1.0 — conservative but not biased.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

if TYPE_CHECKING:
    import pandas as pd
from scipy.optimize import minimize_scalar

try:
    from numba import njit, prange

    @njit(cache=True)
    def _pair_log_lik(
        p: int,
        r: float,
        sqrt_1mr2: float,
        delta_i: np.ndarray,
        delta_j: np.ndarray,
        const_i: np.ndarray,
        const_j: np.ndarray,
        hazard_base_i: np.ndarray,
        hazard_base_j: np.ndarray,
        beta: float,
        nodes: np.ndarray,
        log_weights: np.ndarray,
    ) -> float:
        """Log-likelihood for a single pair, summed over quadrature grid."""
        n_quad = len(nodes)
        # First pass: find max for log-sum-exp stability
        max_val = -1e300
        for m in range(n_quad):
            beta_li = beta * nodes[m]
            exp_beta_li = math.exp(beta_li)
            log_g_i = delta_i[p] * (const_i[p] + beta_li) - hazard_base_i[p] * exp_beta_li
            for n in range(n_quad):
                lj = r * nodes[m] + sqrt_1mr2 * nodes[n]
                beta_lj = beta * lj
                log_g_j = delta_j[p] * (const_j[p] + beta_lj) - hazard_base_j[p] * math.exp(beta_lj)
                val = log_g_i + log_g_j + log_weights[m] + log_weights[n]
                if val > max_val:
                    max_val = val
        # Second pass: accumulate
        acc = 0.0
        for m in range(n_quad):
            beta_li = beta * nodes[m]
            exp_beta_li = math.exp(beta_li)
            log_g_i = delta_i[p] * (const_i[p] + beta_li) - hazard_base_i[p] * exp_beta_li
            for n in range(n_quad):
                lj = r * nodes[m] + sqrt_1mr2 * nodes[n]
                beta_lj = beta * lj
                log_g_j = delta_j[p] * (const_j[p] + beta_lj) - hazard_base_j[p] * math.exp(beta_lj)
                val = log_g_i + log_g_j + log_weights[m] + log_weights[n]
                acc += math.exp(val - max_val)
        return max_val + math.log(acc)

    @njit(cache=True, parallel=True)
    def _neg_log_lik_numba(
        r: float,
        delta_i: np.ndarray,
        delta_j: np.ndarray,
        const_i: np.ndarray,
        const_j: np.ndarray,
        hazard_base_i: np.ndarray,
        hazard_base_j: np.ndarray,
        beta: float,
        nodes: np.ndarray,
        log_weights: np.ndarray,
    ) -> float:
        """Fused neg-log-likelihood loop over pairs and quadrature nodes."""
        n_pairs = len(delta_i)
        sqrt_1mr2 = math.sqrt(max(1.0 - r * r, 1e-10))
        pair_ll = np.empty(n_pairs)
        for p in prange(n_pairs):
            pair_ll[p] = _pair_log_lik(
                p,
                r,
                sqrt_1mr2,
                delta_i,
                delta_j,
                const_i,
                const_j,
                hazard_base_i,
                hazard_base_j,
                beta,
                nodes,
                log_weights,
            )
        return -pair_ll.sum()

    @njit(cache=True)
    def _pair_log_lik_cross(
        p: int,
        r: float,
        sqrt_1mr2: float,
        delta_i: np.ndarray,
        delta_j: np.ndarray,
        const_i: np.ndarray,
        const_j: np.ndarray,
        hazard_base_i: np.ndarray,
        hazard_base_j: np.ndarray,
        beta_i: float,
        beta_j: float,
        nodes: np.ndarray,
        log_weights: np.ndarray,
    ) -> float:
        """Log-likelihood for a single cross-trait pair (separate beta per trait)."""
        n_quad = len(nodes)
        max_val = -1e300
        for m in range(n_quad):
            bli = beta_i * nodes[m]
            exp_bli = math.exp(bli)
            log_g_i = delta_i[p] * (const_i[p] + bli) - hazard_base_i[p] * exp_bli
            for n in range(n_quad):
                lj = r * nodes[m] + sqrt_1mr2 * nodes[n]
                blj = beta_j * lj
                log_g_j = delta_j[p] * (const_j[p] + blj) - hazard_base_j[p] * math.exp(blj)
                val = log_g_i + log_g_j + log_weights[m] + log_weights[n]
                if val > max_val:
                    max_val = val
        acc = 0.0
        for m in range(n_quad):
            bli = beta_i * nodes[m]
            exp_bli = math.exp(bli)
            log_g_i = delta_i[p] * (const_i[p] + bli) - hazard_base_i[p] * exp_bli
            for n in range(n_quad):
                lj = r * nodes[m] + sqrt_1mr2 * nodes[n]
                blj = beta_j * lj
                log_g_j = delta_j[p] * (const_j[p] + blj) - hazard_base_j[p] * math.exp(blj)
                val = log_g_i + log_g_j + log_weights[m] + log_weights[n]
                acc += math.exp(val - max_val)
        return max_val + math.log(acc)

    @njit(cache=True, parallel=True)
    def _neg_log_lik_numba_cross(
        r: float,
        delta_i: np.ndarray,
        delta_j: np.ndarray,
        const_i: np.ndarray,
        const_j: np.ndarray,
        hazard_base_i: np.ndarray,
        hazard_base_j: np.ndarray,
        beta_i: float,
        beta_j: float,
        nodes: np.ndarray,
        log_weights: np.ndarray,
        ipcw_weights: np.ndarray,
    ) -> float:
        """Fused neg-log-likelihood for cross-trait pairs (separate beta per trait)."""
        n_pairs = len(delta_i)
        sqrt_1mr2 = math.sqrt(max(1.0 - r * r, 1e-10))
        pair_ll = np.empty(n_pairs)
        for p in prange(n_pairs):
            pair_ll[p] = _pair_log_lik_cross(
                p,
                r,
                sqrt_1mr2,
                delta_i,
                delta_j,
                const_i,
                const_j,
                hazard_base_i,
                hazard_base_j,
                beta_i,
                beta_j,
                nodes,
                log_weights,
            )
        return -(ipcw_weights * pair_ll).sum()

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False

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
    _t_sorted = t_obs[order]
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


PAIR_TYPES = [
    "MZ twin",
    "Full sib",
    "Mother-offspring",
    "Father-offspring",
    "Maternal half sib",
    "Paternal half sib",
    "1st cousin",
]


def pairwise_weibull_corr_se(
    t_i: np.ndarray,
    delta_i: np.ndarray,
    t_j: np.ndarray,
    delta_j: np.ndarray,
    scale: float,
    rho: float,
    beta: float,
    n_quad: int = 20,
) -> tuple[float, float]:
    """Estimate liability correlation from paired Weibull survival data.

    Args:
        t_i, t_j: observed times for members of each pair, shape (n_pairs,)
        delta_i, delta_j: event indicators (1=affected, 0=censored), shape (n_pairs,)
        scale, rho, beta: Weibull frailty model parameters (lifelines convention)
        n_quad: number of Gauss-Hermite quadrature nodes per dimension

    Returns:
        (r_hat, se): estimated liability correlation and standard error.
    """
    t_i = np.asarray(t_i, dtype=np.float64)
    t_j = np.asarray(t_j, dtype=np.float64)
    delta_i = np.asarray(delta_i, dtype=np.float64)
    delta_j = np.asarray(delta_j, dtype=np.float64)

    n_pairs = len(t_i)
    if n_pairs < 10:
        return np.nan, np.nan

    # Guard against log(0)
    t_i = np.clip(t_i, 1e-10, None)
    t_j = np.clip(t_j, 1e-10, None)

    # Gauss-Hermite nodes and weights (probabilist's: weight function exp(-x^2/2))
    nodes, weights = hermegauss(n_quad)
    log_weights = np.log(weights)

    # Precomputed Weibull constants (lifelines convention)
    # Model: h(t|L) = (rho/scale) * (t/scale)^(rho-1) * exp(beta*L)
    #        S(t|L) = exp(-(t/scale)^rho * exp(beta*L))
    log_rho = np.log(rho)
    log_scale = np.log(scale)

    # Precompute per-individual terms that don't depend on L
    log_t_i = np.log(t_i)  # (n_pairs,)
    log_t_j = np.log(t_j)  # (n_pairs,)
    # H_base(t) = (t/scale)^rho = exp(rho * (log_t - log_scale))
    hazard_base_i = np.exp(rho * (log_t_i - log_scale))  # (t_i/scale)^rho
    hazard_base_j = np.exp(rho * (log_t_j - log_scale))  # (t_j/scale)^rho
    # log h_base = log(rho) - rho*log(scale) + (rho-1)*log(t)
    const_i = log_rho - rho * log_scale + (rho - 1) * log_t_i  # (n_pairs,)
    const_j = log_rho - rho * log_scale + (rho - 1) * log_t_j  # (n_pairs,)

    if _HAS_NUMBA:

        def neg_log_pairwise_lik(r: float) -> float:
            return _neg_log_lik_numba(
                r,
                delta_i,
                delta_j,
                const_i,
                const_j,
                hazard_base_i,
                hazard_base_j,
                beta,
                nodes,
                log_weights,
            )
    else:

        def neg_log_pairwise_lik(r: float) -> float:
            sqrt_1mr2 = np.sqrt(max(1.0 - r * r, 1e-10))

            li = nodes
            lj = r * nodes[:, None] + sqrt_1mr2 * nodes[None, :]

            beta_li = beta * li[None, :]
            log_g_i = delta_i[:, None] * (const_i[:, None] + beta_li) - hazard_base_i[:, None] * np.exp(beta_li)

            beta_lj = beta * lj[None, :, :]
            log_g_j = delta_j[:, None, None] * (const_j[:, None, None] + beta_lj) - hazard_base_j[
                :, None, None
            ] * np.exp(beta_lj)

            log_w2d = log_weights[:, None] + log_weights[None, :]
            log_integrand = log_g_i[:, :, None] + log_g_j + log_w2d[None, :, :]

            flat = log_integrand.reshape(n_pairs, -1)
            max_val = flat.max(axis=1, keepdims=True)
            log_pair_lik = max_val[:, 0] + np.log(np.sum(np.exp(flat - max_val), axis=1))

            return -np.sum(log_pair_lik)

    result = minimize_scalar(neg_log_pairwise_lik, bounds=(-0.999, 0.999), method="bounded")
    r_hat = result.x

    if np.isnan(r_hat):
        return np.nan, np.nan

    # Boundary hit: likelihood is monotone — estimate is unreliable
    if abs(r_hat) > 0.999:
        logger.warning(
            "pairwise_weibull_corr_se: estimate hit boundary (r=%.3f), likely too few events for reliable estimation",
            r_hat,
        )
        return np.nan, np.nan

    # SE via numerical Hessian (second difference)
    h = 1e-4
    r_lo = max(r_hat - h, -0.999)
    r_hi = min(r_hat + h, 0.999)
    f0 = neg_log_pairwise_lik(r_hat)
    fp = neg_log_pairwise_lik(r_hi)
    fm = neg_log_pairwise_lik(r_lo)
    d2 = (fp - 2 * f0 + fm) / (h * h)
    if d2 <= 0:
        se = np.nan
    else:
        se = 1.0 / np.sqrt(d2)

    return float(r_hat), float(se)


def cross_trait_weibull_corr_se(
    t1: np.ndarray,
    delta1: np.ndarray,
    t2: np.ndarray,
    delta2: np.ndarray,
    scale1: float,
    rho1: float,
    beta1: float,
    scale2: float,
    rho2: float,
    beta2: float,
    n_quad: int = 20,
    ipcw_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate cross-trait liability correlation from paired Weibull survival data.

    Each individual contributes a "pair" of (trait 1, trait 2) with potentially
    different Weibull parameters per trait.

    Args:
        t1, t2: observed times for trait 1 and trait 2, shape (n,)
        delta1, delta2: event indicators (1=affected, 0=censored), shape (n,)
        scale1, rho1, beta1: Weibull parameters for trait 1
        scale2, rho2, beta2: Weibull parameters for trait 2
        n_quad: number of Gauss-Hermite quadrature nodes per dimension
        ipcw_weights: optional IPCW weights, shape (n,). If None, uniform weights.

    Returns:
        (r_hat, se): estimated cross-trait liability correlation and standard error.
    """
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    delta1 = np.asarray(delta1, dtype=np.float64)
    delta2 = np.asarray(delta2, dtype=np.float64)

    n = len(t1)
    if n < 10:
        return np.nan, np.nan

    # Default: uniform weights (no IPCW)
    if ipcw_weights is None:
        _weights = np.ones(n, dtype=np.float64)
    else:
        _weights = np.asarray(ipcw_weights, dtype=np.float64)

    t1 = np.clip(t1, 1e-10, None)
    t2 = np.clip(t2, 1e-10, None)

    nodes, weights = hermegauss(n_quad)

    # Normalization constant not need for optimization
    # log_weights = np.log(weights) - 0.5 * np.log(2 * np.pi)
    log_weights = np.log(weights)

    # Precompute Weibull constants for trait 1
    log_rho1 = np.log(rho1)
    log_scale1 = np.log(scale1)
    log_t1 = np.log(t1)
    hazard_base_1 = np.exp(rho1 * (log_t1 - log_scale1))
    const_1 = log_rho1 - rho1 * log_scale1 + (rho1 - 1) * log_t1

    # Precompute Weibull constants for trait 2
    log_rho2 = np.log(rho2)
    log_scale2 = np.log(scale2)
    log_t2 = np.log(t2)
    hazard_base_2 = np.exp(rho2 * (log_t2 - log_scale2))
    const_2 = log_rho2 - rho2 * log_scale2 + (rho2 - 1) * log_t2

    if _HAS_NUMBA:

        def neg_log_lik(r: float) -> float:
            return _neg_log_lik_numba_cross(
                r,
                delta1,
                delta2,
                const_1,
                const_2,
                hazard_base_1,
                hazard_base_2,
                beta1,
                beta2,
                nodes,
                log_weights,
                _weights,
            )
    else:

        def neg_log_lik(r: float) -> float:
            sqrt_1mr2 = np.sqrt(max(1.0 - r * r, 1e-10))

            li = nodes
            lj = r * nodes[:, None] + sqrt_1mr2 * nodes[None, :]

            b_li = beta1 * li[None, :]
            log_g_i = delta1[:, None] * (const_1[:, None] + b_li) - hazard_base_1[:, None] * np.exp(b_li)

            b_lj = beta2 * lj[None, :, :]
            log_g_j = delta2[:, None, None] * (const_2[:, None, None] + b_lj) - hazard_base_2[:, None, None] * np.exp(
                b_lj
            )

            log_w2d = log_weights[:, None] + log_weights[None, :]
            log_integrand = log_g_i[:, :, None] + log_g_j + log_w2d[None, :, :]

            flat = log_integrand.reshape(n, -1)
            max_val = flat.max(axis=1, keepdims=True)
            log_pair_lik = max_val[:, 0] + np.log(np.sum(np.exp(flat - max_val), axis=1))

            return -np.sum(_weights * log_pair_lik)

    result = minimize_scalar(neg_log_lik, bounds=(-0.999, 0.999), method="bounded")
    r_hat = result.x

    if np.isnan(r_hat):
        return np.nan, np.nan

    if abs(r_hat) > 0.999:
        logger.warning(
            "cross_trait_weibull_corr_se: estimate hit boundary (r=%.3f)",
            r_hat,
        )
        return np.nan, np.nan

    # SE via numerical Hessian
    h = 1e-4
    r_lo = max(r_hat - h, -0.999)
    r_hi = min(r_hat + h, 0.999)
    f0 = neg_log_lik(r_hat)
    fp = neg_log_lik(r_hi)
    fm = neg_log_lik(r_lo)
    d2 = (fp - 2 * f0 + fm) / (h * h)
    if d2 <= 0:
        se = np.nan
    else:
        se = 1.0 / np.sqrt(d2)

    return float(r_hat), float(se)


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
        df: phenotype DataFrame with t_observed{trait_num} and affected{trait_num}
        trait_num: 1 or 2
        scale, rho, beta: Weibull parameters for this trait (lifelines convention)
        pairs: pre-extracted relationship pair indices
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
