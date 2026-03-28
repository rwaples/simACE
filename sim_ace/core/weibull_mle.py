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

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
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

