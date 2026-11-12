"""Numba kernels for the phenotype models.

Kept apart from :mod:`simace.phenotype.hazards` and the model classes so that
building a model (config load, ``simace ls``) does not import numba. Callers
import this module inside the function that runs a kernel.
"""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from simace.core.numerics import ndtri as ndtri_approx

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange


# ---------------------------------------------------------------------------
# Numba kernels — fuse frailty computation + inversion in a single pass
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _nb_weibull(neg_log_u, liability, mean, scaled_beta, scale, inv_rho):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        val = scale * np.exp(np.log(neg_log_u[i] / z) * inv_rho)
        t[i] = min(max(val, 1e-10), 1e6)
    return t


@njit(parallel=True, cache=True)
def _nb_exponential(neg_log_u, liability, mean, scaled_beta, inv_rate):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        val = neg_log_u[i] * inv_rate / z
        t[i] = min(max(val, 1e-10), 1e6)
    return t


@njit(parallel=True, cache=True)
def _nb_gompertz(neg_log_u, liability, mean, scaled_beta, g_over_b, inv_g):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        target = neg_log_u[i] / z
        val = np.log1p(target * g_over_b) * inv_g
        t[i] = min(max(val, 1e-10), 1e6)
    return t


@njit(parallel=True, cache=True)
def _nb_lognormal(neg_log_u, liability, mean, scaled_beta, mu, sigma):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        target = neg_log_u[i] / z
        surv = np.exp(-target)
        if surv <= 0.0:
            t[i] = 1e6
        elif surv >= 1.0:
            # target == 0 means S(t) == 1, i.e. an event time of 0, and
            # ndtri_approx diverges to NaN at exactly 1.0.  Mirror the
            # surv <= 0 guard above and return the documented floor.
            t[i] = 1e-10
        else:
            val = np.exp(mu - sigma * ndtri_approx(surv))
            t[i] = min(max(val, 1e-10), 1e6)
    return t


@njit(parallel=True, cache=True)
def _nb_loglogistic(neg_log_u, liability, mean, scaled_beta, alpha, inv_k):
    n = len(neg_log_u)
    t = np.empty(n)
    for i in prange(n):
        z = np.exp(scaled_beta * (liability[i] - mean))
        target = neg_log_u[i] / z
        val = alpha * np.exp(np.log(np.expm1(target)) * inv_k)
        t[i] = min(max(val, 1e-10), 1e6)
    return t


# ---------------------------------------------------------------------------
# Numba kernels — fused y0 + Michael-Schucany-Haas inverse-Gaussian sampler
# ---------------------------------------------------------------------------


@njit(cache=True)
def _msh_sample(normal, uniform, mu, lam):
    """Michael-Schucany-Haas (1976) inverse Gaussian sampler, clamped to [1e-10, 1e6]."""
    y = normal * normal
    mu2 = mu * mu
    half_mu_over_lam = mu / (2.0 * lam)
    x = mu + half_mu_over_lam * (mu * y - np.sqrt(4.0 * mu * lam * y + mu2 * y * y))
    if uniform <= mu / (mu + x):
        return min(max(x, 1e-10), 1e6)
    return min(max(mu2 / x, 1e-10), 1e6)


@njit(parallel=True, cache=True)
def _nb_fpt(normals, uniforms, liability, mean, scaled_beta, sex, beta_sex, y0_base, inv_drift):
    """Fused FPT kernel for drift < 0 (everyone hits)."""
    n = len(normals)
    t = np.empty(n)
    for i in prange(n):
        y0 = y0_base * np.exp(-scaled_beta * (liability[i] - mean) - beta_sex * sex[i])
        if y0 < 1e-300:
            t[i] = 1e-10
        elif y0 > 1e150:
            t[i] = 1e6
        else:
            t[i] = _msh_sample(normals[i], uniforms[i], y0 * inv_drift, y0 * y0)
    return t


@njit(parallel=True, cache=True)
def _nb_fpt_cure(normals, uniforms, cure_draws, liability, mean, scaled_beta, sex, beta_sex, y0_base, drift, inv_drift):
    """Fused FPT kernel for drift > 0 (emergent cure fraction)."""
    n = len(normals)
    t = np.empty(n)
    for i in prange(n):
        y0 = y0_base * np.exp(-scaled_beta * (liability[i] - mean) - beta_sex * sex[i])
        if y0 < 1e-300:
            t[i] = 1e-10
        else:
            p_hit = np.exp(-2.0 * y0 * drift)
            if cure_draws[i] >= p_hit:
                t[i] = 1e6
            else:
                t[i] = _msh_sample(normals[i], uniforms[i], y0 * inv_drift, y0 * y0)
    return t
