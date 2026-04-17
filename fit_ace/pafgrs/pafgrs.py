"""PA-FGRS (Pearson-Aitken Family Genetic Risk Score) implementation.

Port of BioPsyk/PAFGRS R package for estimating individual genetic liability
from family pedigree data.  Given a pedigree, binary phenotype, and CIP table,
produces per-individual posterior mean and variance of genetic liability.

Reference: Krebs et al. (2024), AJHG.  PMID 39471805.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit, prange
from scipy.stats import norm

from fit_ace.kinship.kinship import build_kinship_from_pairs, build_sparse_kinship
from sim_ace.core._numba_utils import (
    _ndtri_approx,
)
from sim_ace.core._numba_utils import (
    _norm_cdf as _nb_norm_cdf,
)
from sim_ace.core._numba_utils import (
    _norm_pdf as _nb_norm_pdf,
)
from sim_ace.core._numba_utils import (
    _norm_sf as _nb_norm_sf,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba-compiled truncated normal helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def _nb_trunc_norm_below(mu: float, var: float, trunc: float) -> tuple[float, float]:
    sd = math.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    beta = (trunc - mu) / sd
    phi_b = _nb_norm_pdf(beta)
    cdf_b = _nb_norm_cdf(beta)
    if cdf_b < 1e-15:
        return trunc, 0.0
    r = phi_b / cdf_b
    return mu - sd * r, max(var * (1 - beta * r - r * r), 0.0)


@njit(cache=True)
def _nb_trunc_norm_above(mu: float, var: float, trunc: float) -> tuple[float, float]:
    sd = math.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    alpha = (trunc - mu) / sd
    phi_a = _nb_norm_pdf(alpha)
    sf_a = _nb_norm_sf(alpha)
    if sf_a < 1e-15:
        return trunc, 0.0
    r = phi_a / sf_a
    return mu + sd * r, max(var * (1 + alpha * r - r * r), 0.0)


@njit(cache=True)
def _nb_trunc_norm(mu: float, var: float, lower: float, upper: float) -> tuple[float, float]:
    if lower == upper:
        return (1e10 if math.isinf(lower) else lower), 0.0
    if lower == -math.inf:
        return _nb_trunc_norm_below(mu, var, upper)
    if upper == math.inf:
        return _nb_trunc_norm_above(mu, var, lower)
    sd = math.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    Phi_diff = _nb_norm_cdf(b) - _nb_norm_cdf(a)
    if Phi_diff < 1e-15:
        return (lower + upper) / 2.0, 0.0
    phi_a, phi_b = _nb_norm_pdf(a), _nb_norm_pdf(b)
    ratio = (phi_b - phi_a) / Phi_diff
    m = mu - sd * ratio
    v = var * (1 - (b * phi_b - a * phi_a) / Phi_diff - ratio * ratio)
    return m, max(v, 0.0)


@njit(cache=True)
def _nb_trunc_norm_mixture(
    mu: float,
    var: float,
    lower: float,
    upper: float,
    kp: float,
) -> tuple[float, float]:
    if kp <= 0 or upper == math.inf:
        return _nb_trunc_norm(mu, var, lower, upper)

    sd = math.sqrt(var)
    cdf_u_cond = _nb_norm_cdf((upper - mu) / sd)
    sf_u_cond = 1.0 - cdf_u_cond
    sf_u_marg = _nb_norm_sf(upper)

    if sf_u_marg < 1e-15:
        w_below = 1.0
    else:
        denom = 1.0 - sf_u_cond * kp / sf_u_marg
        w_below = cdf_u_cond / denom if abs(denom) > 1e-15 else 1.0

    w_below = min(max(w_below, 0.0), 1.0)
    w_above = 1.0 - w_below

    m0, v0 = _nb_trunc_norm(mu, var, lower, upper)
    m1, v1 = _nb_trunc_norm(mu, var, upper, math.inf)

    new_mean = w_below * m0 + w_above * m1
    new_var = w_below * (m0 * m0 + v0) + w_above * (m1 * m1 + v1) - new_mean * new_mean
    return new_mean, max(new_var, 0.0)


# ---------------------------------------------------------------------------
# Numba single-proband PA-FGRS kernel
# ---------------------------------------------------------------------------


@njit(cache=True)
def _nb_pa_fgrs_single(
    n_rel: int,
    rel_aff: np.ndarray,
    rel_thr: np.ndarray,
    rel_w: np.ndarray,
    covmat: np.ndarray,
    h2: float,
) -> tuple[float, float]:
    """Score a single proband given pre-built covmat.

    covmat must be (1+n_rel, 1+n_rel) with proband at [0,0].
    """
    # Build truncation bounds from status
    t1 = np.empty(n_rel)
    t2 = np.empty(n_rel)
    for r in range(n_rel):
        if rel_aff[r]:
            t1[r] = rel_thr[r]
            t2[r] = math.inf
        else:
            t1[r] = -math.inf
            t2[r] = rel_thr[r]

    # Filter valid relatives
    valid_idx = np.empty(n_rel, dtype=np.int32)
    n_valid = 0
    for r in range(n_rel):
        if rel_w[r] > 0:
            valid_idx[n_valid] = r
            n_valid += 1

    if n_valid == 0:
        return 0.0, h2

    # Extract valid data into contiguous arrays
    vt1 = np.empty(n_valid)
    vt2 = np.empty(n_valid)
    vw = np.empty(n_valid)
    for i in range(n_valid):
        r = valid_idx[i]
        vt1[i] = t1[r]
        vt2[i] = t2[r]
        vw[i] = rel_w[r]

    # Build subset covmat: proband + valid relatives
    sz = 1 + n_valid
    cm = np.empty((sz, sz))
    cm[0, 0] = covmat[0, 0]
    for i in range(n_valid):
        r = valid_idx[i]
        cm[0, 1 + i] = covmat[0, 1 + r]
        cm[1 + i, 0] = covmat[0, 1 + r]
        for j in range(n_valid):
            rj = valid_idx[j]
            cm[1 + i, 1 + j] = covmat[1 + r, 1 + rj]

    # Sort by informativeness: composite key (descending w, then cov with proband)
    sort_key = np.empty(n_valid)
    for i in range(n_valid):
        sort_key[i] = vw[i] * 1e12 + cm[1 + i, 0] * 1e6
    order = np.argsort(-sort_key)

    # Reorder all arrays
    st1 = np.empty(n_valid)
    st2 = np.empty(n_valid)
    sw = np.empty(n_valid)
    for i in range(n_valid):
        st1[i] = vt1[order[i]]
        st2[i] = vt2[order[i]]
        sw[i] = vw[order[i]]
    reorder = np.empty(sz, dtype=np.int32)
    reorder[0] = 0
    for i in range(n_valid):
        reorder[1 + i] = 1 + order[i]
    scm = np.empty((sz, sz))
    for i in range(sz):
        for j in range(sz):
            scm[i, j] = cm[reorder[i], reorder[j]]

    # Pearson-Aitken conditioning with active_size counter
    mu = np.zeros(sz)
    cov = scm.copy()
    active = sz

    for _ in range(n_valid):
        j = active - 1
        ri = j - 1  # index into st1/st2/sw

        kp = sw[ri] * _nb_norm_sf(st2[ri])
        upd_m, upd_v = _nb_trunc_norm_mixture(mu[j], cov[j, j], st1[ri], st2[ri], kp)

        inv_vj = 1.0 / cov[j, j] if cov[j, j] > 1e-30 else 0.0
        delta_m = upd_m - mu[j]
        factor = inv_vj - inv_vj * upd_v * inv_vj

        for a in range(j):
            mu[a] += cov[a, j] * inv_vj * delta_m
        for a in range(j):
            for b in range(j):
                cov[a, b] -= cov[a, j] * cov[b, j] * factor

        active -= 1

    return mu[0], max(cov[0, 0], 0.0)


# ---------------------------------------------------------------------------
# Numba batch kernel with prange
# ---------------------------------------------------------------------------


@njit(cache=True)
def _nb_csc_lookup(indices: np.ndarray, data: np.ndarray, start: int, end: int, target: int) -> float:
    """Binary search for target row index in a CSC column segment."""
    lo, hi = start, end
    while lo < hi:
        mid = (lo + hi) // 2
        if indices[mid] < target:
            lo = mid + 1
        elif indices[mid] > target:
            hi = mid
        else:
            return data[mid]
    return 0.0


@njit(parallel=True, cache=True)
def _nb_score_batch(
    n_probands: int,
    rel_starts: np.ndarray,
    rel_flat_idx: np.ndarray,
    rel_flat_kin: np.ndarray,
    pheno_aff: np.ndarray,
    pheno_thr: np.ndarray,
    pheno_w: np.ndarray,
    kmat_indptr: np.ndarray,
    kmat_indices: np.ndarray,
    kmat_data: np.ndarray,
    h2: float,
    est_out: np.ndarray,
    var_out: np.ndarray,
    nrel_out: np.ndarray,
) -> None:
    """Score all probands in parallel."""
    for i in prange(n_probands):
        s = rel_starts[i]
        e = rel_starts[i + 1]
        n_rel = e - s
        if n_rel == 0:
            est_out[i] = 0.0
            var_out[i] = h2
            continue

        my_idx = rel_flat_idx[s:e]
        my_kin = rel_flat_kin[s:e]

        # Build covmat: [0,0]=h2, [0,1:]=2*kin*h2, [1:,1:]=2*sub_kin*h2 with diag=1
        sz = 1 + n_rel
        covmat = np.empty((sz, sz))
        covmat[0, 0] = h2
        for r in range(n_rel):
            c = 2.0 * my_kin[r] * h2
            covmat[0, 1 + r] = c
            covmat[1 + r, 0] = c

        # Relative-relative kinship via CSC lookup
        for r1 in range(n_rel):
            covmat[1 + r1, 1 + r1] = 1.0
            ri = my_idx[r1]
            col_start = kmat_indptr[ri]
            col_end = kmat_indptr[ri + 1]
            for r2 in range(r1 + 1, n_rel):
                rj = my_idx[r2]
                k12 = _nb_csc_lookup(kmat_indices, kmat_data, col_start, col_end, rj)
                c12 = 2.0 * k12 * h2
                covmat[1 + r1, 1 + r2] = c12
                covmat[1 + r2, 1 + r1] = c12

        # Gather phenotype data
        aff = np.empty(n_rel, dtype=np.bool_)
        thr = np.empty(n_rel)
        w = np.empty(n_rel)
        for r in range(n_rel):
            idx = my_idx[r]
            aff[r] = pheno_aff[idx]
            thr[r] = pheno_thr[idx]
            w[r] = pheno_w[idx]

        est_out[i], var_out[i] = _nb_pa_fgrs_single(n_rel, aff, thr, w, covmat, h2)
        nrel_out[i] = n_rel


# ---------------------------------------------------------------------------
# Truncated normal helpers  (exact port of BioPsyk/PAFGRS R/est_liab.R)
# ---------------------------------------------------------------------------


def _trunc_norm_below_py(mu: float, var: float, trunc: float) -> tuple[float, float]:
    """E[X] and Var[X] for X ~ N(mu, var) truncated to (-inf, trunc]."""
    sd = np.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    beta = (trunc - mu) / sd
    phi_b = norm.pdf(beta)
    cdf_b = norm.cdf(beta)
    if cdf_b < 1e-15:
        return trunc, 0.0
    r = phi_b / cdf_b
    return mu - sd * r, max(var * (1 - beta * r - r * r), 0.0)


def _trunc_norm_above_py(mu: float, var: float, trunc: float) -> tuple[float, float]:
    """E[X] and Var[X] for X ~ N(mu, var) truncated to [trunc, +inf)."""
    sd = np.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    alpha = (trunc - mu) / sd
    phi_a = norm.pdf(alpha)
    sf_a = norm.sf(alpha)
    if sf_a < 1e-15:
        return trunc, 0.0
    r = phi_a / sf_a
    return mu + sd * r, max(var * (1 + alpha * r - r * r), 0.0)


def _trunc_norm_py(mu: float, var: float, lower: float, upper: float) -> tuple[float, float]:
    """E[X] and Var[X] for X ~ N(mu, var) truncated to [lower, upper]."""
    if lower == upper:
        return (1e10 if np.isinf(lower) else lower), 0.0
    if np.isneginf(lower):
        return _trunc_norm_below_py(mu, var, upper)
    if np.isposinf(upper):
        return _trunc_norm_above_py(mu, var, lower)
    # Doubly truncated
    sd = np.sqrt(var)
    if sd < 1e-15:
        return mu, 0.0
    a = (lower - mu) / sd
    b = (upper - mu) / sd
    Phi_diff = norm.cdf(b) - norm.cdf(a)
    if Phi_diff < 1e-15:
        return (lower + upper) / 2.0, 0.0
    phi_a, phi_b = norm.pdf(a), norm.pdf(b)
    ratio = (phi_b - phi_a) / Phi_diff
    m = mu - sd * ratio
    v = var * (1 - (b * phi_b - a * phi_a) / Phi_diff - ratio * ratio)
    return m, max(v, 0.0)


def _trunc_norm_mixture_py(
    mu: float,
    var: float,
    lower: float,
    upper: float,
    kp: float,
) -> tuple[float, float]:
    """Mixture model moments for partially-observed controls.

    Two components weighted by conditional probability:
      - Below: N(mu, var) truncated to [lower, upper]  (truly unaffected)
      - Above: N(mu, var) truncated to [upper, +inf)   (future case)
    *kp* is the remaining lifetime risk: ``w * (1 - Phi(threshold))``.
    """
    if kp <= 0 or np.isposinf(upper):
        return _trunc_norm_py(mu, var, lower, upper)

    sd = np.sqrt(var)
    cdf_u_cond = norm.cdf(upper, loc=mu, scale=sd)
    sf_u_cond = 1.0 - cdf_u_cond
    sf_u_marg = norm.sf(upper)  # N(0,1) tail prob at threshold

    if sf_u_marg < 1e-15:
        w_below = 1.0
    else:
        denom = 1.0 - sf_u_cond * kp / sf_u_marg
        w_below = cdf_u_cond / denom if abs(denom) > 1e-15 else 1.0

    w_below = np.clip(w_below, 0.0, 1.0)
    w_above = 1.0 - w_below

    m0, v0 = _trunc_norm_py(mu, var, lower, upper)
    m1, v1 = _trunc_norm_py(mu, var, upper, np.inf)

    new_mean = w_below * m0 + w_above * m1
    new_var = w_below * (m0 * m0 + v0) + w_above * (m1 * m1 + v1) - new_mean * new_mean
    return new_mean, max(new_var, 0.0)


# ---------------------------------------------------------------------------
# PA-FGRS core algorithm
# ---------------------------------------------------------------------------


def pa_fgrs(
    rel_status: np.ndarray,
    rel_thr: np.ndarray,
    rel_w: np.ndarray,
    covmat: np.ndarray,
) -> tuple[float, float]:
    """PA-FGRS[mix]: posterior mean/variance of proband genetic liability.

    Parameters
    ----------
    rel_status : bool array (n_rel,)
        True = affected for each relative.
    rel_thr : float array (n_rel,)
        Liability threshold per relative (``qnorm(1 - prevalence)``).
    rel_w : float array (n_rel,)
        Proportion of lifetime risk observed (1.0 for cases).
    covmat : float array (1+n_rel, 1+n_rel)
        Row/col 0 = proband genetic liability (variance h2);
        rows/cols 1+ = relatives' total liability (variance 1).
        Off-diag = ``2 * kinship * h2``.

    Returns
    -------
    (est, var) : posterior mean and variance.
    """
    rel_status = np.asarray(rel_status, dtype=bool)
    rel_thr = np.asarray(rel_thr, dtype=np.float64)
    rel_w = np.asarray(rel_w, dtype=np.float64)
    covmat = np.asarray(covmat, dtype=np.float64)

    t1 = np.where(rel_status, rel_thr, -np.inf)
    t2 = np.where(rel_status, np.inf, rel_thr)
    return _pa_fgrs_core_py(t1, t2, rel_w, covmat)


def pa_fgrs_adt(
    rel_status: np.ndarray,
    rel_thr: np.ndarray,
    covmat: np.ndarray,
) -> tuple[float, float]:
    """PA-FGRS[adt]: affected / disease-threshold (no mixture model)."""
    rel_status = np.asarray(rel_status, dtype=bool)
    rel_thr = np.asarray(rel_thr, dtype=np.float64)
    covmat = np.asarray(covmat, dtype=np.float64)

    t1 = np.where(rel_status, rel_thr, -np.inf)
    t2 = rel_thr.copy()
    w = np.ones(len(rel_status))
    return _pa_fgrs_core_py(t1, t2, w, covmat)


def _pa_fgrs_core_py(
    rel_t1: np.ndarray,
    rel_t2: np.ndarray,
    rel_w: np.ndarray,
    covmat: np.ndarray,
) -> tuple[float, float]:
    """Sequential Pearson-Aitken conditioning (Python fallback)."""
    valid = (np.isfinite(rel_t1) | np.isfinite(rel_t2)) & (rel_w > 0)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return 0.0, float(covmat[0, 0])

    # Keep proband (index 0) + valid relatives
    rel_t1 = rel_t1[valid]
    rel_t2 = rel_t2[valid]
    rel_w = rel_w[valid]
    keep = np.concatenate([[0], np.where(valid)[0] + 1])
    cm = covmat[np.ix_(keep, keep)].copy()

    # Sort relatives by informativeness (most informative first in array;
    # algorithm conditions from the last row, i.e. least informative first).
    cov_with_proband = cm[1:, 0]
    total_cov = cm[1:].sum(axis=1)
    order = np.lexsort((-total_cov, -cov_with_proband, -rel_w))
    rel_t1 = rel_t1[order]
    rel_t2 = rel_t2[order]
    rel_w = rel_w[order]
    reorder = np.concatenate([[0], order + 1])
    cm = cm[np.ix_(reorder, reorder)]

    # Initialise posterior moments
    n = cm.shape[0]
    mu = np.zeros(n)
    cov = cm

    # Condition on each relative, from last (least informative) to first
    for _ in range(n_valid):
        j = cov.shape[0] - 1  # last row/col index
        ri = j - 1  # index into rel arrays (0-based)

        kp = rel_w[ri] * norm.sf(rel_t2[ri])
        upd_m, upd_v = _trunc_norm_mixture_py(
            mu[j],
            cov[j, j],
            rel_t1[ri],
            rel_t2[ri],
            kp,
        )

        # Pearson-Aitken update
        c_j = cov[:j, j].copy()
        inv_vj = 1.0 / cov[j, j] if cov[j, j] > 1e-30 else 0.0

        mu[:j] += c_j * inv_vj * (upd_m - mu[j])
        cov[:j, :j] -= np.outer(c_j, c_j) * (inv_vj - inv_vj * upd_v * inv_vj)

        # Shrink by removing row/col j
        mu = mu[:j]
        cov = cov[:j, :j]

    return float(mu[0]), float(max(cov[0, 0], 0.0))


# ---------------------------------------------------------------------------
# CIP computation
# ---------------------------------------------------------------------------


def compute_empirical_cip(
    phenotype_df: pd.DataFrame,
    trait_num: int,
    n_points: int = 200,
    max_age: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute empirical CIP from phenotype data (Kaplan-Meier-like).

    Returns (ages, cip_values, lifetime_prevalence).
    """
    affected = phenotype_df[f"affected{trait_num}"].values.astype(bool)
    t_obs = phenotype_df[f"t_observed{trait_num}"].values
    n = len(phenotype_df)

    ages = np.linspace(0, max_age, n_points)
    sorted_onset = np.sort(t_obs[affected])
    cip = np.searchsorted(sorted_onset, ages, side="right") / n

    lifetime_prev = float(affected.mean())
    return ages, cip, lifetime_prev


def compute_true_cip_weibull(
    scale: float,
    rho: float,
    beta: float,
    max_age: float = 80.0,
    n_points: int = 200,
    n_quad: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute true CIP for Weibull frailty model via Gauss-Hermite quadrature.

    CIP(t) = 1 - E[exp(-H0(t) * exp(beta * L))]  where  L ~ N(0,1)
    and  H0(t) = (t / scale)^rho.
    """
    from numpy.polynomial.hermite import hermgauss

    x, w = hermgauss(n_quad)
    liab = np.sqrt(2) * x  # liability points on N(0,1) scale
    wt = w / np.sqrt(np.pi)  # normalised weights

    ages = np.linspace(0, max_age, n_points)
    H0 = (ages / scale) ** rho  # cumulative hazard at each age

    # E[exp(-H0 * exp(beta * L))] via quadrature
    exponent = -H0[:, None] * np.exp(beta * liab[None, :])
    survival = (np.exp(exponent) * wt[None, :]).sum(axis=1)
    cip = 1.0 - survival
    return ages, cip


# ---------------------------------------------------------------------------
# Threshold and w computation
# ---------------------------------------------------------------------------


def compute_thresholds_and_w(
    affected: np.ndarray,
    t_observed: np.ndarray,
    cip_ages: np.ndarray,
    cip_values: np.ndarray,
    lifetime_prevalence: float,
    age_dependent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-individual liability thresholds and w from CIP table.

    Parameters
    ----------
    affected : bool array (n,)
    t_observed : float array (n,) — age at onset (cases) or censoring (controls)
    cip_ages, cip_values : CIP lookup table
    lifetime_prevalence : K (population lifetime prevalence)
    age_dependent : if True, each individual gets an age-specific threshold
        θ_i = Φ⁻¹(1 - CIP(t_i)) instead of the single lifetime threshold.

    Returns
    -------
    (thresholds, w) : float arrays (n,)
        thresholds = qnorm(1 - CIP(age_i)) if age_dependent, else qnorm(1 - K)
        w = CIP(age_i) / K  for controls, 1.0 for cases
    """
    n = len(affected)
    K = max(lifetime_prevalence, 1e-10)

    # Interpolate CIP at each individual's observed age
    cip_at_age = np.interp(t_observed, cip_ages, cip_values, left=0.0, right=cip_values[-1])

    w = np.where(affected, 1.0, np.clip(cip_at_age / K, 0.0, 1.0))

    _ndtri_vec = np.vectorize(_ndtri_approx)
    if age_dependent:
        # Per-individual threshold from age-specific CIP
        cip_clipped = np.clip(cip_at_age, 1e-15, 1.0 - 1e-15)
        thresholds = _ndtri_vec(1.0 - cip_clipped)
    else:
        threshold = float(_ndtri_approx(1.0 - K))
        thresholds = np.full(n, threshold)

    return thresholds, w


def compute_thresholds_and_w_by_sex(
    affected: np.ndarray,
    t_observed: np.ndarray,
    sex: np.ndarray,
    cip_ages_female: np.ndarray,
    cip_values_female: np.ndarray,
    lifetime_prev_female: float,
    cip_ages_male: np.ndarray,
    cip_values_male: np.ndarray,
    lifetime_prev_male: float,
    age_dependent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute thresholds and w using sex-stratified CIP tables.

    Parameters
    ----------
    sex : int array (n,) — 0=female, 1=male
    cip_ages_female/male, cip_values_female/male : per-sex CIP tables
    lifetime_prev_female/male : per-sex lifetime prevalence

    Returns
    -------
    (thresholds, w) : float arrays (n,)
    """
    n = len(affected)
    thresholds = np.empty(n)
    w = np.empty(n)

    is_male = sex == 1

    for mask, cip_a, cip_v, K in [
        (~is_male, cip_ages_female, cip_values_female, lifetime_prev_female),
        (is_male, cip_ages_male, cip_values_male, lifetime_prev_male),
    ]:
        if not mask.any():
            continue
        thr_sex, w_sex = compute_thresholds_and_w(
            affected[mask],
            t_observed[mask],
            cip_a,
            cip_v,
            K,
            age_dependent=age_dependent,
        )
        thresholds[mask] = thr_sex
        w[mask] = w_sex

    return thresholds, w


# ---------------------------------------------------------------------------
# Shared relative extraction
# ---------------------------------------------------------------------------


def extract_relatives(
    pheno_ped_idx: np.ndarray,
    kmat: sp.csc_matrix,
    kin_threshold: float,
    pheno_lookup_valid: np.ndarray,
    n_pheno: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract ragged relative arrays from kinship matrix.

    Vectorized extraction of relatives for each phenotyped individual,
    filtered by kinship threshold and phenotype availability.

    Returns
    -------
    (rel_starts, rel_flat_idx, rel_flat_kin, rel_counts)
        rel_starts : int32 array (n_pheno+1,) — CSR-style row pointers
        rel_flat_idx : int32 array — pedigree-row indices of relatives
        rel_flat_kin : float array — kinship values
        rel_counts : int32 array (n_pheno,) — number of relatives per proband
    """
    active_idx = np.where(pheno_ped_idx >= 0)[0].astype(np.int32)
    pi_arr = pheno_ped_idx[active_idx]
    n_active = len(active_idx)

    col_starts = kmat.indptr[pi_arr]
    col_ends = kmat.indptr[pi_arr + 1]
    col_lengths = (col_ends - col_starts).astype(np.int64)
    total_candidates = int(col_lengths.sum())

    active_labels = np.repeat(np.arange(n_active, dtype=np.int32), col_lengths)
    offsets = np.arange(total_candidates, dtype=np.int64)
    cum = np.empty(n_active + 1, dtype=np.int64)
    cum[0] = 0
    np.cumsum(col_lengths, out=cum[1:])
    offsets -= cum[active_labels]
    flat_pos = col_starts[active_labels] + offsets

    all_ri = kmat.indices[flat_pos]
    all_rk = kmat.data[flat_pos]
    all_pi = pi_arr[active_labels]

    filt = (all_ri != all_pi) & (all_rk >= kin_threshold) & pheno_lookup_valid[all_ri]
    filt_labels = active_labels[filt]
    filt_ri = all_ri[filt].astype(np.int32)
    filt_rk = all_rk[filt]

    active_counts = np.bincount(filt_labels, minlength=n_active).astype(np.int32)
    rel_counts = np.zeros(n_pheno, dtype=np.int32)
    rel_counts[active_idx] = active_counts

    rel_starts = np.zeros(n_pheno + 1, dtype=np.int32)
    np.cumsum(rel_counts, out=rel_starts[1:])

    return rel_starts, filt_ri, filt_rk, rel_counts


@njit(parallel=True, cache=True)
def _extract_rel_kinship(
    n_pheno: int,
    rel_starts: np.ndarray,
    rel_flat_idx: np.ndarray,
    kmat_indptr: np.ndarray,
    kmat_indices: np.ndarray,
    kmat_data: np.ndarray,
    kin_starts: np.ndarray,
    kin_flat: np.ndarray,
) -> None:
    """Pre-extract relative-relative kinship into flat upper-triangle arrays.

    For proband i with n_rel relatives, stores n_rel*(n_rel-1)/2 kinship
    values in upper-triangle order: pairs (0,1), (0,2), ..., (n_rel-2, n_rel-1).
    """
    for i in prange(n_pheno):
        s = rel_starts[i]
        e = rel_starts[i + 1]
        n_rel = e - s
        if n_rel < 2:
            continue
        offset = kin_starts[i]
        k = 0
        for r1 in range(n_rel):
            ri = rel_flat_idx[s + r1]
            col_start = kmat_indptr[ri]
            col_end = kmat_indptr[ri + 1]
            for r2 in range(r1 + 1, n_rel):
                rj = rel_flat_idx[s + r2]
                kin_flat[offset + k] = _nb_csc_lookup(kmat_indices, kmat_data, col_start, col_end, rj)
                k += 1


def _build_rel_kinship_arrays(
    rel_counts: np.ndarray,
    rel_starts: np.ndarray,
    rel_flat_idx: np.ndarray,
    kmat: sp.csc_matrix,
    n_pheno: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-extract dense upper-triangle relative kinship.

    Returns (kin_starts, kin_flat) arrays for use in prepped batch kernels.
    """
    tri_sizes = (rel_counts.astype(np.int64) * (rel_counts.astype(np.int64) - 1)) // 2
    tri_sizes = np.maximum(tri_sizes, 0)
    kin_starts = np.zeros(n_pheno + 1, dtype=np.int64)
    np.cumsum(tri_sizes, out=kin_starts[1:])
    total_kin = int(kin_starts[-1])
    kin_flat = np.zeros(total_kin, dtype=np.float64)

    _extract_rel_kinship(
        n_pheno,
        rel_starts,
        rel_flat_idx,
        kmat.indptr,
        kmat.indices,
        kmat.data,
        kin_starts,
        kin_flat,
    )
    return kin_starts, kin_flat


# ---------------------------------------------------------------------------
# Univariate prep/score split
# ---------------------------------------------------------------------------


@dataclass
class UnivariatePrepData:
    """Pre-computed data shared across all univariate scoring variants."""

    pheno_ids: np.ndarray
    pheno_ped_idx: np.ndarray
    generation: np.ndarray
    n_ped: int

    # Ragged relative structure
    rel_starts: np.ndarray
    rel_flat_idx: np.ndarray
    rel_flat_kin: np.ndarray
    rel_counts: np.ndarray

    # Pre-extracted dense relative-relative kinship (upper triangle)
    rel_kin_starts: np.ndarray
    rel_kin_flat: np.ndarray

    # Phenotype validity mask
    pheno_lookup_valid: np.ndarray


def prepare_univariate_scoring(
    pedigree_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    ndegree: int = 2,
    kmat: sp.csc_matrix | None = None,
) -> UnivariatePrepData:
    """Prepare shared data for univariate PA-FGRS scoring.

    Call once, then use ``score_univariate_variant()`` per trait/CIP/h2 combo.
    """
    t0 = time.perf_counter()
    n_ped = len(pedigree_df)
    n_pheno = len(phenotype_df)

    if kmat is None:
        kmat = build_kinship_from_pairs(pedigree_df, ndegree=ndegree)

    ped_ids = pedigree_df["id"].values
    max_id = int(ped_ids.max())
    id_to_ped_idx = np.full(max_id + 2, -1, dtype=np.int32)
    id_to_ped_idx[ped_ids] = np.arange(n_ped, dtype=np.int32)

    pheno_ids = phenotype_df["id"].values
    pheno_ped_idx = id_to_ped_idx[pheno_ids]
    generation = phenotype_df["generation"].values

    pheno_lookup_valid = np.zeros(n_ped, dtype=bool)
    valid_mask = pheno_ped_idx >= 0
    pheno_lookup_valid[pheno_ped_idx[valid_mask]] = True

    kin_threshold = 0.5 ** (ndegree + 1) - 1e-6
    rel_starts, rel_flat_idx, rel_flat_kin, rel_counts = extract_relatives(
        pheno_ped_idx,
        kmat,
        kin_threshold,
        pheno_lookup_valid,
        n_pheno,
    )

    kin_starts, kin_flat = _build_rel_kinship_arrays(
        rel_counts,
        rel_starts,
        rel_flat_idx,
        kmat,
        n_pheno,
    )

    elapsed = time.perf_counter() - t0
    n_scored = int((rel_counts > 0).sum())
    avg_rel = float(rel_counts[rel_counts > 0].mean()) if n_scored > 0 else 0
    logger.info(
        "Univariate prep: %.2fs, %d probands, %d scored, avg %.0f relatives, %.1f MB kinship cache",
        elapsed,
        n_pheno,
        n_scored,
        avg_rel,
        kin_flat.nbytes / 1e6,
    )

    return UnivariatePrepData(
        pheno_ids=pheno_ids,
        pheno_ped_idx=pheno_ped_idx,
        generation=generation,
        n_ped=n_ped,
        rel_starts=rel_starts,
        rel_flat_idx=rel_flat_idx,
        rel_flat_kin=rel_flat_kin,
        rel_counts=rel_counts,
        rel_kin_starts=kin_starts,
        rel_kin_flat=kin_flat,
        pheno_lookup_valid=pheno_lookup_valid,
    )


@njit(parallel=True, cache=True)
def _nb_score_batch_prepped(
    n_probands: int,
    rel_starts: np.ndarray,
    rel_flat_idx: np.ndarray,
    rel_flat_kin: np.ndarray,
    kin_starts: np.ndarray,
    kin_flat: np.ndarray,
    pheno_aff: np.ndarray,
    pheno_thr: np.ndarray,
    pheno_w: np.ndarray,
    h2: float,
    est_out: np.ndarray,
    var_out: np.ndarray,
    nrel_out: np.ndarray,
) -> None:
    """Score all probands using pre-extracted dense kinship (no CSC lookups)."""
    for i in prange(n_probands):
        s = rel_starts[i]
        e = rel_starts[i + 1]
        n_rel = e - s
        if n_rel == 0:
            est_out[i] = 0.0
            var_out[i] = h2
            continue

        my_idx = rel_flat_idx[s:e]
        my_kin = rel_flat_kin[s:e]

        sz = 1 + n_rel
        covmat = np.empty((sz, sz))
        covmat[0, 0] = h2
        for r in range(n_rel):
            c = 2.0 * my_kin[r] * h2
            covmat[0, 1 + r] = c
            covmat[1 + r, 0] = c

        kin_offset = kin_starts[i]
        for r1 in range(n_rel):
            covmat[1 + r1, 1 + r1] = 1.0
            for r2 in range(r1 + 1, n_rel):
                k_idx = r1 * n_rel - r1 * (r1 + 1) // 2 + (r2 - r1 - 1)
                k12 = kin_flat[kin_offset + k_idx]
                c12 = 2.0 * k12 * h2
                covmat[1 + r1, 1 + r2] = c12
                covmat[1 + r2, 1 + r1] = c12

        aff = np.empty(n_rel, dtype=np.bool_)
        thr = np.empty(n_rel)
        w = np.empty(n_rel)
        for r in range(n_rel):
            idx = my_idx[r]
            aff[r] = pheno_aff[idx]
            thr[r] = pheno_thr[idx]
            w[r] = pheno_w[idx]

        est_out[i], var_out[i] = _nb_pa_fgrs_single(n_rel, aff, thr, w, covmat, h2)
        nrel_out[i] = n_rel


def score_univariate_variant(
    prep: UnivariatePrepData,
    h2: float,
    trait_num: int,
    pheno_lookup_aff: np.ndarray,
    pheno_lookup_thr: np.ndarray,
    pheno_lookup_w: np.ndarray,
    phenotype_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score one parameter variant using pre-computed prep data.

    Parameters
    ----------
    prep : from ``prepare_univariate_scoring()``
    h2 : heritability on the liability scale
    trait_num : 1 or 2
    pheno_lookup_aff/thr/w : pedigree-indexed phenotype arrays
    phenotype_df : for true_A column

    Returns
    -------
    DataFrame with est, var, true_A, affected, generation, n_relatives.
    """
    n_pheno = len(prep.pheno_ids)

    est_arr = np.zeros(n_pheno)
    var_arr = np.full(n_pheno, h2)
    n_relatives_arr = prep.rel_counts.copy()

    _nb_score_batch_prepped(
        n_pheno,
        prep.rel_starts,
        prep.rel_flat_idx,
        prep.rel_flat_kin,
        prep.rel_kin_starts,
        prep.rel_kin_flat,
        pheno_lookup_aff,
        pheno_lookup_thr,
        pheno_lookup_w,
        h2,
        est_arr,
        var_arr,
        n_relatives_arr,
    )

    return pd.DataFrame(
        {
            "id": prep.pheno_ids,
            "est": est_arr,
            "var": var_arr,
            "true_A": phenotype_df[f"A{trait_num}"].values,
            "affected": phenotype_df[f"affected{trait_num}"].values.astype(bool),
            "generation": prep.generation,
            "n_relatives": n_relatives_arr,
        }
    )


def build_pheno_lookups_univariate(
    prep: UnivariatePrepData,
    affected: np.ndarray,
    thresholds: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pedigree-indexed aff/thr/w lookup arrays.

    Returns (pheno_lookup_aff, pheno_lookup_thr, pheno_lookup_w).
    """
    pheno_lookup_aff = np.full(prep.n_ped, False)
    pheno_lookup_thr = np.full(prep.n_ped, np.nan)
    pheno_lookup_w = np.full(prep.n_ped, 0.0)

    valid_mask = prep.pheno_ped_idx >= 0
    valid_pi = prep.pheno_ped_idx[valid_mask]
    pheno_lookup_aff[valid_pi] = affected[valid_mask]
    pheno_lookup_thr[valid_pi] = thresholds[valid_mask]
    pheno_lookup_w[valid_pi] = w[valid_mask]

    return pheno_lookup_aff, pheno_lookup_thr, pheno_lookup_w


# ---------------------------------------------------------------------------
# Scoring pipeline (convenience wrapper)
# ---------------------------------------------------------------------------


def score_probands(
    pedigree_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    h2: float,
    cip_ages: np.ndarray,
    cip_values: np.ndarray,
    lifetime_prevalence: float,
    trait_num: int = 1,
    ndegree: int = 2,
    kmat: sp.csc_matrix | None = None,
) -> pd.DataFrame:
    """Score all phenotyped individuals using PA-FGRS.

    Parameters
    ----------
    pedigree_df : full pedigree (all generations)
    phenotype_df : phenotyped subset (G_pheno generations)
    h2 : heritability on the liability scale
    cip_ages, cip_values : CIP lookup table
    lifetime_prevalence : K
    trait_num : 1 or 2
    ndegree : max relationship degree to include (kinship threshold)
    kmat : pre-built kinship matrix (built from pedigree if None)

    Returns
    -------
    DataFrame with columns: id, est, var, true_A, affected, generation
    """
    if kmat is None:
        kmat = build_sparse_kinship(
            pedigree_df["id"].values,
            pedigree_df["mother"].values,
            pedigree_df["father"].values,
            pedigree_df["twin"].values if "twin" in pedigree_df.columns else None,
        )

    prep = prepare_univariate_scoring(pedigree_df, phenotype_df, ndegree, kmat)

    affected = phenotype_df[f"affected{trait_num}"].values.astype(bool)
    t_observed = phenotype_df[f"t_observed{trait_num}"].values

    thresholds, w = compute_thresholds_and_w(
        affected,
        t_observed,
        cip_ages,
        cip_values,
        lifetime_prevalence,
    )
    lookup_aff, lookup_thr, lookup_w = build_pheno_lookups_univariate(
        prep,
        affected,
        thresholds,
        w,
    )

    return score_univariate_variant(
        prep,
        h2,
        trait_num,
        lookup_aff,
        lookup_thr,
        lookup_w,
        phenotype_df,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    pedigree_path: str,
    phenotype_path: str,
    output_scores_path: str,
    trait_num: int = 1,
    h2: float | None = None,
    cip_source: str = "empirical",
    ndegree: int = 2,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run PA-FGRS scoring end-to-end.

    Parameters
    ----------
    pedigree_path, phenotype_path : input files
    output_scores_path : where to write scores parquet
    trait_num : 1 or 2
    h2 : heritability (if None, read from config A{trait_num})
    cip_source : "empirical" or "true"
    ndegree : relationship degree cutoff
    config : scenario config dict (needed for true CIP and h2)

    Returns
    -------
    Scores DataFrame.
    """
    logger.info("Loading pedigree: %s", pedigree_path)
    pedigree_df = pd.read_parquet(pedigree_path)
    logger.info("Loading phenotype: %s", phenotype_path)
    phenotype_df = pd.read_parquet(phenotype_path)

    # Resolve h2
    if h2 is None and config is not None:
        h2 = float(config[f"A{trait_num}"])
    if h2 is None:
        raise ValueError("h2 must be provided or available in config")

    # Compute CIP
    censor_age = float(config.get("censor_age", 80)) if config else 80.0
    if cip_source == "empirical":
        cip_ages, cip_values, lifetime_prev = compute_empirical_cip(
            phenotype_df,
            trait_num,
            max_age=censor_age,
        )
    elif cip_source == "true":
        if config is None:
            raise ValueError("config required for true CIP")
        model = config.get(f"phenotype_model{trait_num}", "frailty")
        params = config.get(f"phenotype_params{trait_num}", {})
        if model == "frailty" and params.get("distribution") == "weibull":
            cip_ages, cip_values = compute_true_cip_weibull(
                scale=params["scale"],
                rho=params["rho"],
                beta=config[f"beta{trait_num}"],
                max_age=censor_age,
            )
        else:
            logger.warning("True CIP for model %s not implemented; falling back to empirical", model)
            cip_ages, cip_values, _ = compute_empirical_cip(
                phenotype_df,
                trait_num,
                max_age=censor_age,
            )
        lifetime_prev = float(config[f"prevalence{trait_num}"])
    else:
        raise ValueError(f"Unknown cip_source: {cip_source!r}")

    scores = score_probands(
        pedigree_df,
        phenotype_df,
        h2=h2,
        cip_ages=cip_ages,
        cip_values=cip_values,
        lifetime_prevalence=lifetime_prev,
        trait_num=trait_num,
        ndegree=ndegree,
    )

    from sim_ace.core.utils import save_parquet

    save_parquet(scores, output_scores_path)
    logger.info("Wrote PA-FGRS scores to %s", output_scores_path)
    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> None:
    """Command-line interface for PA-FGRS scoring."""
    from sim_ace import setup_logging

    parser = argparse.ArgumentParser(description="PA-FGRS genetic risk scoring")
    parser.add_argument("--pedigree", required=True, help="Path to pedigree.parquet")
    parser.add_argument("--phenotype", required=True, help="Path to phenotype.parquet")
    parser.add_argument("--output", required=True, help="Output scores parquet path")
    parser.add_argument("--trait", type=int, default=1, choices=[1, 2])
    parser.add_argument("--h2", type=float, default=None, help="Heritability (default: from config)")
    parser.add_argument("--cip-source", default="empirical", choices=["empirical", "true"])
    parser.add_argument("--ndegree", type=int, default=2)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--scenario", default=None, help="Scenario name in config")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    setup_logging(log_file=args.log_file)

    config = None
    if args.config:
        import yaml

        with open(args.config) as f:
            raw = yaml.safe_load(f)
        defaults = raw.get("defaults", {})
        if args.scenario and args.scenario in raw.get("scenarios", {}):
            defaults.update(raw["scenarios"][args.scenario])
        config = defaults

    main(
        pedigree_path=args.pedigree,
        phenotype_path=args.phenotype,
        output_scores_path=args.output,
        trait_num=args.trait,
        h2=args.h2,
        cip_source=args.cip_source,
        ndegree=args.ndegree,
        config=config,
    )


if __name__ == "__main__":
    cli()
