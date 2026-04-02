"""Bivariate PA-FGRS: joint scoring of two genetically correlated traits.

Extends univariate PA-FGRS to jointly estimate genetic liability for two
traits using Pearson-Aitken conditioning on a bivariate liability threshold
model.  Each relative contributes two observations (one per trait), and the
posterior is a bivariate normal for the proband's genetic liabilities.

Reference: Krebs et al. (2024), AJHG.  PMID 39471805 (univariate method).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import norm

if TYPE_CHECKING:
    import scipy.sparse as sp

from sim_ace.core._numba_utils import (
    _bvn_cdf as _nb_bvn_cdf,
)
from sim_ace.core._numba_utils import (
    _ndtri_approx,
    _tetrachoric_core,
)
from sim_ace.core._numba_utils import (
    _norm_cdf as _nb_norm_cdf,
)
from sim_ace.core._numba_utils import (
    _norm_sf as _nb_norm_sf,
)

from .pafgrs import (
    _build_rel_kinship_arrays,
    _nb_csc_lookup,
    _nb_trunc_norm_mixture,
    _trunc_norm_mixture_py,
    build_kinship_from_pairs,
    compute_thresholds_and_w,
    extract_relatives,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Covariance matrix construction (Python, for testing / reference)
# ---------------------------------------------------------------------------


def build_bivariate_covmat(
    proband_kin: np.ndarray,
    rel_kin: np.ndarray,
    h2_1: float,
    h2_2: float,
    cov_g12: float,
    rho_within: float,
) -> np.ndarray:
    """Build bivariate covariance matrix for a proband + n relatives.

    Parameters
    ----------
    proband_kin : float array (n_rel,)
        kinship(proband, relative_i) for each relative.
    rel_kin : float array (n_rel, n_rel)
        kinship matrix among relatives (symmetric).
    h2_1, h2_2 : heritability for each trait.
    cov_g12 : cross-trait genetic covariance = rA * sqrt(h2_1 * h2_2).
    rho_within : within-person cross-trait liability correlation.
        Genetic-only: equals cov_g12.
        Genetic+C: cov_g12 + rC * sqrt(C1 * C2).

    Returns
    -------
    (2 + 2*n_rel, 2 + 2*n_rel) covariance matrix.
    Layout: [proband_t1, proband_t2, rel1_t1, rel1_t2, ...].
    """
    n_rel = len(proband_kin)
    sz = 2 + 2 * n_rel
    cm = np.zeros((sz, sz))

    # Proband genetic liability block
    cm[0, 0] = h2_1
    cm[1, 1] = h2_2
    cm[0, 1] = cm[1, 0] = cov_g12

    for r in range(n_rel):
        k = proband_kin[r]
        c_same1 = 2.0 * k * h2_1
        c_same2 = 2.0 * k * h2_2
        c_cross = 2.0 * k * cov_g12

        # Proband <-> relative r
        cm[0, 2 + 2 * r] = cm[2 + 2 * r, 0] = c_same1  # p_t1 <-> r_t1
        cm[0, 3 + 2 * r] = cm[3 + 2 * r, 0] = c_cross  # p_t1 <-> r_t2
        cm[1, 2 + 2 * r] = cm[2 + 2 * r, 1] = c_cross  # p_t2 <-> r_t1
        cm[1, 3 + 2 * r] = cm[3 + 2 * r, 1] = c_same2  # p_t2 <-> r_t2

        # Relative r self block
        cm[2 + 2 * r, 2 + 2 * r] = 1.0  # var(L1)
        cm[3 + 2 * r, 3 + 2 * r] = 1.0  # var(L2)
        cm[2 + 2 * r, 3 + 2 * r] = rho_within  # cov(L1, L2)
        cm[3 + 2 * r, 2 + 2 * r] = rho_within

        # Relative r <-> relative r2
        for r2 in range(r + 1, n_rel):
            k12 = rel_kin[r, r2]
            c12_t1 = 2.0 * k12 * h2_1
            c12_t2 = 2.0 * k12 * h2_2
            c12_cross = 2.0 * k12 * cov_g12

            cm[2 + 2 * r, 2 + 2 * r2] = cm[2 + 2 * r2, 2 + 2 * r] = c12_t1
            cm[3 + 2 * r, 3 + 2 * r2] = cm[3 + 2 * r2, 3 + 2 * r] = c12_t2
            cm[2 + 2 * r, 3 + 2 * r2] = cm[3 + 2 * r2, 2 + 2 * r] = c12_cross
            cm[3 + 2 * r, 2 + 2 * r2] = cm[2 + 2 * r2, 3 + 2 * r] = c12_cross

    return cm


# ---------------------------------------------------------------------------
# Python fallback (reference implementation for testing)
# ---------------------------------------------------------------------------


def pa_fgrs_bivariate(
    rel_status1: np.ndarray,
    rel_thr1: np.ndarray,
    rel_w1: np.ndarray,
    rel_status2: np.ndarray,
    rel_thr2: np.ndarray,
    rel_w2: np.ndarray,
    covmat: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Bivariate PA-FGRS: posterior of proband genetic liability for two traits.

    Parameters
    ----------
    rel_status1/2 : bool arrays (n_rel,) — affected for each trait.
    rel_thr1/2 : float arrays (n_rel,) — liability threshold per trait.
    rel_w1/2 : float arrays (n_rel,) — proportion of lifetime risk observed.
    covmat : float array (2+2*n_rel, 2+2*n_rel)
        Layout: [proband_t1, proband_t2, rel1_t1, rel1_t2, ...].

    Returns
    -------
    (est_1, est_2, var_1, var_2, cov_12)
    """
    rel_status1 = np.asarray(rel_status1, dtype=bool)
    rel_status2 = np.asarray(rel_status2, dtype=bool)
    n_rel = len(rel_status1)

    # Build observation arrays (interleaved: rel1_t1, rel1_t2, rel2_t1, ...)
    n_obs = 2 * n_rel
    obs_lower = np.empty(n_obs)
    obs_upper = np.empty(n_obs)
    obs_w = np.empty(n_obs)

    for r in range(n_rel):
        # Trait 1
        if rel_status1[r]:
            obs_lower[2 * r] = rel_thr1[r]
            obs_upper[2 * r] = np.inf
        else:
            obs_lower[2 * r] = -np.inf
            obs_upper[2 * r] = rel_thr1[r]
        obs_w[2 * r] = rel_w1[r]

        # Trait 2
        if rel_status2[r]:
            obs_lower[2 * r + 1] = rel_thr2[r]
            obs_upper[2 * r + 1] = np.inf
        else:
            obs_lower[2 * r + 1] = -np.inf
            obs_upper[2 * r + 1] = rel_thr2[r]
        obs_w[2 * r + 1] = rel_w2[r]

    return _pa_fgrs_bivariate_core_py(obs_lower, obs_upper, obs_w, covmat)


def _pa_fgrs_bivariate_core_py(
    obs_lower: np.ndarray,
    obs_upper: np.ndarray,
    obs_w: np.ndarray,
    covmat: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Bivariate Pearson-Aitken conditioning (Python fallback)."""
    valid = (np.isfinite(obs_lower) | np.isfinite(obs_upper)) & (obs_w > 0)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return 0.0, 0.0, float(covmat[0, 0]), float(covmat[1, 1]), float(covmat[0, 1])

    obs_lower = obs_lower[valid]
    obs_upper = obs_upper[valid]
    obs_w = obs_w[valid]

    # Map valid observation indices to covmat indices (+2 for proband block)
    obs_cm_idx = np.where(valid)[0] + 2
    keep = np.concatenate([[0, 1], obs_cm_idx])
    cm = covmat[np.ix_(keep, keep)].copy()

    # Sort by informativeness: most informative first (conditioned last)
    cov_with_proband = np.abs(cm[2:, 0]) + np.abs(cm[2:, 1])
    total_cov = np.abs(cm[2:]).sum(axis=1)
    order = np.lexsort((-total_cov, -cov_with_proband, -obs_w))
    obs_lower = obs_lower[order]
    obs_upper = obs_upper[order]
    obs_w = obs_w[order]
    reorder = np.concatenate([[0, 1], order + 2])
    cm = cm[np.ix_(reorder, reorder)]

    mu = np.zeros(cm.shape[0])
    cov = cm

    for _ in range(n_valid):
        j = cov.shape[0] - 1
        ri = j - 2  # obs index (proband occupies 0, 1)

        kp = obs_w[ri] * norm.sf(obs_upper[ri])
        upd_m, upd_v = _trunc_norm_mixture_py(
            mu[j],
            cov[j, j],
            obs_lower[ri],
            obs_upper[ri],
            kp,
        )

        c_j = cov[:j, j].copy()
        inv_vj = 1.0 / cov[j, j] if cov[j, j] > 1e-30 else 0.0

        mu[:j] += c_j * inv_vj * (upd_m - mu[j])
        cov[:j, :j] -= np.outer(c_j, c_j) * (inv_vj - inv_vj * upd_v * inv_vj)

        mu = mu[:j]
        cov = cov[:j, :j]

    return (
        float(mu[0]),
        float(mu[1]),
        float(max(cov[0, 0], 0.0)),
        float(max(cov[1, 1], 0.0)),
        float(cov[0, 1]),
    )


# ---------------------------------------------------------------------------
# Numba bivariate PA-FGRS kernel
# ---------------------------------------------------------------------------


@njit(cache=True)
def _nb_pa_fgrs_bivariate_single(
    n_rel: int,
    rel_aff1: np.ndarray,
    rel_thr1: np.ndarray,
    rel_w1: np.ndarray,
    rel_aff2: np.ndarray,
    rel_thr2: np.ndarray,
    rel_w2: np.ndarray,
    covmat: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Score one proband using bivariate PA-FGRS.

    covmat layout: [proband_t1, proband_t2, rel1_t1, rel1_t2, ...].
    Returns (est_1, est_2, var_1, var_2, cov_12).
    """
    n_obs = 2 * n_rel

    # Build truncation bounds and weights for all observations
    obs_lower = np.empty(n_obs)
    obs_upper = np.empty(n_obs)
    obs_w = np.empty(n_obs)

    for r in range(n_rel):
        if rel_aff1[r]:
            obs_lower[2 * r] = rel_thr1[r]
            obs_upper[2 * r] = math.inf
        else:
            obs_lower[2 * r] = -math.inf
            obs_upper[2 * r] = rel_thr1[r]
        obs_w[2 * r] = rel_w1[r]

        if rel_aff2[r]:
            obs_lower[2 * r + 1] = rel_thr2[r]
            obs_upper[2 * r + 1] = math.inf
        else:
            obs_lower[2 * r + 1] = -math.inf
            obs_upper[2 * r + 1] = rel_thr2[r]
        obs_w[2 * r + 1] = rel_w2[r]

    # Filter valid observations
    valid_idx = np.empty(n_obs, dtype=np.int32)
    n_valid = 0
    for i in range(n_obs):
        if obs_w[i] > 0:
            valid_idx[n_valid] = i
            n_valid += 1

    if n_valid == 0:
        return 0.0, 0.0, covmat[0, 0], covmat[1, 1], covmat[0, 1]

    # Extract valid data
    vl = np.empty(n_valid)
    vu = np.empty(n_valid)
    vw = np.empty(n_valid)
    for i in range(n_valid):
        vl[i] = obs_lower[valid_idx[i]]
        vu[i] = obs_upper[valid_idx[i]]
        vw[i] = obs_w[valid_idx[i]]

    # Build subset covmat: 2 proband dims + n_valid observation dims
    sz = 2 + n_valid
    cm = np.empty((sz, sz))
    # Proband block
    cm[0, 0] = covmat[0, 0]
    cm[0, 1] = covmat[0, 1]
    cm[1, 0] = covmat[1, 0]
    cm[1, 1] = covmat[1, 1]
    # Proband-observation cross
    for i in range(n_valid):
        oi = valid_idx[i] + 2
        cm[0, 2 + i] = covmat[0, oi]
        cm[2 + i, 0] = covmat[0, oi]
        cm[1, 2 + i] = covmat[1, oi]
        cm[2 + i, 1] = covmat[1, oi]
    # Observation-observation
    for i in range(n_valid):
        oi = valid_idx[i] + 2
        cm[2 + i, 2 + i] = covmat[oi, oi]
        for j in range(i + 1, n_valid):
            oj = valid_idx[j] + 2
            cm[2 + i, 2 + j] = covmat[oi, oj]
            cm[2 + j, 2 + i] = covmat[oi, oj]

    # Sort by informativeness
    sort_key = np.empty(n_valid)
    for i in range(n_valid):
        sort_key[i] = vw[i] * 1e12 + (abs(cm[2 + i, 0]) + abs(cm[2 + i, 1])) * 1e6
    order = np.argsort(-sort_key)

    # Reorder
    sl = np.empty(n_valid)
    su = np.empty(n_valid)
    sw = np.empty(n_valid)
    for i in range(n_valid):
        sl[i] = vl[order[i]]
        su[i] = vu[order[i]]
        sw[i] = vw[order[i]]
    reorder = np.empty(sz, dtype=np.int32)
    reorder[0] = 0
    reorder[1] = 1
    for i in range(n_valid):
        reorder[2 + i] = 2 + order[i]
    scm = np.empty((sz, sz))
    for i in range(sz):
        for j in range(sz):
            scm[i, j] = cm[reorder[i], reorder[j]]

    # Pearson-Aitken conditioning
    mu = np.zeros(sz)
    cov = scm.copy()
    active = sz

    for _ in range(n_valid):
        j = active - 1
        ri = j - 2  # obs index

        kp = sw[ri] * _nb_norm_sf(su[ri])
        upd_m, upd_v = _nb_trunc_norm_mixture(mu[j], cov[j, j], sl[ri], su[ri], kp)

        inv_vj = 1.0 / cov[j, j] if cov[j, j] > 1e-30 else 0.0
        delta_m = upd_m - mu[j]
        factor = inv_vj - inv_vj * upd_v * inv_vj

        for a in range(j):
            mu[a] += cov[a, j] * inv_vj * delta_m
        for a in range(j):
            for b in range(j):
                cov[a, b] -= cov[a, j] * cov[b, j] * factor

        active -= 1

    return mu[0], mu[1], max(cov[0, 0], 0.0), max(cov[1, 1], 0.0), cov[0, 1]


# ---------------------------------------------------------------------------
# Numba batch kernel
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _nb_score_batch_bivariate(
    n_probands: int,
    rel_starts: np.ndarray,
    rel_flat_idx: np.ndarray,
    rel_flat_kin: np.ndarray,
    pheno_aff1: np.ndarray,
    pheno_thr1: np.ndarray,
    pheno_w1: np.ndarray,
    pheno_aff2: np.ndarray,
    pheno_thr2: np.ndarray,
    pheno_w2: np.ndarray,
    kmat_indptr: np.ndarray,
    kmat_indices: np.ndarray,
    kmat_data: np.ndarray,
    h2_1: float,
    h2_2: float,
    cov_g12: float,
    rho_within: float,
    est1_out: np.ndarray,
    est2_out: np.ndarray,
    var1_out: np.ndarray,
    var2_out: np.ndarray,
    cov12_out: np.ndarray,
    nrel_out: np.ndarray,
) -> None:
    """Score all probands in parallel using bivariate PA-FGRS."""
    for i in prange(n_probands):
        s = rel_starts[i]
        e = rel_starts[i + 1]
        n_rel = e - s
        if n_rel == 0:
            est1_out[i] = 0.0
            est2_out[i] = 0.0
            var1_out[i] = h2_1
            var2_out[i] = h2_2
            cov12_out[i] = cov_g12
            continue

        my_idx = rel_flat_idx[s:e]
        my_kin = rel_flat_kin[s:e]

        # Build bivariate covmat: 2(1+n_rel) x 2(1+n_rel)
        sz = 2 + 2 * n_rel
        covmat = np.empty((sz, sz))

        # Proband block
        covmat[0, 0] = h2_1
        covmat[1, 1] = h2_2
        covmat[0, 1] = cov_g12
        covmat[1, 0] = cov_g12

        for r in range(n_rel):
            c_same1 = 2.0 * my_kin[r] * h2_1
            c_same2 = 2.0 * my_kin[r] * h2_2
            c_cross = 2.0 * my_kin[r] * cov_g12

            covmat[0, 2 + 2 * r] = c_same1
            covmat[2 + 2 * r, 0] = c_same1
            covmat[0, 3 + 2 * r] = c_cross
            covmat[3 + 2 * r, 0] = c_cross

            covmat[1, 2 + 2 * r] = c_cross
            covmat[2 + 2 * r, 1] = c_cross
            covmat[1, 3 + 2 * r] = c_same2
            covmat[3 + 2 * r, 1] = c_same2

        # Relative-relative blocks
        for r1 in range(n_rel):
            covmat[2 + 2 * r1, 2 + 2 * r1] = 1.0
            covmat[3 + 2 * r1, 3 + 2 * r1] = 1.0
            covmat[2 + 2 * r1, 3 + 2 * r1] = rho_within
            covmat[3 + 2 * r1, 2 + 2 * r1] = rho_within

            ri = my_idx[r1]
            col_start = kmat_indptr[ri]
            col_end = kmat_indptr[ri + 1]

            for r2 in range(r1 + 1, n_rel):
                rj = my_idx[r2]
                k12 = _nb_csc_lookup(kmat_indices, kmat_data, col_start, col_end, rj)

                c12_t1 = 2.0 * k12 * h2_1
                c12_t2 = 2.0 * k12 * h2_2
                c12_x = 2.0 * k12 * cov_g12

                covmat[2 + 2 * r1, 2 + 2 * r2] = c12_t1
                covmat[2 + 2 * r2, 2 + 2 * r1] = c12_t1
                covmat[3 + 2 * r1, 3 + 2 * r2] = c12_t2
                covmat[3 + 2 * r2, 3 + 2 * r1] = c12_t2
                covmat[2 + 2 * r1, 3 + 2 * r2] = c12_x
                covmat[3 + 2 * r2, 2 + 2 * r1] = c12_x
                covmat[3 + 2 * r1, 2 + 2 * r2] = c12_x
                covmat[2 + 2 * r2, 3 + 2 * r1] = c12_x

        # Gather phenotype data
        aff1 = np.empty(n_rel, dtype=np.bool_)
        thr1 = np.empty(n_rel)
        w1 = np.empty(n_rel)
        aff2 = np.empty(n_rel, dtype=np.bool_)
        thr2 = np.empty(n_rel)
        w2 = np.empty(n_rel)

        for r in range(n_rel):
            idx = my_idx[r]
            aff1[r] = pheno_aff1[idx]
            thr1[r] = pheno_thr1[idx]
            w1[r] = pheno_w1[idx]
            aff2[r] = pheno_aff2[idx]
            thr2[r] = pheno_thr2[idx]
            w2[r] = pheno_w2[idx]

        e1, e2, v1, v2, c12 = _nb_pa_fgrs_bivariate_single(
            n_rel,
            aff1,
            thr1,
            w1,
            aff2,
            thr2,
            w2,
            covmat,
        )
        est1_out[i] = e1
        est2_out[i] = e2
        var1_out[i] = v1
        var2_out[i] = v2
        cov12_out[i] = c12
        nrel_out[i] = n_rel


# ---------------------------------------------------------------------------
# Pre-extracted kinship and prep/score split
# ---------------------------------------------------------------------------


@dataclass
class BivariatePrepData:
    """Pre-computed data shared across all bivariate scoring variants."""

    # Identity / metadata
    pheno_ids: np.ndarray
    pheno_ped_idx: np.ndarray
    generation: np.ndarray
    true_A1: np.ndarray
    true_A2: np.ndarray

    # Ragged relative structure
    rel_starts: np.ndarray
    rel_flat_idx: np.ndarray
    rel_flat_kin: np.ndarray
    rel_counts: np.ndarray

    # Pre-extracted dense relative-relative kinship (upper triangle)
    rel_kin_starts: np.ndarray
    rel_kin_flat: np.ndarray

    # Invariant phenotype data
    affected1: np.ndarray
    affected2: np.ndarray
    t_observed1: np.ndarray
    t_observed2: np.ndarray

    # Pedigree-indexed lookups (invariant across variants)
    n_ped: int
    pheno_lookup_valid: np.ndarray
    pheno_lookup_aff1: np.ndarray
    pheno_lookup_aff2: np.ndarray


def prepare_bivariate_scoring(
    pedigree_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    ndegree: int = 2,
    kmat: sp.csc_matrix | None = None,
) -> BivariatePrepData:
    """Prepare shared data for bivariate PA-FGRS scoring.

    Call once, then use ``score_bivariate_variant()`` for each parameter set.

    Parameters
    ----------
    pedigree_df : full pedigree
    phenotype_df : phenotyped subset
    ndegree : relationship degree cutoff
    kmat : pre-built kinship matrix (built if None)

    Returns
    -------
    BivariatePrepData with all shared structures pre-computed.
    """
    t0 = time.perf_counter()
    n_ped = len(pedigree_df)
    n_pheno = len(phenotype_df)

    if kmat is None:
        kmat = build_kinship_from_pairs(pedigree_df, ndegree=ndegree)

    # ID mapping
    ped_ids = pedigree_df["id"].values
    max_id = int(ped_ids.max())
    id_to_ped_idx = np.full(max_id + 2, -1, dtype=np.int32)
    id_to_ped_idx[ped_ids] = np.arange(n_ped, dtype=np.int32)

    pheno_ids = phenotype_df["id"].values
    pheno_ped_idx = id_to_ped_idx[pheno_ids]

    # Invariant phenotype data
    affected1 = phenotype_df["affected1"].values.astype(bool)
    affected2 = phenotype_df["affected2"].values.astype(bool)
    t_observed1 = phenotype_df["t_observed1"].values
    t_observed2 = phenotype_df["t_observed2"].values
    true_A1 = phenotype_df["A1"].values
    true_A2 = phenotype_df["A2"].values
    generation = phenotype_df["generation"].values

    # Pedigree-indexed lookups (invariant: affected status, valid mask)
    pheno_lookup_valid = np.zeros(n_ped, dtype=bool)
    pheno_lookup_aff1 = np.full(n_ped, False)
    pheno_lookup_aff2 = np.full(n_ped, False)

    valid_mask = pheno_ped_idx >= 0
    valid_pi = pheno_ped_idx[valid_mask]
    pheno_lookup_valid[valid_pi] = True
    pheno_lookup_aff1[valid_pi] = affected1[valid_mask]
    pheno_lookup_aff2[valid_pi] = affected2[valid_mask]

    # Extract relatives
    kin_threshold = 0.5 ** (ndegree + 1) - 1e-6
    rel_starts, rel_flat_idx, rel_flat_kin, rel_counts = extract_relatives(
        pheno_ped_idx,
        kmat,
        kin_threshold,
        pheno_lookup_valid,
        n_pheno,
    )

    rel_kin_starts, rel_kin_flat = _build_rel_kinship_arrays(
        rel_counts,
        rel_starts,
        rel_flat_idx,
        kmat,
        n_pheno,
    )

    elapsed = time.perf_counter() - t0
    n_scored = int((rel_counts > 0).sum())
    avg_rel = float(rel_counts[rel_counts > 0].mean()) if n_scored > 0 else 0
    kin_mb = rel_kin_flat.nbytes / 1e6
    logger.info(
        "Bivariate prep: %.2fs, %d probands, %d scored, avg %.0f relatives, %.1f MB kinship cache",
        elapsed,
        n_pheno,
        n_scored,
        avg_rel,
        kin_mb,
    )

    return BivariatePrepData(
        pheno_ids=pheno_ids,
        pheno_ped_idx=pheno_ped_idx,
        generation=generation,
        true_A1=true_A1,
        true_A2=true_A2,
        rel_starts=rel_starts,
        rel_flat_idx=rel_flat_idx,
        rel_flat_kin=rel_flat_kin,
        rel_counts=rel_counts,
        rel_kin_starts=rel_kin_starts,
        rel_kin_flat=rel_kin_flat,
        affected1=affected1,
        affected2=affected2,
        t_observed1=t_observed1,
        t_observed2=t_observed2,
        n_ped=n_ped,
        pheno_lookup_valid=pheno_lookup_valid,
        pheno_lookup_aff1=pheno_lookup_aff1,
        pheno_lookup_aff2=pheno_lookup_aff2,
    )


@njit(parallel=True, cache=True)
def _nb_score_batch_bivariate_prepped(
    n_probands: int,
    rel_starts: np.ndarray,
    rel_flat_idx: np.ndarray,
    rel_flat_kin: np.ndarray,
    kin_starts: np.ndarray,
    kin_flat: np.ndarray,
    pheno_aff1: np.ndarray,
    pheno_thr1: np.ndarray,
    pheno_w1: np.ndarray,
    pheno_aff2: np.ndarray,
    pheno_thr2: np.ndarray,
    pheno_w2: np.ndarray,
    h2_1: float,
    h2_2: float,
    cov_g12: float,
    rho_within: float,
    est1_out: np.ndarray,
    est2_out: np.ndarray,
    var1_out: np.ndarray,
    var2_out: np.ndarray,
    cov12_out: np.ndarray,
    nrel_out: np.ndarray,
) -> None:
    """Score all probands using pre-extracted dense kinship (no CSC lookups)."""
    for i in prange(n_probands):
        s = rel_starts[i]
        e = rel_starts[i + 1]
        n_rel = e - s
        if n_rel == 0:
            est1_out[i] = 0.0
            est2_out[i] = 0.0
            var1_out[i] = h2_1
            var2_out[i] = h2_2
            cov12_out[i] = cov_g12
            continue

        my_idx = rel_flat_idx[s:e]
        my_kin = rel_flat_kin[s:e]

        sz = 2 + 2 * n_rel
        covmat = np.empty((sz, sz))

        covmat[0, 0] = h2_1
        covmat[1, 1] = h2_2
        covmat[0, 1] = cov_g12
        covmat[1, 0] = cov_g12

        for r in range(n_rel):
            c_same1 = 2.0 * my_kin[r] * h2_1
            c_same2 = 2.0 * my_kin[r] * h2_2
            c_cross = 2.0 * my_kin[r] * cov_g12

            covmat[0, 2 + 2 * r] = c_same1
            covmat[2 + 2 * r, 0] = c_same1
            covmat[0, 3 + 2 * r] = c_cross
            covmat[3 + 2 * r, 0] = c_cross
            covmat[1, 2 + 2 * r] = c_cross
            covmat[2 + 2 * r, 1] = c_cross
            covmat[1, 3 + 2 * r] = c_same2
            covmat[3 + 2 * r, 1] = c_same2

        # Relative-relative blocks from pre-extracted upper-triangle kinship
        kin_offset = kin_starts[i]
        for r1 in range(n_rel):
            covmat[2 + 2 * r1, 2 + 2 * r1] = 1.0
            covmat[3 + 2 * r1, 3 + 2 * r1] = 1.0
            covmat[2 + 2 * r1, 3 + 2 * r1] = rho_within
            covmat[3 + 2 * r1, 2 + 2 * r1] = rho_within

            for r2 in range(r1 + 1, n_rel):
                # Upper-triangle flat index
                k_idx = r1 * n_rel - r1 * (r1 + 1) // 2 + (r2 - r1 - 1)
                k12 = kin_flat[kin_offset + k_idx]

                c12_t1 = 2.0 * k12 * h2_1
                c12_t2 = 2.0 * k12 * h2_2
                c12_x = 2.0 * k12 * cov_g12

                covmat[2 + 2 * r1, 2 + 2 * r2] = c12_t1
                covmat[2 + 2 * r2, 2 + 2 * r1] = c12_t1
                covmat[3 + 2 * r1, 3 + 2 * r2] = c12_t2
                covmat[3 + 2 * r2, 3 + 2 * r1] = c12_t2
                covmat[2 + 2 * r1, 3 + 2 * r2] = c12_x
                covmat[3 + 2 * r2, 2 + 2 * r1] = c12_x
                covmat[3 + 2 * r1, 2 + 2 * r2] = c12_x
                covmat[2 + 2 * r2, 3 + 2 * r1] = c12_x

        aff1 = np.empty(n_rel, dtype=np.bool_)
        thr1 = np.empty(n_rel)
        w1 = np.empty(n_rel)
        aff2 = np.empty(n_rel, dtype=np.bool_)
        thr2 = np.empty(n_rel)
        w2 = np.empty(n_rel)

        for r in range(n_rel):
            idx = my_idx[r]
            aff1[r] = pheno_aff1[idx]
            thr1[r] = pheno_thr1[idx]
            w1[r] = pheno_w1[idx]
            aff2[r] = pheno_aff2[idx]
            thr2[r] = pheno_thr2[idx]
            w2[r] = pheno_w2[idx]

        e1, e2, v1, v2, c12 = _nb_pa_fgrs_bivariate_single(
            n_rel,
            aff1,
            thr1,
            w1,
            aff2,
            thr2,
            w2,
            covmat,
        )
        est1_out[i] = e1
        est2_out[i] = e2
        var1_out[i] = v1
        var2_out[i] = v2
        cov12_out[i] = c12
        nrel_out[i] = n_rel


def score_bivariate_variant(
    prep: BivariatePrepData,
    h2_1: float,
    h2_2: float,
    cov_g12: float,
    rho_within: float,
    pheno_lookup_thr1: np.ndarray,
    pheno_lookup_w1: np.ndarray,
    pheno_lookup_thr2: np.ndarray,
    pheno_lookup_w2: np.ndarray,
) -> pd.DataFrame:
    """Score one parameter variant using pre-computed prep data.

    Parameters
    ----------
    prep : from ``prepare_bivariate_scoring()``
    h2_1, h2_2 : heritability per trait
    cov_g12 : cross-trait genetic covariance (rA * sqrt(h2_1 * h2_2))
    rho_within : within-person cross-trait liability correlation
    pheno_lookup_thr1/w1/thr2/w2 : pedigree-indexed threshold and w arrays
        (vary by CIP source)

    Returns
    -------
    DataFrame with est_1, est_2, var_1, var_2, cov_12, etc.
    """
    n_pheno = len(prep.pheno_ids)

    est1_arr = np.zeros(n_pheno)
    est2_arr = np.zeros(n_pheno)
    var1_arr = np.full(n_pheno, h2_1)
    var2_arr = np.full(n_pheno, h2_2)
    cov12_arr = np.full(n_pheno, cov_g12)
    n_relatives_arr = prep.rel_counts.copy()

    _nb_score_batch_bivariate_prepped(
        n_pheno,
        prep.rel_starts,
        prep.rel_flat_idx,
        prep.rel_flat_kin,
        prep.rel_kin_starts,
        prep.rel_kin_flat,
        prep.pheno_lookup_aff1,
        pheno_lookup_thr1,
        pheno_lookup_w1,
        prep.pheno_lookup_aff2,
        pheno_lookup_thr2,
        pheno_lookup_w2,
        h2_1,
        h2_2,
        cov_g12,
        rho_within,
        est1_arr,
        est2_arr,
        var1_arr,
        var2_arr,
        cov12_arr,
        n_relatives_arr,
    )

    return pd.DataFrame(
        {
            "id": prep.pheno_ids,
            "est_1": est1_arr,
            "est_2": est2_arr,
            "var_1": var1_arr,
            "var_2": var2_arr,
            "cov_12": cov12_arr,
            "true_A1": prep.true_A1,
            "true_A2": prep.true_A2,
            "affected1": prep.affected1,
            "affected2": prep.affected2,
            "generation": prep.generation,
            "n_relatives": n_relatives_arr,
        }
    )


def build_pheno_lookups(
    prep: BivariatePrepData,
    thresholds1: np.ndarray,
    w1: np.ndarray,
    thresholds2: np.ndarray,
    w2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pedigree-indexed threshold/w lookup arrays from per-phenotype arrays.

    Returns (pheno_lookup_thr1, pheno_lookup_w1, pheno_lookup_thr2, pheno_lookup_w2).
    """
    pheno_lookup_thr1 = np.full(prep.n_ped, np.nan)
    pheno_lookup_w1 = np.full(prep.n_ped, 0.0)
    pheno_lookup_thr2 = np.full(prep.n_ped, np.nan)
    pheno_lookup_w2 = np.full(prep.n_ped, 0.0)

    valid_mask = prep.pheno_ped_idx >= 0
    valid_pi = prep.pheno_ped_idx[valid_mask]
    pheno_lookup_thr1[valid_pi] = thresholds1[valid_mask]
    pheno_lookup_w1[valid_pi] = w1[valid_mask]
    pheno_lookup_thr2[valid_pi] = thresholds2[valid_mask]
    pheno_lookup_w2[valid_pi] = w2[valid_mask]

    return pheno_lookup_thr1, pheno_lookup_w1, pheno_lookup_thr2, pheno_lookup_w2


# ---------------------------------------------------------------------------
# Scoring pipeline (convenience wrapper)
# ---------------------------------------------------------------------------


def score_probands_bivariate(
    pedigree_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    h2_1: float,
    h2_2: float,
    rA: float,
    cip_ages1: np.ndarray,
    cip_values1: np.ndarray,
    lifetime_prev1: float,
    cip_ages2: np.ndarray,
    cip_values2: np.ndarray,
    lifetime_prev2: float,
    rho_within: float | None = None,
    ndegree: int = 2,
    kmat: sp.csc_matrix | None = None,
) -> pd.DataFrame:
    """Score all phenotyped individuals using bivariate PA-FGRS.

    Parameters
    ----------
    pedigree_df : full pedigree (all generations)
    phenotype_df : phenotyped subset (G_pheno generations)
    h2_1, h2_2 : heritability on the liability scale per trait
    rA : genetic correlation between traits
    cip_ages1/2, cip_values1/2 : CIP lookup tables per trait
    lifetime_prev1/2 : lifetime prevalence per trait
    rho_within : within-person cross-trait liability correlation.
        Default (None): genetic-only, rA * sqrt(h2_1 * h2_2).
        For genetic+C variant: rA*sqrt(h2_1*h2_2) + rC*sqrt(C1*C2).
    ndegree : max relationship degree to include
    kmat : pre-built kinship matrix (built from pedigree if None)

    Returns
    -------
    DataFrame with columns: id, est_1, est_2, var_1, var_2, cov_12,
    true_A1, true_A2, affected1, affected2, generation, n_relatives
    """
    cov_g12 = rA * math.sqrt(h2_1 * h2_2)
    if rho_within is None:
        rho_within = cov_g12

    prep = prepare_bivariate_scoring(pedigree_df, phenotype_df, ndegree, kmat)

    thresholds1, w1 = compute_thresholds_and_w(
        prep.affected1,
        prep.t_observed1,
        cip_ages1,
        cip_values1,
        lifetime_prev1,
    )
    thresholds2, w2 = compute_thresholds_and_w(
        prep.affected2,
        prep.t_observed2,
        cip_ages2,
        cip_values2,
        lifetime_prev2,
    )
    lookup_thr1, lookup_w1, lookup_thr2, lookup_w2 = build_pheno_lookups(
        prep,
        thresholds1,
        w1,
        thresholds2,
        w2,
    )

    return score_bivariate_variant(
        prep,
        h2_1,
        h2_2,
        cov_g12,
        rho_within,
        lookup_thr1,
        lookup_w1,
        lookup_thr2,
        lookup_w2,
    )


# ---------------------------------------------------------------------------
# Tetrachoric correlation and r_g estimation
# ---------------------------------------------------------------------------


def _bvn_cdf(x: float, y: float, rho: float) -> float:
    """P(X < x, Y < y) for standard bivariate normal with correlation rho."""
    return _nb_bvn_cdf(x, y, rho)


def compute_tetrachoric(table: np.ndarray) -> float:
    """Estimate tetrachoric correlation from a 2x2 contingency table.

    Uses the numba-jitted MLE from ``_numba_utils._tetrachoric_core``.

    Parameters
    ----------
    table : (2, 2) array of counts
        [[both_neg, row_neg_col_pos],
         [row_pos_col_neg, both_pos]]

    Returns
    -------
    Estimated tetrachoric correlation in [-1, 1].
    """
    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n == 0:
        return 0.0

    # Marginal proportions
    p = table / n
    p1 = p[1, 0] + p[1, 1]  # P(row variable = 1)
    p2 = p[0, 1] + p[1, 1]  # P(col variable = 1)

    if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        return 0.0

    n00 = float(table[0, 0])
    n01 = float(table[0, 1])
    n10 = float(table[1, 0])
    n11 = float(table[1, 1])

    t1 = float(_ndtri_approx(1.0 - p1))
    t2 = float(_ndtri_approx(1.0 - p2))
    phi_t1 = float(_nb_norm_cdf(t1))
    phi_t2 = float(_nb_norm_cdf(t2))

    r, _se = _tetrachoric_core(n11, n10, n01, n00, t1, t2, phi_t1, phi_t2)
    return float(r)


def estimate_rg(
    phenotype_df: pd.DataFrame,
    pedigree_df: pd.DataFrame,
    h2_1: float,
    h2_2: float,
) -> float:
    """Estimate genetic correlation rA from sibling concordance data.

    Uses Falconer-style cross-trait decomposition:
    1. Within-person cross-trait tetrachoric correlation (rho_within)
    2. Cross-sibling cross-trait tetrachoric correlation (rho_cross)
    3. rA = 2 * (rho_within - rho_cross) / sqrt(h2_1 * h2_2)

    Parameters
    ----------
    phenotype_df : DataFrame with affected1, affected2 columns
    pedigree_df : DataFrame with id, mother, father columns
    h2_1, h2_2 : estimated heritability for each trait

    Returns
    -------
    Estimated genetic correlation, clamped to [-1, 1].
    """
    from sim_ace.core.pedigree_graph import PedigreeGraph

    affected1 = phenotype_df["affected1"].values.astype(bool)
    affected2 = phenotype_df["affected2"].values.astype(bool)

    # Within-person cross-trait tetrachoric
    table_within = np.array(
        [
            [(~affected1 & ~affected2).sum(), (~affected1 & affected2).sum()],
            [(affected1 & ~affected2).sum(), (affected1 & affected2).sum()],
        ]
    )
    rho_within = compute_tetrachoric(table_within)

    # Cross-sibling cross-trait tetrachoric
    graph = PedigreeGraph(pedigree_df)
    pairs = graph.extract_pairs(max_degree=2)
    fs_pairs = pairs.get("FS", (np.array([], dtype=int), np.array([], dtype=int)))
    idx_a, idx_b = fs_pairs

    if len(idx_a) == 0:
        logger.warning("No full-sibling pairs found; returning rho_within as rA estimate")
        denom = math.sqrt(h2_1 * h2_2)
        return float(np.clip(rho_within / denom if denom > 1e-10 else 0.0, -1.0, 1.0))

    # Build id-to-phenotype-row mapping
    pheno_ids = phenotype_df["id"].values
    max_id = int(max(pheno_ids.max(), pedigree_df["id"].values.max()))
    id_to_pheno = np.full(max_id + 2, -1, dtype=np.int32)
    id_to_pheno[pheno_ids] = np.arange(len(phenotype_df), dtype=np.int32)

    # Map pair indices (pedigree-row indices) to phenotype rows
    ped_ids = pedigree_df["id"].values
    sib_a_ids = ped_ids[idx_a]
    sib_b_ids = ped_ids[idx_b]
    pa = id_to_pheno[sib_a_ids]
    pb = id_to_pheno[sib_b_ids]

    # Filter pairs where both siblings have phenotype data
    valid = (pa >= 0) & (pb >= 0)
    pa = pa[valid]
    pb = pb[valid]

    if len(pa) == 0:
        logger.warning("No phenotyped sibling pairs; returning rho_within as rA estimate")
        denom = math.sqrt(h2_1 * h2_2)
        return float(np.clip(rho_within / denom if denom > 1e-10 else 0.0, -1.0, 1.0))

    # Cross-sibling cross-trait: sib_a trait 1 vs sib_b trait 2
    a1 = affected1[pa]
    b2 = affected2[pb]
    table_cross = np.array(
        [
            [(~a1 & ~b2).sum(), (~a1 & b2).sum()],
            [(a1 & ~b2).sum(), (a1 & b2).sum()],
        ]
    )
    rho_cross = compute_tetrachoric(table_cross)

    # Falconer decomposition: rho_within - rho_cross = 0.5 * rA * sqrt(h2_1 * h2_2)
    denom = math.sqrt(h2_1 * h2_2)
    if denom < 1e-10:
        return 0.0

    rA = 2.0 * (rho_within - rho_cross) / denom
    rA = float(np.clip(rA, -1.0, 1.0))

    logger.info(
        "r_g estimation: rho_within=%.4f, rho_cross=%.4f, rA=%.4f (%d within-person, %d sibling pairs)",
        rho_within,
        rho_cross,
        rA,
        len(phenotype_df),
        len(pa),
    )
    return rA
