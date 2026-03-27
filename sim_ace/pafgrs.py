"""PA-FGRS (Pearson-Aitken Family Genetic Risk Score) implementation.

Port of BioPsyk/PAFGRS R package for estimating individual genetic liability
from family pedigree data.  Given a pedigree, binary phenotype, and CIP table,
produces per-individual posterior mean and variance of genetic liability.

Reference: Krebs et al. (2024), AJHG.  PMID 39471805.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Truncated normal helpers  (exact port of BioPsyk/PAFGRS R/est_liab.R)
# ---------------------------------------------------------------------------


def _trunc_norm_below(mu: float, var: float, trunc: float) -> tuple[float, float]:
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


def _trunc_norm_above(mu: float, var: float, trunc: float) -> tuple[float, float]:
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


def _trunc_norm(mu: float, var: float, lower: float, upper: float) -> tuple[float, float]:
    """E[X] and Var[X] for X ~ N(mu, var) truncated to [lower, upper]."""
    if lower == upper:
        return (1e10 if np.isinf(lower) else lower), 0.0
    if np.isneginf(lower):
        return _trunc_norm_below(mu, var, upper)
    if np.isposinf(upper):
        return _trunc_norm_above(mu, var, lower)
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


def _trunc_norm_mixture(
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
        return _trunc_norm(mu, var, lower, upper)

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

    m0, v0 = _trunc_norm(mu, var, lower, upper)
    m1, v1 = _trunc_norm(mu, var, upper, np.inf)

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
    return _pa_fgrs_core(t1, t2, rel_w, covmat)


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
    return _pa_fgrs_core(t1, t2, w, covmat)


def _pa_fgrs_core(
    rel_t1: np.ndarray,
    rel_t2: np.ndarray,
    rel_w: np.ndarray,
    covmat: np.ndarray,
) -> tuple[float, float]:
    """Sequential Pearson-Aitken conditioning (internal)."""
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
        upd_m, upd_v = _trunc_norm_mixture(
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
# Kinship matrix  (generation-by-generation DP, mirrors kinship2::kinship)
# ---------------------------------------------------------------------------


def build_sparse_kinship(
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
    twins: np.ndarray | None = None,
) -> sp.csc_matrix:
    """Build a sparse kinship matrix from pedigree arrays.

    Uses the generation-by-generation dynamic programming algorithm
    (same as ``kinship2::kinship`` in R).  Founders (parent == -1) get
    self-kinship 0.5 and zero kinship with everyone else.

    Parameters
    ----------
    ids, mothers, fathers : int arrays (n,)
        Individual, mother, and father IDs.  -1 = missing/founder parent.
    twins : int array (n,) or None
        MZ twin partner ID (-1 if not a twin).

    Returns
    -------
    scipy.sparse.csc_matrix (n, n), symmetric.
    """
    t0 = time.perf_counter()
    n = len(ids)

    # Map IDs → contiguous 0-based indices
    max_id = int(ids.max())
    id_to_idx = np.full(max_id + 2, -1, dtype=np.int32)  # +2 for safety
    id_to_idx[ids] = np.arange(n, dtype=np.int32)

    def _remap(arr):
        out = np.full(len(arr), -1, dtype=np.int32)
        valid = (arr >= 0) & (arr <= max_id)
        out[valid] = id_to_idx[arr[valid]]
        return out

    m_idx = _remap(mothers)
    f_idx = _remap(fathers)
    tw_idx = _remap(twins) if twins is not None else np.full(n, -1, dtype=np.int32)

    # Compute generation depth from pedigree structure
    depth = _compute_depth(m_idx, f_idx, n)
    max_depth = int(depth.max())

    # Build kinship using dict-of-dicts: kin[i] = {j: value}
    # Full symmetric storage (kin[i][j] == kin[j][i]) for correct iteration
    kin: list[dict[int, float]] = [{i: 0.5} for i in range(n)]

    for d in range(1, max_depth + 1):
        gen_indices = np.where(depth == d)[0]
        for j in gen_indices:
            m, f = int(m_idx[j]), int(f_idx[j])

            # Self-kinship: (1 + kinship(m, f)) / 2
            km_f = 0.0
            if m >= 0 and f >= 0:
                km_f = kin[m].get(f, 0.0)
            kin[j][j] = (1.0 + km_f) / 2.0

            # Kinship(j, k) = (kinship(m, k) + kinship(f, k)) / 2
            # kin[parent] contains ALL of that parent's relatives (symmetric)
            m_row = kin[m] if m >= 0 else {}
            f_row = kin[f] if f >= 0 else {}

            all_k = set(m_row.keys()) | set(f_row.keys())
            all_k.discard(j)
            for k in all_k:
                mk = m_row.get(k, 0.0)
                fk = f_row.get(k, 0.0)
                val = (mk + fk) / 2.0
                if val > 1e-10:
                    kin[j][k] = val
                    kin[k][j] = val  # symmetric

        # Handle MZ twins in this generation
        for j in gen_indices:
            tw = int(tw_idx[j])
            if tw >= 0 and tw != j:
                self_kin = kin[j].get(j, 0.5)
                kin[j][tw] = self_kin
                kin[tw][j] = self_kin

    # Convert to COO then CSC
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_key, v in kin[i].items():
            if j_key >= i:  # upper triangle only (avoid duplicates)
                rows.append(i)
                cols.append(j_key)
                vals.append(v)
                if i != j_key:
                    rows.append(j_key)
                    cols.append(i)
                    vals.append(v)

    kmat = sp.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n),
    )
    elapsed = time.perf_counter() - t0
    nnz = len(vals)
    logger.info("Kinship matrix: %d individuals, %d nonzero entries, %.1fs", n, nnz, elapsed)
    return kmat


def _compute_depth(m_idx: np.ndarray, f_idx: np.ndarray, n: int) -> np.ndarray:
    """Compute generation depth: founders=0, children=max(parent_depth)+1."""
    depth = np.full(n, -1, dtype=np.int32)
    # Founders: both parents missing
    founders = (m_idx < 0) & (f_idx < 0)
    depth[founders] = 0

    changed = True
    while changed:
        changed = False
        unset = np.where(depth < 0)[0]
        for j in unset:
            m, f = int(m_idx[j]), int(f_idx[j])
            md = depth[m] if m >= 0 else 0
            fd = depth[f] if f >= 0 else 0
            if md >= 0 and fd >= 0:
                depth[j] = max(md, fd) + 1
                changed = True

    # Any remaining unset → treat as depth 0 (disconnected founders)
    depth[depth < 0] = 0
    return depth


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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-individual liability thresholds and w from CIP table.

    Parameters
    ----------
    affected : bool array (n,)
    t_observed : float array (n,) — age at onset (cases) or censoring (controls)
    cip_ages, cip_values : CIP lookup table
    lifetime_prevalence : K (population lifetime prevalence)

    Returns
    -------
    (thresholds, w) : float arrays (n,)
        thresholds = qnorm(1 - K)  (single lifetime threshold for all)
        w = CIP(age_i) / K  for controls, 1.0 for cases
    """
    n = len(affected)
    K = max(lifetime_prevalence, 1e-10)
    threshold = float(norm.ppf(1.0 - K))

    # Interpolate CIP at each individual's observed age
    cip_at_age = np.interp(t_observed, cip_ages, cip_values, left=0.0, right=cip_values[-1])

    w = np.where(affected, 1.0, np.clip(cip_at_age / K, 0.0, 1.0))
    thresholds = np.full(n, threshold)

    return thresholds, w


# ---------------------------------------------------------------------------
# Scoring pipeline
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
    t0 = time.perf_counter()
    n_ped = len(pedigree_df)
    n_pheno = len(phenotype_df)

    logger.info(
        "PA-FGRS scoring: %d probands, %d pedigree, h2=%.3f, ndegree=%d, trait=%d",
        n_pheno,
        n_ped,
        h2,
        ndegree,
        trait_num,
    )

    if kmat is None:
        kmat = build_sparse_kinship(
            pedigree_df["id"].values,
            pedigree_df["mother"].values,
            pedigree_df["father"].values,
            pedigree_df["twin"].values if "twin" in pedigree_df.columns else None,
        )

    ped_ids = pedigree_df["id"].values
    max_id = int(ped_ids.max())
    id_to_ped_idx = np.full(max_id + 2, -1, dtype=np.int32)
    id_to_ped_idx[ped_ids] = np.arange(n_ped, dtype=np.int32)

    pheno_ids = phenotype_df["id"].values
    pheno_ped_idx = id_to_ped_idx[pheno_ids]

    affected = phenotype_df[f"affected{trait_num}"].values.astype(bool)
    t_observed = phenotype_df[f"t_observed{trait_num}"].values
    true_A = phenotype_df[f"A{trait_num}"].values
    generation = phenotype_df["generation"].values

    thresholds, w = compute_thresholds_and_w(
        affected,
        t_observed,
        cip_ages,
        cip_values,
        lifetime_prevalence,
    )

    # Phenotype lookup arrays indexed by pedigree row
    pheno_lookup_aff = np.full(n_ped, False)
    pheno_lookup_thr = np.full(n_ped, np.nan)
    pheno_lookup_w = np.full(n_ped, 0.0)
    pheno_lookup_valid = np.zeros(n_ped, dtype=bool)

    valid_mask = pheno_ped_idx >= 0
    valid_pi = pheno_ped_idx[valid_mask]
    pheno_lookup_aff[valid_pi] = affected[valid_mask]
    pheno_lookup_thr[valid_pi] = thresholds[valid_mask]
    pheno_lookup_w[valid_pi] = w[valid_mask]
    pheno_lookup_valid[valid_pi] = True

    kin_threshold = 0.5 ** (ndegree + 1) - 1e-6

    est_arr = np.zeros(n_pheno)
    var_arr = np.full(n_pheno, h2)
    n_relatives_arr = np.zeros(n_pheno, dtype=np.int32)

    for i in range(n_pheno):
        pi = pheno_ped_idx[i]
        if pi < 0:
            continue

        # Extract column directly from CSC internals (zero-copy)
        start, end = kmat.indptr[pi], kmat.indptr[pi + 1]
        rel_ped_indices = kmat.indices[start:end]
        rel_kinships = kmat.data[start:end]

        # Filter: not self, above kinship threshold, has phenotype data
        mask = (rel_ped_indices != pi) & (rel_kinships >= kin_threshold) & pheno_lookup_valid[rel_ped_indices]
        rel_idx = rel_ped_indices[mask]
        rel_kin = rel_kinships[mask]

        n_rel = len(rel_idx)
        if n_rel == 0:
            est_arr[i] = 0.0
            var_arr[i] = h2
            continue

        n_relatives_arr[i] = n_rel

        # Build covariance matrix via batch sparse submatrix extraction
        # Row/col 0 = proband genetic liability (variance h2)
        # Rows/cols 1+ = relatives' total liability (variance 1)
        sub_kin = kmat[np.ix_(rel_idx, rel_idx)].toarray()
        covmat = np.zeros((1 + n_rel, 1 + n_rel))
        covmat[0, 0] = h2
        covmat[0, 1:] = 2.0 * rel_kin * h2
        covmat[1:, 0] = covmat[0, 1:]
        covmat[1:, 1:] = 2.0 * sub_kin * h2
        np.fill_diagonal(covmat[1:, 1:], 1.0)

        rel_aff = pheno_lookup_aff[rel_idx]
        rel_thr = pheno_lookup_thr[rel_idx]
        rel_w_vals = pheno_lookup_w[rel_idx]

        est, var = pa_fgrs(rel_aff, rel_thr, rel_w_vals, covmat)
        est_arr[i] = est
        var_arr[i] = var

    elapsed = time.perf_counter() - t0
    logger.info("PA-FGRS scoring complete: %.1fs (%.1f ms/proband)", elapsed, 1000 * elapsed / max(n_pheno, 1))

    return pd.DataFrame(
        {
            "id": pheno_ids,
            "est": est_arr,
            "var": var_arr,
            "true_A": true_A,
            "affected": affected,
            "generation": generation,
            "n_relatives": n_relatives_arr,
        }
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
        model = config.get(f"phenotype_model{trait_num}", "weibull")
        params = config.get(f"phenotype_params{trait_num}", {})
        if model == "weibull":
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

    from sim_ace.utils import save_parquet

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
