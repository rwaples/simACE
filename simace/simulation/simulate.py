"""ACE pedigree simulation.

Simulates multi-generational pedigrees with:
- A: Additive genetic component
- C: Common/shared environment component
- E: Unique environment component

Supports single-trait and two-trait (bivariate) modes with configurable
cross-trait correlations for genetic (rA) and common environment (rC) components.
"""

__all__ = [
    "add_to_pedigree",
    "allocate_offspring",
    "assign_twins",
    "balance_mating_slots",
    "draw_mating_counts",
    "generate_correlated_components",
    "generate_mendelian_noise",
    "mating",
    "pair_partners",
    "reproduce",
    "resolve_per_gen_param",
    "run_simulation",
]

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from simace.core._numba_utils import _ndtri_approx
from simace.core.parquet import save_parquet
from simace.core.schema import PEDIGREE
from simace.core.stage import stage

try:
    from numba import njit
except ImportError:
    njit = None

logger = logging.getLogger(__name__)


def resolve_per_gen_param(value: float | dict[int, float], G: int, name: str = "param") -> list[float]:
    """Resolve a variance-component parameter to a per-generation list.

    Args:
        value: scalar (constant across all generations) or dict mapping
               generation index → value.  Missing generation keys are
               forward-filled from the most recent earlier key.
        G: total number of generations to resolve for (indices 0..G-1).
        name: parameter name, used in error messages.

    Returns:
        List of length *G* with the resolved value for each generation.

    Raises:
        ValueError: if any resolved value is negative or if a dict has no
            key <= 0 (so generation 0 would be undefined).
    """
    if isinstance(value, (int, float)):
        v = float(value)
        if v < 0:
            raise ValueError(f"{name} must be >= 0, got {v}")
        return [v] * G

    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a scalar or dict, got {type(value).__name__}")

    if not value:
        raise ValueError(f"{name} dict must not be empty")

    # Sort keys and validate
    sorted_keys = sorted(value)
    if sorted_keys[0] > 0:
        raise ValueError(
            f"{name} dict must have a key <= 0 so generation 0 is defined; smallest key is {sorted_keys[0]}"
        )
    for k in sorted_keys:
        v = float(value[k])
        if v < 0:
            raise ValueError(f"{name}[{k}] must be >= 0, got {v}")

    # Forward-fill
    result = [0.0] * G
    key_idx = 0
    current_val = float(value[sorted_keys[0]])
    for gen in range(G):
        # Advance to the latest key <= gen
        while key_idx + 1 < len(sorted_keys) and sorted_keys[key_idx + 1] <= gen:
            key_idx += 1
            current_val = float(value[sorted_keys[key_idx]])
        result[gen] = current_val
    return result


def generate_correlated_components(
    rng: np.random.Generator, n: int, sd1: float, sd2: float, correlation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two correlated normal variables via multivariate normal.

    Args:
        rng: numpy random generator
        n: number of samples
        sd1: standard deviation for component 1
        sd2: standard deviation for component 2
        correlation: correlation between components

    Returns:
        (comp1, comp2): tuple of arrays, each shape (n,)

    Raises:
        ValueError: if sd1 or sd2 is negative, or correlation is outside [-1, 1]
    """
    if sd1 < 0 or sd2 < 0:
        raise ValueError(f"Standard deviations must be non-negative, got sd1={sd1}, sd2={sd2}")
    if not (-1 <= correlation <= 1):
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")

    cov = [
        [sd1**2, correlation * sd1 * sd2],
        [correlation * sd1 * sd2, sd2**2],
    ]
    samples = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    return samples[:, 0], samples[:, 1]


def generate_mendelian_noise(
    rng: np.random.Generator, n: int, sd_A1: float, sd_A2: float, rA: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated Mendelian sampling noise for two traits.

    Under the infinitesimal model, the Mendelian sampling variance is
    0.5 * Var(A) for each trait, so sd_noise = sd_A / sqrt(2)
    (Bulmer, 1971, Am. Nat., 105, 201-211).

    Args:
        rng: numpy random generator
        n: number of offspring
        sd_A1: standard deviation of A1 (sqrt of A1 variance)
        sd_A2: standard deviation of A2 (sqrt of A2 variance)
        rA: genetic correlation between traits

    Returns:
        (noise1, noise2): tuple of arrays, each shape (n,)
    """
    sd_noise1 = sd_A1 / np.sqrt(2)
    sd_noise2 = sd_A2 / np.sqrt(2)
    return generate_correlated_components(rng, n, sd_noise1, sd_noise2, rA)


def draw_mating_counts(rng: np.random.Generator, n: int, mating_lambda: float) -> np.ndarray:
    """Draw zero-truncated Poisson mating counts for *n* individuals.

    Args:
        rng: numpy random generator
        n: number of individuals
        mating_lambda: Poisson lambda for the ZTP distribution

    Returns:
        Array of shape ``(n,)`` with all values >= 1.
    """
    counts = rng.poisson(lam=mating_lambda, size=n)
    zeros = counts == 0
    while zeros.any():
        counts[zeros] = rng.poisson(lam=mating_lambda, size=zeros.sum())
        zeros = counts == 0
    return counts


def balance_mating_slots(
    rng: np.random.Generator, male_counts: np.ndarray, female_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Balance total mating slots between males and females.

    Takes ``T = min(sum(male), sum(female))`` and randomly trims the larger
    side so both sum to *T*.

    Returns:
        ``(balanced_male_counts, balanced_female_counts)``
    """
    m_total = int(male_counts.sum())
    f_total = int(female_counts.sum())
    T = min(m_total, f_total)

    def _trim(counts: np.ndarray, total: int, target: int) -> np.ndarray:
        if total == target:
            return counts.copy()
        # Expand to per-slot array, keep a random subset of size target
        idx = np.repeat(np.arange(len(counts)), counts)
        keep = rng.choice(len(idx), size=target, replace=False)
        keep.sort()
        kept_idx = idx[keep]
        return np.bincount(kept_idx, minlength=len(counts)).astype(counts.dtype)

    return _trim(male_counts, m_total, T), _trim(female_counts, f_total, T)


def _find_duplicate_pairs(matings: np.ndarray) -> np.ndarray:
    """Return boolean mask of duplicate (mother, father) pairs (keeps first occurrence).

    Uses vectorized numpy sort instead of Python set iteration.
    """
    M = len(matings)
    if M == 0:
        return np.zeros(0, dtype=np.bool_)
    max_id = max(int(matings[:, 0].max()), int(matings[:, 1].max())) + 1
    # int64 cast required: max_id² overflows int32 when IDs are int32
    keys = matings[:, 0].astype(np.int64) * max_id + matings[:, 1].astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    is_dup_sorted = np.empty(M, dtype=np.bool_)
    is_dup_sorted[0] = False
    is_dup_sorted[1:] = sorted_keys[1:] == sorted_keys[:-1]
    is_dup = np.empty(M, dtype=np.bool_)
    is_dup[order] = is_dup_sorted
    return is_dup


def _fast_rank(arr: np.ndarray) -> np.ndarray:
    """Rank values into (0, 1) via argsort. Suitable for continuous data (no ties)."""
    M = len(arr)
    order = np.argsort(arr)
    rank = np.empty(M, dtype=np.float64)
    rank[order] = np.arange(1, M + 1, dtype=np.float64)
    return rank / (M + 1)


def _quantile_normal_nb_python(arr):
    """Convert array to quantile-normal scores using Numba-compatible ndtri."""
    M = len(arr)
    order = np.argsort(arr)
    out = np.empty(M, dtype=np.float64)
    for i in range(M):
        p = float(i + 1) / (M + 1)
        out[order[i]] = _ndtri_approx(p)
    return out


if njit is not None:
    _quantile_normal_nb = njit(cache=True)(_quantile_normal_nb_python)
else:
    _quantile_normal_nb = _quantile_normal_nb_python


def _midparent_python(pheno_col, parents):
    """Compute midparent values without creating (N, 2) temporary."""
    n = len(parents)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = (pheno_col[parents[i, 0]] + pheno_col[parents[i, 1]]) * 0.5
    return out


if njit is not None:
    _midparent = njit(cache=True)(_midparent_python)
else:
    _midparent = _midparent_python


def pair_partners(
    rng: np.random.Generator,
    male_idxs: np.ndarray,
    male_counts: np.ndarray,
    female_idxs: np.ndarray,
    female_counts: np.ndarray,
) -> np.ndarray:
    """Create mating pairs via random bipartite matching.

    Expands each sex into slot arrays using ``np.repeat``, shuffles one side,
    and pairs positionally. Duplicate ``(mother, father)`` pairs are resolved
    by swapping conflicting entries.

    Returns:
        ``(M, 2)`` array of ``[mother_idx, father_idx]``.
    """
    male_slots = np.repeat(male_idxs, male_counts)
    female_slots = np.repeat(female_idxs, female_counts)
    rng.shuffle(male_slots)

    matings = np.column_stack([female_slots, male_slots])

    # Deduplicate: swap conflicting entries (rare at low lambda)
    for _attempt in range(5):
        is_dup = _find_duplicate_pairs(matings)
        if not is_dup.any():
            break
        dup_idxs = np.where(is_dup)[0]
        non_dup_idxs = np.where(~is_dup)[0]
        if len(non_dup_idxs) == 0:
            break
        for d in dup_idxs:
            swap_with = rng.choice(non_dup_idxs)
            matings[d, 1], matings[swap_with, 1] = matings[swap_with, 1], matings[d, 1]

    return matings


def _metropolis_sweep_python(
    f1_z, f2_z, m1_z, m2_z, male_perm, idx_i, idx_j, S1, S2, S12, S21, T1, T2, T12, T21, batch
):
    """Greedy Metropolis sweep: accept swaps that reduce squared error on four targets."""
    for k in range(batch):
        df = f1_z[idx_i[k]] - f1_z[idx_j[k]]
        df2 = f2_z[idx_i[k]] - f2_z[idx_j[k]]
        dm1 = m1_z[idx_j[k]] - m1_z[idx_i[k]]
        dm2 = m2_z[idx_j[k]] - m2_z[idx_i[k]]
        dk1 = df * dm1
        dk2 = df2 * dm2
        dk12 = df * dm2
        dk21 = df2 * dm1
        ne1 = S1 + dk1 - T1
        ne2 = S2 + dk2 - T2
        ne12 = S12 + dk12 - T12
        ne21 = S21 + dk21 - T21
        oe1 = S1 - T1
        oe2 = S2 - T2
        oe12 = S12 - T12
        oe21 = S21 - T21
        if ne1 * ne1 + ne2 * ne2 + ne12 * ne12 + ne21 * ne21 < oe1 * oe1 + oe2 * oe2 + oe12 * oe12 + oe21 * oe21:
            i = idx_i[k]
            j = idx_j[k]
            tmp = m1_z[i]
            m1_z[i] = m1_z[j]
            m1_z[j] = tmp
            tmp = m2_z[i]
            m2_z[i] = m2_z[j]
            m2_z[j] = tmp
            tmp_p = male_perm[i]
            male_perm[i] = male_perm[j]
            male_perm[j] = tmp_p
            S1 += dk1
            S2 += dk2
            S12 += dk12
            S21 += dk21
    return S1, S2, S12, S21


if njit is not None:
    _metropolis_sweep = njit(cache=True, fastmath=True)(_metropolis_sweep_python)
else:
    _metropolis_sweep = _metropolis_sweep_python


def _metropolis_full_python(
    fz,
    mz,
    male_perm,
    S1,
    S2,
    S12,
    S21,
    T1,
    T2,
    T12,
    T21,
    M,
    tol,
    max_proposals,
    seed,
):
    """Full Metropolis loop with direct random pair proposals, no Python transitions.

    Uses interleaved (M, 2) arrays for cache-friendly access: fz[:, 0] = f1_z,
    fz[:, 1] = f2_z, mz[:, 0] = m1_z, mz[:, 1] = m2_z.
    """
    np.random.seed(seed)
    indices = np.arange(M, dtype=np.int64)
    proposals_done = 0

    while proposals_done < max_proposals:
        r1_err = abs(S1 / M - T1 / M)
        r2_err = abs(S2 / M - T2 / M)
        r12_err = abs(S12 / M - T12 / M)
        r21_err = abs(S21 / M - T21 / M)
        if max(r1_err, max(r2_err, max(r12_err, r21_err))) < tol:
            break

        batch = min(M // 2, max_proposals - proposals_done)
        n_shuffle = min(2 * batch, M)

        # Partial Fisher-Yates: shuffle first n_shuffle elements
        for i in range(n_shuffle - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            indices[i], indices[j] = indices[j], indices[i]

        # Process proposals from shuffled pairs
        for _k in range(batch):
            ii = indices[2 * _k]
            jj = indices[2 * _k + 1]
            df = fz[ii, 0] - fz[jj, 0]
            df2 = fz[ii, 1] - fz[jj, 1]
            dm1 = mz[jj, 0] - mz[ii, 0]
            dm2 = mz[jj, 1] - mz[ii, 1]
            dk1 = df * dm1
            dk2 = df2 * dm2
            dk12 = df * dm2
            dk21 = df2 * dm1
            ne1 = S1 + dk1 - T1
            ne2 = S2 + dk2 - T2
            ne12 = S12 + dk12 - T12
            ne21 = S21 + dk21 - T21
            oe1 = S1 - T1
            oe2 = S2 - T2
            oe12 = S12 - T12
            oe21 = S21 - T21
            if ne1 * ne1 + ne2 * ne2 + ne12 * ne12 + ne21 * ne21 < oe1 * oe1 + oe2 * oe2 + oe12 * oe12 + oe21 * oe21:
                tmp0 = mz[ii, 0]
                tmp1 = mz[ii, 1]
                mz[ii, 0] = mz[jj, 0]
                mz[ii, 1] = mz[jj, 1]
                mz[jj, 0] = tmp0
                mz[jj, 1] = tmp1
                tmp_p = male_perm[ii]
                male_perm[ii] = male_perm[jj]
                male_perm[jj] = tmp_p
                S1 += dk1
                S2 += dk2
                S12 += dk12
                S21 += dk21

        proposals_done += batch

    return S1, S2, S12, S21, proposals_done


if njit is not None:
    _metropolis_full = njit(cache=True, fastmath=True)(_metropolis_full_python)
else:
    _metropolis_full = _metropolis_full_python


def _assortative_pair_partners(
    rng: np.random.Generator,
    male_idxs: np.ndarray,
    male_counts: np.ndarray,
    female_idxs: np.ndarray,
    female_counts: np.ndarray,
    pheno: np.ndarray,
    assort1: float,
    assort2: float,
    rho_w: float = 0.0,
    assort_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Create mating pairs with assortative mating.

    Single-trait case (one of assort1/assort2 zero): bivariate Gaussian copula
    on the active trait.

    Both-nonzero case: 4-variate Gaussian copula targeting Pearson mate
    correlations, following Border et al. (2022, Science) Eq. 2.
    Uses conditional-expectation initialization + Metropolis greedy swaps.

    Args:
        rng: numpy random generator
        male_idxs: indices of males in the parental population
        male_counts: balanced mating slot counts per male
        female_idxs: indices of females in the parental population
        female_counts: balanced mating slot counts per female
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2] for parents
        assort1: target mate correlation on trait 1 liability, in [-1, 1]
        assort2: target mate correlation on trait 2 liability, in [-1, 1]
        rho_w: within-person cross-trait liability correlation
        assort_matrix: optional full 2x2 mate correlation matrix R_mf.
            When provided, overrides the auto-computed off-diagonals.

    Returns:
        ``(M, 2)`` array of ``[mother_idx, father_idx]``.
    """
    # 1. Expand slots
    female_slots = np.repeat(female_idxs, female_counts)
    male_slots = np.repeat(male_idxs, male_counts)
    M = len(female_slots)

    # 2. Compute liability per slot
    liab1_f = pheno[female_slots, 0] + pheno[female_slots, 1] + pheno[female_slots, 2]
    liab2_f = pheno[female_slots, 3] + pheno[female_slots, 4] + pheno[female_slots, 5]
    liab1_m = pheno[male_slots, 0] + pheno[male_slots, 1] + pheno[male_slots, 2]
    liab2_m = pheno[male_slots, 3] + pheno[male_slots, 4] + pheno[male_slots, 5]

    if assort1 != 0 and assort2 != 0:
        # --- Both traits nonzero: 4-variate copula ---
        r1, r2 = assort1, assort2

        # Build full R_mf with cross-trait off-diagonals
        if assort_matrix is not None:
            R_mf = np.asarray(assort_matrix, dtype=np.float64)
        else:
            c = rho_w * np.sqrt(abs(r1 * r2)) * np.sign(r1 * r2)
            R_mf = np.array([[r1, c], [c, r2]])
        c_target = R_mf[0, 1]

        # Phase 1: Conditional-expectation initialization
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(_quantile_normal_nb, arr) for arr in [liab1_f, liab2_f, liab1_m, liab2_m]]
            qn_f1, qn_f2, qn_m1, qn_m2 = [f.result() for f in futs]

        R_ff = np.array([[1.0, rho_w], [rho_w, 1.0]])
        B = R_mf @ np.linalg.inv(R_ff)

        female_qn = np.column_stack([qn_f1, qn_f2])
        target_m = female_qn @ B.T

        _U, _s, Vt = np.linalg.svd(R_mf)
        v = Vt[0]  # dominant right singular vector

        male_qn = np.column_stack([qn_m1, qn_m2])
        target_proj = target_m @ v
        male_proj = male_qn @ v

        order_target = np.argsort(target_proj)
        order_male = np.argsort(male_proj)
        male_perm = np.empty(M, dtype=np.int64)
        male_perm[order_target] = order_male

        # Phase 2: Metropolis refinement on standardized liabilities
        def _zscore(arr):
            s = arr.std()
            return (arr - arr.mean()) / s if s > 1e-12 else np.zeros_like(arr)

        f1_z = _zscore(liab1_f)
        f2_z = _zscore(liab2_f)
        m1_all_z = _zscore(liab1_m)
        m2_all_z = _zscore(liab2_m)

        m1_z = m1_all_z[male_perm].copy()
        m2_z = m2_all_z[male_perm].copy()

        # Interleave for cache-friendly access in Metropolis inner loop
        fz = np.column_stack([f1_z, f2_z])
        mz = np.column_stack([m1_z, m2_z])

        T1 = r1 * M
        T2 = r2 * M
        T12 = c_target * M
        T21 = c_target * M
        S1 = float(fz[:, 0] @ mz[:, 0])
        S2 = float(fz[:, 1] @ mz[:, 1])
        S12 = float(fz[:, 0] @ mz[:, 1])
        S21 = float(fz[:, 1] @ mz[:, 0])

        tol = 5e-4
        max_proposals = 8 * M
        metro_seed = int(rng.integers(2**63))

        S1, S2, S12, S21, proposals_done = _metropolis_full(
            fz,
            mz,
            male_perm,
            S1,
            S2,
            S12,
            S21,
            T1,
            T2,
            T12,
            T21,
            M,
            tol,
            max_proposals,
            metro_seed,
        )

        logger.info(
            "Metropolis: %d/%d proposals (%.1f%%), M=%d, err1=%.5f, err2=%.5f, err12=%.5f, err21=%.5f",
            proposals_done,
            max_proposals,
            100.0 * proposals_done / max_proposals,
            M,
            abs(S1 / M - r1),
            abs(S2 / M - r2),
            abs(S12 / M - c_target),
            abs(S21 / M - c_target),
        )
        if proposals_done >= max_proposals:
            logger.warning(
                "Assortative mating Metropolis did not converge after %d proposals: "
                "err1=%.4f, err2=%.4f, err12=%.4f, err21=%.4f",
                max_proposals,
                S1 / M - r1,
                S2 / M - r2,
                S12 / M - c_target,
                S21 / M - c_target,
            )

        matings = np.column_stack([female_slots, male_slots[male_perm]])

    else:
        # --- Single-trait: bivariate Gaussian copula ---
        rank1_f = _fast_rank(liab1_f)
        rank2_f = _fast_rank(liab2_f)
        rank1_m = _fast_rank(liab1_m)
        rank2_m = _fast_rank(liab2_m)

        if assort1 < 0:
            rank1_m = 1.0 - rank1_m
        if assort2 < 0:
            rank2_m = 1.0 - rank2_m

        score_f = abs(assort1) * rank1_f + abs(assort2) * rank2_f
        score_m = abs(assort1) * rank1_m + abs(assort2) * rank2_m

        order_f = np.argsort(score_f)
        order_m = np.argsort(score_m)
        female_sorted = female_slots[order_f]
        male_sorted = male_slots[order_m]

        r_eff = min(np.sqrt(assort1**2 + assort2**2), 1.0)

        cov = [[1.0, r_eff], [r_eff, 1.0]]
        z = rng.multivariate_normal([0.0, 0.0], cov, size=M)

        rank_f = np.argsort(np.argsort(z[:, 0]))
        rank_m = np.argsort(np.argsort(z[:, 1]))

        matings = np.column_stack([female_sorted[rank_f], male_sorted[rank_m]])

    # Deduplicate: swap with female-proximity partner to preserve correlations
    t_dedup = time.perf_counter()
    n_dups_total = 0
    for _attempt in range(5):
        is_dup = _find_duplicate_pairs(matings)
        if not is_dup.any():
            break
        dup_idxs = np.where(is_dup)[0]
        n_dups_total += len(dup_idxs)
        non_dup_idxs = np.where(~is_dup)[0]
        if len(non_dup_idxs) == 0:
            break

        # KD-tree nearest-neighbor lookup: O(M log M) build + O(dups × log M) query
        nondup_liabs = np.column_stack([liab1_f[non_dup_idxs], liab2_f[non_dup_idxs]])
        tree = cKDTree(nondup_liabs)
        dup_liabs = np.column_stack([liab1_f[dup_idxs], liab2_f[dup_idxs]])
        _, nn_idx = tree.query(dup_liabs, k=1)

        for i, d in enumerate(dup_idxs):
            swap_with = non_dup_idxs[nn_idx[i]]
            matings[d, 1], matings[swap_with, 1] = matings[swap_with, 1], matings[d, 1]
    logger.debug(
        "Dedup: %d total dups resolved in %.1fs",
        n_dups_total,
        time.perf_counter() - t_dedup,
    )

    return matings


def allocate_offspring(rng: np.random.Generator, n_matings: int, N: int) -> np.ndarray:
    """Distribute *N* offspring across *n_matings* matings via multinomial.

    Returns:
        Array of shape ``(n_matings,)`` summing to exactly *N*.
    """
    probs = np.ones(n_matings) / n_matings
    return rng.multinomial(N, probs)


def assign_twins(rng: np.random.Generator, offspring_counts: np.ndarray, p_mztwin: float) -> np.ndarray:
    """Decide which matings produce an MZ twin pair.

    Only matings with >= 2 offspring are eligible. At most one twin pair
    per mating.

    Returns:
        Boolean mask of shape ``(M,)`` — True where a twin pair occurs.
    """
    mask = np.zeros(len(offspring_counts), dtype=bool)
    eligible = offspring_counts >= 2
    if eligible.any():
        rolls = rng.uniform(size=eligible.sum()) < p_mztwin
        mask[eligible] = rolls
    return mask


def mating(
    rng: np.random.Generator,
    parental_sex: np.ndarray,
    mating_lambda: float,
    p_mztwin: float,
    pheno: np.ndarray | None = None,
    assort1: float = 0.0,
    assort2: float = 0.0,
    rho_w: float = 0.0,
    assort_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate parent-offspring pairings via a modular mating pipeline.

    1. Separate males/females and draw ZTP(mating_lambda) mating counts.
    2. Balance slots so both sexes have equal total.
    3. Pair partners randomly (or assortatively if assort1/assort2 != 0).
    4. Allocate N offspring across matings via multinomial.
    5. Assign MZ twins to eligible matings.
    6. Build output arrays.

    Args:
        rng: numpy random generator
        parental_sex: array of sex values (0=female, 1=male) for parents
        mating_lambda: Poisson lambda for zero-truncated mating count distribution
        p_mztwin: probability of a mating producing MZ twins (if >= 2 offspring)
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2]; required when
            assort1 or assort2 is nonzero
        assort1: target mate correlation on trait 1 liability
        assort2: target mate correlation on trait 2 liability
        rho_w: within-person cross-trait liability correlation, used to
            derive off-diagonal entries of the mate correlation matrix
        assort_matrix: optional full 2x2 mate correlation matrix R_mf

    Returns:
        parent_idxs: (N, 2) array of [mother_idx, father_idx] for each offspring
        twins: (m, 2) array of [twin1_idx, twin2_idx] pairs for MZ twins
        household_ids: (N,) array mapping each offspring to a household index

    Raises:
        ValueError: if assort is nonzero but pheno is None
    """
    if (assort1 != 0 or assort2 != 0) and pheno is None:
        raise ValueError("pheno must be provided when assort1 or assort2 is nonzero")
    N = len(parental_sex)
    male_idxs = np.where(parental_sex == 1)[0]
    female_idxs = np.where(parental_sex == 0)[0]

    # 1. Draw mating counts per individual
    male_counts = draw_mating_counts(rng, len(male_idxs), mating_lambda)
    female_counts = draw_mating_counts(rng, len(female_idxs), mating_lambda)

    # 2. Balance slots
    male_counts, female_counts = balance_mating_slots(rng, male_counts, female_counts)

    # 3. Pair partners -> (M, 2) of [mother_idx, father_idx]
    if assort1 != 0 or assort2 != 0:
        matings = _assortative_pair_partners(
            rng,
            male_idxs,
            male_counts,
            female_idxs,
            female_counts,
            pheno,
            assort1,
            assort2,
            rho_w=rho_w,
            assort_matrix=assort_matrix,
        )
    else:
        matings = pair_partners(rng, male_idxs, male_counts, female_idxs, female_counts)
    M = len(matings)

    # 4. Allocate offspring
    offspring_counts = allocate_offspring(rng, M, N)

    # 5. Assign twins
    twin_mask = assign_twins(rng, offspring_counts, p_mztwin)

    # 6. Build output arrays
    parent_idxs = np.repeat(matings, offspring_counts, axis=0)

    # Household: all offspring of the same mother share a household
    _, household_ids = np.unique(parent_idxs[:, 0], return_inverse=True)

    # Twin pairs: first two offspring of each twin-flagged mating
    starts = np.zeros(M + 1, dtype=int)
    np.cumsum(offspring_counts, out=starts[1:])
    twin_indices = np.where(twin_mask)[0]
    if len(twin_indices) > 0:
        t1 = starts[twin_indices]
        t2 = t1 + 1
        twins = np.column_stack([t1, t2])
    else:
        twins = np.array([], dtype=int).reshape(0, 2)

    return parent_idxs, twins, household_ids


def reproduce(
    rng: np.random.Generator,
    pheno: np.ndarray,
    parents: np.ndarray,
    twins: np.ndarray,
    household_ids: np.ndarray,
    sd_A1: float,
    sd_E1: float,
    sd_C1: float,
    sd_A2: float,
    sd_E2: float,
    sd_C2: float,
    rA: float,
    rC: float,
    rE: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate offspring phenotypes from parents for two correlated traits.

    Additive genetic values are inherited as midparent + Mendelian noise.
    Common environment (C) is drawn freshly per household — it is NOT
    inherited from parents but represents the offspring's own shared rearing
    environment (siblings share C; parents and children do not). Unique
    environment (E) is drawn independently per individual.

    Args:
        rng: numpy random generator
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2] for parents
        parents: (n, 2) array of [mother_idx, father_idx]
        twins: array of MZ twin index pairs
        household_ids: (n,) array mapping each offspring to a household
        sd_A1: standard deviation of A for trait 1
        sd_E1: standard deviation of E for trait 1
        sd_C1: standard deviation of C for trait 1
        sd_A2: standard deviation of A for trait 2
        sd_E2: standard deviation of E for trait 2
        sd_C2: standard deviation of C for trait 2
        rA: genetic correlation between traits
        rC: common environment correlation between traits
        rE: unique environment correlation between traits

    Returns:
        offspring: (n, 6) array of [A1, C1, E1, A2, C2, E2]
        sex_offspring: (n,) array of sex values (0=female, 1=male)
    """
    n = len(parents)

    # Sex assignment
    sex_offspring = rng.binomial(size=n, n=1, p=0.5)

    # Additive genetic: midparent + correlated Mendelian noise
    mp1 = _midparent(pheno[:, 0], parents)  # A1 midparent
    mp2 = _midparent(pheno[:, 3], parents)  # A2 midparent

    noise1, noise2 = generate_mendelian_noise(rng, n, sd_A1, sd_A2, rA)
    a1_offspring = mp1 + noise1
    a2_offspring = mp2 + noise2

    # Common environment: freshly drawn per household each generation.
    # C is NOT inherited from parents -- it reflects the offspring's own
    # shared rearing environment. Siblings share C; parents and children do not.
    # This is the standard ACE model assumption (no autoregressive C transmission).
    # household_ids are already 0-based contiguous integers from mating()
    n_hh = int(household_ids.max()) + 1
    hh_c1, hh_c2 = generate_correlated_components(rng, n_hh, sd_C1, sd_C2, rC)
    c1_offspring = hh_c1[household_ids]
    c2_offspring = hh_c2[household_ids]

    # Unique environment: correlated via rE; independent when rE=0
    if rE != 0:
        e1_offspring, e2_offspring = generate_correlated_components(rng, n, sd_E1, sd_E2, rE)
    else:
        e1_offspring = rng.normal(size=n, loc=0, scale=sd_E1)
        e2_offspring = rng.normal(size=n, loc=0, scale=sd_E2)

    # MZ twins share A values and sex for both traits
    if len(twins) > 0:
        a1_offspring[twins[:, 1]] = a1_offspring[twins[:, 0]]
        a2_offspring[twins[:, 1]] = a2_offspring[twins[:, 0]]
        sex_offspring[twins[:, 1]] = sex_offspring[twins[:, 0]]

    offspring = np.stack(
        [
            a1_offspring,
            c1_offspring,
            e1_offspring,
            a2_offspring,
            c2_offspring,
            e2_offspring,
        ],
        axis=-1,
    )

    return offspring, sex_offspring


_PED_COLUMNS = [
    "id",
    "sex",
    "mother",
    "father",
    "twin",
    "generation",
    "household_id",
    "A1",
    "C1",
    "E1",
    "liability1",
    "A2",
    "C2",
    "E2",
    "liability2",
]


def _init_pedigree_arrays(total_rows: int) -> dict[str, np.ndarray]:
    """Pre-allocate numpy arrays for the pedigree.

    Dtype choices for memory efficiency at large N:
    - int32 for IDs (supports up to 2.1B individuals)
    - int8 for sex (0/1)
    - float32 for ACE variance components (~7 significant digits)
    - float64 for liabilities (full precision for phenotype models)
    """
    _dtypes = {
        "id": np.int32,
        "mother": np.int32,
        "father": np.int32,
        "twin": np.int32,
        "household_id": np.int32,
        "sex": np.int8,
        "generation": np.int32,
        "A1": np.float32,
        "C1": np.float32,
        "E1": np.float32,
        "A2": np.float32,
        "C2": np.float32,
        "E2": np.float32,
        "liability1": np.float64,
        "liability2": np.float64,
    }
    arrays: dict[str, np.ndarray] = {}
    for col, dtype in _dtypes.items():
        arrays[col] = np.empty(total_rows, dtype=dtype)
    arrays["twin"][:] = -1
    return arrays


def _fill_pedigree_slice(
    arrays: dict[str, np.ndarray],
    offset: int,
    pheno: np.ndarray,
    sex: np.ndarray,
    parents: np.ndarray,
    twins: np.ndarray,
    household_ids: np.ndarray,
    generation: int,
    is_founder: bool,
    id_offset: int,
    parent_offset: int,
    household_offset: int,
) -> None:
    """Fill a slice of pre-allocated pedigree arrays for one generation."""
    n = len(pheno)
    s = slice(offset, offset + n)

    arrays["id"][s] = np.arange(n, dtype=np.int32) + np.int32(id_offset)
    arrays["sex"][s] = sex
    arrays["generation"][s] = generation
    arrays["household_id"][s] = (household_ids + household_offset).astype(np.int32)

    if is_founder:
        arrays["mother"][s] = -1
        arrays["father"][s] = -1
    else:
        arrays["mother"][s] = (parents[:, 0] + parent_offset).astype(np.int32)
        arrays["father"][s] = (parents[:, 1] + parent_offset).astype(np.int32)

    # Twin column
    arrays["twin"][offset : offset + n] = -1
    if len(twins) > 0:
        twin_ids = (twins + np.int32(id_offset)).astype(np.int32)
        arrays["twin"][offset + twins[:, 0]] = twin_ids[:, 1]
        arrays["twin"][offset + twins[:, 1]] = twin_ids[:, 0]

    # ACE components and liabilities
    arrays["A1"][s] = pheno[:, 0]
    arrays["C1"][s] = pheno[:, 1]
    arrays["E1"][s] = pheno[:, 2]
    arrays["liability1"][s] = pheno[:, 0] + pheno[:, 1] + pheno[:, 2]
    arrays["A2"][s] = pheno[:, 3]
    arrays["C2"][s] = pheno[:, 4]
    arrays["E2"][s] = pheno[:, 5]
    arrays["liability2"][s] = pheno[:, 3] + pheno[:, 4] + pheno[:, 5]


def _arrays_to_dataframe(arrays: dict[str, np.ndarray], total_rows: int) -> pd.DataFrame:
    """Convert pre-allocated arrays to a pedigree DataFrame."""
    return pd.DataFrame({col: arrays[col][:total_rows] for col in _PED_COLUMNS})


def add_to_pedigree(
    pheno: np.ndarray,
    sex: np.ndarray,
    parents: np.ndarray,
    twins: np.ndarray,
    household_ids: np.ndarray,
    generation: int,
    pedigree: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a generation to the pedigree DataFrame (backward-compatible wrapper).

    The internal simulation loop uses pre-allocated arrays for performance.
    This function is retained for external callers and tests.
    """
    n = len(pheno)
    is_founder = pedigree is None
    id_offset = 0 if is_founder else len(pedigree)
    parent_offset = id_offset - n if not is_founder else 0
    household_offset = 0 if is_founder else int(pedigree["household_id"].iloc[-1]) + 1

    arrays = _init_pedigree_arrays(n)
    _fill_pedigree_slice(
        arrays,
        0,
        pheno,
        sex,
        parents,
        twins,
        household_ids,
        generation=generation,
        is_founder=is_founder,
        id_offset=id_offset,
        parent_offset=parent_offset,
        household_offset=household_offset,
    )
    df = _arrays_to_dataframe(arrays, n)
    if pedigree is not None:
        return pd.concat([pedigree, df], ignore_index=True)
    return df


@stage(reads=None, writes=PEDIGREE)
def run_simulation(
    *,
    seed: int,
    N: int,
    G_ped: int,
    mating_lambda: float,
    p_mztwin: float,
    A1: float,
    C1: float,
    A2: float,
    C2: float,
    rA: float,
    rC: float,
    rE: float = 0.0,
    E1: float | dict[int, float] | None = None,
    E2: float | dict[int, float] | None = None,
    G_sim: int | None = None,
    assort1: float = 0.0,
    assort2: float = 0.0,
    assort_matrix: list[list[float]] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Run the full ACE simulation for two correlated traits.

    Variance components A (additive genetic), C (shared environment), and
    E (unique environment) are specified as absolute variances.  A is constant
    across generations; C and E may be specified per-generation via a dict
    mapping generation index to value (forward-filled for missing keys).

    When E1/E2 are omitted, they default to ``1 - A - C`` for backward
    compatibility (total variance = 1).

    Args:
        seed: Random seed
        N: Population size per generation (positive integer)
        G_ped: Number of generations to record in pedigree (integer >= 1)
        mating_lambda: Poisson lambda for zero-truncated mating count distribution (> 0)
        p_mztwin: Probability of a mating producing MZ twins, in [0, 1)
        A1: Trait 1 additive genetic variance (>= 0).
        C1: Trait 1 shared-environment variance (>= 0).
        A2: Trait 2 additive genetic variance (>= 0).
        C2: Trait 2 shared-environment variance (>= 0).
        rA: Genetic correlation between traits, in [-1, 1]
        rC: Common environment correlation between traits, in [-1, 1]
        rE: Unique environment correlation between traits, in [-1, 1].
            Default 0 (independent E across traits).
        E1: Trait 1 unique-environment variance.  Scalar (constant) or dict
            mapping generation index → value (forward-filled).
        E2: Trait 2 unique-environment variance.  Same format as E1.
        G_sim: Total generations to simulate (default: G_ped). First G_sim - G_ped
               generations are burn-in and discarded from output.
        assort1: Target mate Pearson correlation on trait 1 liability, in [-1, 1].
        assort2: Target mate Pearson correlation on trait 2 liability, in [-1, 1].
        assort_matrix: Optional full 2x2 mate correlation matrix R_mf.
            Overrides assort1/assort2 diagonal with matrix diagonal.

    Returns:
        Pedigree DataFrame with columns id, sex, mother, father, twin,
        generation, household_id, A1, C1, E1, liability1, A2, C2, E2,
        liability2.

    Raises:
        ValueError: if any parameter is outside its valid range
    """
    if G_sim is None:
        G_sim = G_ped

    # --- Input validation ---
    for name, val in [("A1", A1), ("C1", C1), ("A2", A2), ("C2", C2)]:
        if not (isinstance(val, (int, float)) and val >= 0):
            raise ValueError(f"{name} must be a non-negative scalar, got {val}")

    # Default E to residual (1 - A - C) for backward compatibility
    if E1 is None:
        E1 = 1.0 - A1 - C1
    if E2 is None:
        E2 = 1.0 - A2 - C2

    if not (int(N) == N and N > 0):
        raise ValueError(f"N must be a positive integer, got {N}")
    if not (G_ped == int(G_ped) and G_ped >= 1):
        raise ValueError(f"G_ped must be an integer >= 1, got {G_ped}")
    if not (mating_lambda > 0):
        raise ValueError(f"mating_lambda must be > 0, got {mating_lambda}")
    if not (0 <= p_mztwin < 1):
        raise ValueError(f"p_mztwin must be in [0, 1), got {p_mztwin}")
    if not (-1 <= rA <= 1):
        raise ValueError(f"rA must be in [-1, 1], got {rA}")
    if not (-1 <= rC <= 1):
        raise ValueError(f"rC must be in [-1, 1], got {rC}")
    if not (-1 <= rE <= 1):
        raise ValueError(f"rE must be in [-1, 1], got {rE}")
    if not (-1 <= assort1 <= 1):
        raise ValueError(f"assort1 must be in [-1, 1], got {assort1}")
    if not (-1 <= assort2 <= 1):
        raise ValueError(f"assort2 must be in [-1, 1], got {assort2}")

    # Resolve assort_matrix
    R_mf = None
    if assort_matrix is not None:
        R_mf = np.asarray(assort_matrix, dtype=np.float64)
        if R_mf.shape != (2, 2):
            raise ValueError(f"assort_matrix must be 2x2, got shape {R_mf.shape}")
        if abs(R_mf[0, 1] - R_mf[1, 0]) > 1e-10:
            raise ValueError(f"assort_matrix must be symmetric: got [{R_mf[0, 1]}, {R_mf[1, 0]}]")
        assort1 = float(R_mf[0, 0])
        assort2 = float(R_mf[1, 1])
        if not (-1 <= assort1 <= 1):
            raise ValueError(f"assort_matrix[0,0] must be in [-1, 1], got {assort1}")
        if not (-1 <= assort2 <= 1):
            raise ValueError(f"assort_matrix[1,1] must be in [-1, 1], got {assort2}")

    if G_sim < G_ped:
        raise ValueError(f"G_sim ({G_sim}) must be >= G_ped ({G_ped})")

    logger.info("Starting simulation: N=%d, G_ped=%d, seed=%d", N, G_ped, seed)
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    # Resolve per-generation C and E variance components
    # C1/C2 may be scalar or per-gen dict; A is always scalar (constant)
    C1_per_gen = resolve_per_gen_param(C1, G_sim, name="C1")
    C2_per_gen = resolve_per_gen_param(C2, G_sim, name="C2")
    E1_per_gen = resolve_per_gen_param(E1, G_sim, name="E1")
    E2_per_gen = resolve_per_gen_param(E2, G_sim, name="E2")

    # Compute per-generation standard deviations
    sd_C1_per_gen = [np.sqrt(v) for v in C1_per_gen]
    sd_C2_per_gen = [np.sqrt(v) for v in C2_per_gen]
    sd_E1_per_gen = [np.sqrt(v) for v in E1_per_gen]
    sd_E2_per_gen = [np.sqrt(v) for v in E2_per_gen]

    # A is constant across generations
    sd_A1 = np.sqrt(A1)
    sd_A2 = np.sqrt(A2)

    # Within-person cross-trait liability correlation per C/E generation
    _rho_w_A = rA * np.sqrt(A1 * A2)
    rho_w_per_ce = [
        _rho_w_A + rC * np.sqrt(C1_per_gen[g] * C2_per_gen[g]) + rE * np.sqrt(E1_per_gen[g] * E2_per_gen[g])
        for g in range(G_sim)
    ]

    # Validate |rho_w| < 1 for all C/E generations
    if assort1 != 0 and assort2 != 0:
        for g, rw in enumerate(rho_w_per_ce):
            if abs(rw) >= 1.0 - 1e-10:
                raise ValueError(
                    f"Both-trait assortative mating requires |rho_w| < 1 "
                    f"(got rho_w={rw:.4f} at C/E generation {g}). "
                    f"Traits are perfectly correlated; "
                    f"use single-trait assortment instead."
                )

    # Track whether R_mf was provided explicitly (vs. auto-computed from rho_w)
    R_mf_user = R_mf

    # Validate PSD of full 4x4 Sigma for each generation's rho_w
    if R_mf_user is not None or (assort1 != 0 and assort2 != 0):
        for g, rw in enumerate(rho_w_per_ce):
            if R_mf_user is not None:
                R_mf_g = R_mf_user
            else:
                c = rw * np.sqrt(abs(assort1 * assort2)) * np.sign(assort1 * assort2)
                R_mf_g = np.array([[assort1, c], [c, assort2]])
            R_ff = np.array([[1.0, rw], [rw, 1.0]])
            Sigma_4 = np.block([[R_ff, R_mf_g.T], [R_mf_g, R_ff]])
            eigvals = np.linalg.eigvalsh(Sigma_4)
            if eigvals[0] < -1e-8:
                raise ValueError(
                    f"Full 4x4 mate correlation matrix Sigma_4 is not PSD "
                    f"(min eigenvalue = {eigvals[0]:.6f} at C/E generation {g}). "
                    f"Reduce the magnitude of assort_matrix off-diagonal entries."
                )

    # Initialize founders with correlated components (using gen-0 C/E variances)
    sex = rng.binomial(size=N, n=1, p=0.5)

    # A components: correlated via rA (constant across generations)
    a1, a2 = generate_correlated_components(rng, N, sd_A1, sd_A2, rA)

    # C components: correlated via rC (gen-0 variance)
    c1, c2 = generate_correlated_components(rng, N, sd_C1_per_gen[0], sd_C2_per_gen[0], rC)

    # E components: correlated via rE (gen-0 variance); independent when rE=0
    if rE != 0:
        e1, e2 = generate_correlated_components(rng, N, sd_E1_per_gen[0], sd_E2_per_gen[0], rE)
    else:
        e1 = rng.normal(size=N, loc=0, scale=sd_E1_per_gen[0])
        e2 = rng.normal(size=N, loc=0, scale=sd_E2_per_gen[0])

    pheno = np.stack([a1, c1, e1, a2, c2, e2], axis=-1)

    # Simulate generations
    burnin = G_sim - G_ped

    # Pre-allocate pedigree arrays (avoids pd.concat per generation)
    total_individuals = N * G_ped
    if total_individuals > np.iinfo(np.int32).max:
        raise ValueError(
            f"Total pedigree size {total_individuals:,} exceeds int32 max "
            f"({np.iinfo(np.int32).max:,}). Reduce N or G_ped."
        )
    ped_arrays = _init_pedigree_arrays(total_individuals)
    ped_offset = 0
    id_offset = 0
    household_offset = 0

    for i in range(G_sim):
        t_gen = time.perf_counter()

        # rho_w for the current parental population:
        # founders (i=0) have gen-0 C/E; offspring from iter j have per_gen[j] C/E
        parent_ce_gen = max(0, i - 1)
        rho_w_i = rho_w_per_ce[parent_ce_gen]

        # Auto-compute R_mf for this generation's rho_w if not user-provided
        if R_mf_user is None and assort1 != 0 and assort2 != 0:
            c = rho_w_i * np.sqrt(abs(assort1 * assort2)) * np.sign(assort1 * assort2)
            R_mf_i = np.array([[assort1, c], [c, assort2]])
        else:
            R_mf_i = R_mf_user

        parents, twins, household_ids = mating(
            rng,
            sex,
            mating_lambda,
            p_mztwin,
            pheno=pheno,
            assort1=assort1,
            assort2=assort2,
            rho_w=rho_w_i,
            assort_matrix=R_mf_i,
        )
        t_mate = time.perf_counter()
        pheno, sex = reproduce(
            rng,
            pheno,
            parents,
            twins,
            household_ids,
            sd_A1,
            sd_E1_per_gen[i],
            sd_C1_per_gen[i],
            sd_A2,
            sd_E2_per_gen[i],
            sd_C2_per_gen[i],
            rA,
            rC,
            rE,
        )
        t_repro = time.perf_counter()
        if i >= burnin:
            n = len(sex)
            is_founder = i == burnin
            parent_offset = id_offset - n if not is_founder else 0
            _fill_pedigree_slice(
                ped_arrays,
                ped_offset,
                pheno,
                sex,
                parents,
                twins,
                household_ids,
                generation=i - burnin,
                is_founder=is_founder,
                id_offset=id_offset,
                parent_offset=parent_offset,
                household_offset=household_offset,
            )
            household_offset += int(household_ids.max()) + 1
            id_offset += n
            ped_offset += n
        t_fill = time.perf_counter()

        # Per-generation data shape checkpoints
        fam_sizes = np.bincount(household_ids)
        logger.info(
            "Generation %d: %d twins, mean family size %.2f [mating=%.1fs, reproduce=%.1fs, fill=%.1fs, total=%.1fs]",
            i,
            len(twins) * 2,
            fam_sizes.mean(),
            t_mate - t_gen,
            t_repro - t_mate,
            t_fill - t_repro,
            t_fill - t_gen,
        )

    pedigree = _arrays_to_dataframe(ped_arrays, ped_offset)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Simulation complete in %.1fs: pedigree has %d individuals",
        elapsed,
        len(pedigree),
    )

    return pedigree


def cli() -> None:
    """Command-line interface for running ACE simulations."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Run ACE pedigree simulation")
    add_logging_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--N", type=int, default=1000, help="Founder population size")
    parser.add_argument("--G-ped", type=int, default=3, help="Number of pedigree generations")
    parser.add_argument("--G-sim", type=int, default=None, help="Number of burn-in generations (default: G_ped)")
    parser.add_argument("--mating-lambda", type=float, default=0.5, help="ZTP mating count lambda")
    parser.add_argument("--p-mztwin", type=float, default=0.02, help="Probability of MZ twinning")
    parser.add_argument("--A1", type=float, default=0.5, help="Additive genetic variance for trait 1")
    parser.add_argument("--C1", type=float, default=0.2, help="Shared environment variance for trait 1")
    parser.add_argument(
        "--E1", type=float, default=None, help="Unique environment variance for trait 1 (default: 1-A1-C1)"
    )
    parser.add_argument("--A2", type=float, default=0.5, help="Additive genetic variance for trait 2")
    parser.add_argument("--C2", type=float, default=0.2, help="Shared environment variance for trait 2")
    parser.add_argument(
        "--E2", type=float, default=None, help="Unique environment variance for trait 2 (default: 1-A2-C2)"
    )
    parser.add_argument("--rA", type=float, default=0.5, help="Cross-trait genetic correlation")
    parser.add_argument("--rC", type=float, default=0.3, help="Cross-trait shared environment correlation")
    parser.add_argument("--rE", type=float, default=0.0, help="Cross-trait unique environment correlation")
    parser.add_argument("--assort1", type=float, default=0.0, help="Mate correlation on trait 1 liability")
    parser.add_argument("--assort2", type=float, default=0.0, help="Mate correlation on trait 2 liability")
    parser.add_argument("--output-pedigree", required=True, help="Output pedigree parquet path")
    parser.add_argument("--output-params", required=True, help="Output params YAML path")
    parser.add_argument("--rep", type=int, default=1, help="Replicate number")
    args = parser.parse_args()

    init_logging(args)

    pedigree = run_simulation(
        seed=args.seed,
        N=args.N,
        G_ped=args.G_ped,
        mating_lambda=args.mating_lambda,
        p_mztwin=args.p_mztwin,
        A1=args.A1,
        C1=args.C1,
        A2=args.A2,
        C2=args.C2,
        rA=args.rA,
        rC=args.rC,
        rE=args.rE,
        E1=args.E1,
        E2=args.E2,
        G_sim=args.G_sim,
        assort1=args.assort1,
        assort2=args.assort2,
    )

    save_parquet(pedigree, args.output_pedigree)

    _E1 = args.E1 if args.E1 is not None else 1.0 - args.A1 - args.C1
    _E2 = args.E2 if args.E2 is not None else 1.0 - args.A2 - args.C2
    params_dict = {
        "seed": args.seed,
        "rep": args.rep,
        "A1": args.A1,
        "C1": args.C1,
        "E1": _E1,
        "A2": args.A2,
        "C2": args.C2,
        "E2": _E2,
        "rA": args.rA,
        "rC": args.rC,
        "rE": args.rE,
        "N": args.N,
        "G_ped": args.G_ped,
        "G_sim": args.G_sim or args.G_ped,
        "mating_lambda": args.mating_lambda,
        "p_mztwin": args.p_mztwin,
        "assort1": args.assort1,
        "assort2": args.assort2,
    }
    with open(args.output_params, "w", encoding="utf-8") as f:
        yaml.dump(params_dict, f, default_flow_style=False)
