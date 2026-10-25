"""ACE pedigree simulation.

Simulates multi-generational pedigrees with:
- A: Additive genetic component
- C: Common/shared environment component
- E: Unique environment component

Supports single-trait and two-trait (bivariate) modes with configurable
cross-trait correlations for genetic (rA) and common environment (rC) components.
"""

__all__ = [
    "METROPOLIS_TOL",
    "add_to_pedigree",
    "allocate_offspring",
    "balance_mating_slots",
    "draw_mating_counts",
    "generate_correlated_components",
    "generate_mendelian_noise",
    "mating",
    "pair_partners",
    "place_twins",
    "reproduce",
    "resolve_per_gen_param",
    "run_simulation",
]

import argparse
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from simace.core.cohort import write_pedigree
from simace.core.schema import PEDIGREE
from simace.core.stage import stage
from simace.simulation.assortment import METROPOLIS_TOL, AssortmentPlan, _cross_am_matrix, write_mating_log
from simace.simulation.inbreeding_plan import DEFAULT_DEPTH, INBREEDING_MODES, resolve
from simace.simulation.params import CTransmission, SimulationParams, c_covariance

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, TypeVar

    from simace.simulation.inbreeding import Inbreeding

    _F = TypeVar("_F", bound=Callable[..., Any])

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
        """Identity decorator under type checking only.

        numba 0.66 became a PEP 561 typed package, so importing the real
        ``njit`` here makes every jitted function a ``Dispatcher`` with an
        opaque return type that breaks inference at the call sites.  Aliasing
        to identity lets a type checker see the underlying Python function's
        signature instead.
        """

else:
    try:
        from numba import njit
    except ImportError:
        njit = None

logger = logging.getLogger(__name__)

# Columns of the (n, 6) per-individual component matrix ``pheno``.
_A1, _C1, _E1, _A2, _C2, _E2 = range(6)
_TRAIT_COLUMNS = {1: (_A1, _C1, _E1), 2: (_A2, _C2, _E2)}


def _liability(pheno: np.ndarray, trait: int, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
    """Return ``A + C + E`` for ``trait`` over ``rows`` of ``pheno``."""
    a, c, e = _TRAIT_COLUMNS[trait]
    return pheno[rows, a] + pheno[rows, c] + pheno[rows, e]


def resolve_per_gen_param(
    value: float | dict[int, float],
    G: int,
    name: str = "param",
    allow_negative: bool = False,
) -> list[float]:
    """Resolve a variance-component parameter to a per-generation list.

    Args:
        value: scalar (constant across all generations) or dict mapping
               generation index → value.  Missing generation keys are
               forward-filled from the most recent earlier key.
        G: total number of generations to resolve for (indices 0..G-1).
        name: parameter name, used in error messages.
        allow_negative: if False (default), values must be ≥ 0. Set True for
            parameters with a natural negative range (e.g. assortative-mating
            correlations).

    Returns:
        List of length *G* with the resolved value for each generation.

    Raises:
        ValueError: if any resolved value is negative (and allow_negative is
            False) or if a dict has no key <= 0 (so generation 0 would be
            undefined).
    """
    if isinstance(value, (int, float)):
        v = float(value)
        if not allow_negative and v < 0:
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
    if not allow_negative:
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
    rng: np.random.Generator,
    n: int,
    sd_A1: float,
    sd_A2: float,
    rA: float,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated Mendelian sampling noise for two traits.

    Under the infinitesimal model, the Mendelian sampling variance is
    0.5 * Var(A) for each trait, so sd_noise = sd_A / sqrt(2)
    (Bulmer, 1971, Am. Nat., 105, 201-211). With parental inbreeding it is
    0.5 * Var(A) * (1 - F_bar): ``scale`` multiplies each offspring's noise
    in both traits by ``sqrt(1 - F_bar)`` after the draw, so it consumes the
    same random numbers (ADR 0033).

    Args:
        rng: numpy random generator
        n: number of offspring
        sd_A1: standard deviation of A1 (sqrt of A1 variance)
        sd_A2: standard deviation of A2 (sqrt of A2 variance)
        rA: genetic correlation between traits
        scale: per-offspring factor, shape (n,), or None for none

    Returns:
        (noise1, noise2): tuple of arrays, each shape (n,)
    """
    sd_noise1 = sd_A1 / np.sqrt(2)
    sd_noise2 = sd_A2 / np.sqrt(2)
    noise1, noise2 = generate_correlated_components(rng, n, sd_noise1, sd_noise2, rA)
    if scale is not None:
        noise1, noise2 = noise1 * scale, noise2 * scale
    return noise1, noise2


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
    and pairs positionally. Each duplicate ``(mother, father)`` pair swaps its
    father with a uniformly random non-duplicate row's, for at most 5 passes;
    a duplicate left after them is kept, next to its first row.

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
            return matings  # mothers in index order, each couple once
        dup_idxs = np.where(is_dup)[0]
        non_dup_idxs = np.where(~is_dup)[0]
        if len(non_dup_idxs) == 0:
            break
        for d in dup_idxs:
            swap_with = rng.choice(non_dup_idxs)
            matings[d, 1], matings[swap_with, 1] = matings[swap_with, 1], matings[d, 1]

    return _family_order(matings)


def _family_order(matings: np.ndarray) -> np.ndarray:
    """Sort ``[mother, father]`` rows by mother then father, so a repeated couple's rows are adjacent."""
    return matings[np.lexsort((matings[:, 1], matings[:, 0]))]


def _draw_pairs_python(indices, n):
    """Partial Fisher-Yates: fill ``indices[:n]`` with a uniform random draw from all of ``indices``.

    Positions ``2k`` and ``2k + 1`` then hold one proposed pair, disjoint
    from the batch's other pairs. Drawing from the whole array, not only the
    first ``n`` positions, lets the last row be proposed when ``M`` is odd.
    Draws from the legacy global RNG the calling kernel seeded: NumPy's when
    a ``_python`` kernel calls it, numba's when a jitted kernel inlines it.
    """
    M = len(indices)
    for i in range(n):
        j = np.random.randint(i, M)  # noqa: NPY002
        indices[i], indices[j] = indices[j], indices[i]


if njit is not None:
    from numba.extending import register_jitable

    # Plain Python when called from Python, so a _python kernel stays deterministic
    # under its own np.random.seed; compiled inline when a jitted kernel calls it.
    _draw_pairs = register_jitable(_draw_pairs_python)
else:
    _draw_pairs = _draw_pairs_python


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
    # numba's njit supports only the legacy global RNG, not np.random.Generator.
    np.random.seed(seed)  # noqa: NPY002
    indices = np.arange(M, dtype=np.int64)
    proposals_done = 0

    while proposals_done < max_proposals:
        r1_err = abs(S1 / M - T1 / M)
        r2_err = abs(S2 / M - T2 / M)
        r12_err = abs(S12 / M - T12 / M)
        r21_err = abs(S21 / M - T21 / M)
        if max(r1_err, r2_err, r12_err, r21_err) < tol:
            break

        batch = min(M // 2, max_proposals - proposals_done)
        _draw_pairs(indices, 2 * batch)

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


def _has_couple_python(fathers, rows_ptr, rows, female, father):
    """Whether ``female`` (a CSR group of ``rows``) is already paired with ``father`` in one of her rows."""
    for k in range(rows_ptr[female], rows_ptr[female + 1]):  # noqa: SIM110  # njit compiles no any() over a generator
        if fathers[rows[k]] == father:
            return True
    return False


if njit is not None:
    _has_couple = njit(cache=True)(_has_couple_python)
else:
    _has_couple = _has_couple_python


def _metropolis_single_python(fz, mz, fathers, female_group, rows_ptr, rows, S, T, M, tol, max_proposals, seed):
    """Greedy father swaps until ``|S/M − T/M| < tol`` on one trait, never creating a duplicate couple.

    ``fz`` / ``mz`` are row ``k``'s mother's and father's standardized
    liability, ``fathers`` the father ids, ``female_group[k]`` the mother's
    group in the CSR ``rows_ptr`` / ``rows`` of rows per mother. Swapping the
    fathers of rows ``ii`` and ``jj`` changes ``S`` by
    ``(fz[ii] − fz[jj]) · (mz[jj] − mz[ii])``; an improving swap is refused when
    either new couple already exists. Each batch proposes ``M // 2`` disjoint
    pairs drawn from all ``M`` rows.
    """
    # numba's njit supports only the legacy global RNG, not np.random.Generator.
    np.random.seed(seed)  # noqa: NPY002
    indices = np.arange(M, dtype=np.int64)
    proposals_done = 0

    while proposals_done < max_proposals:
        if abs(S / M - T / M) < tol:
            break

        batch = min(M // 2, max_proposals - proposals_done)
        _draw_pairs(indices, 2 * batch)

        for _k in range(batch):
            ii = indices[2 * _k]
            jj = indices[2 * _k + 1]
            dk = (fz[ii] - fz[jj]) * (mz[jj] - mz[ii])
            if abs(S + dk - T) >= abs(S - T):
                continue
            if _has_couple(fathers, rows_ptr, rows, female_group[ii], fathers[jj]) or _has_couple(
                fathers, rows_ptr, rows, female_group[jj], fathers[ii]
            ):
                continue
            mz[ii], mz[jj] = mz[jj], mz[ii]
            fathers[ii], fathers[jj] = fathers[jj], fathers[ii]
            S += dk

        proposals_done += batch

    return S, proposals_done


if njit is not None:
    _metropolis_single = njit(cache=True, fastmath=True)(_metropolis_single_python)
else:
    _metropolis_single = _metropolis_single_python


def _select_python(pts, idx, axis, lo, hi, k):
    """Quickselect rows ``lo:hi`` of ``pts`` (and ``idx`` with them) on ``pts[:, axis]``.

    Afterwards row ``k`` holds the k-th smallest value, rows left of it are no
    larger and rows right of it no smaller. Rows move in place, so the scans
    read memory sequentially.
    """
    hi -= 1
    while hi > lo:
        a = pts[lo, axis]
        b = pts[(lo + hi) // 2, axis]
        c = pts[hi, axis]
        pivot = max(min(a, b), min(max(a, b), c))
        i = lo
        j = hi
        while i <= j:
            while pts[i, axis] < pivot:
                i += 1
            while pts[j, axis] > pivot:
                j -= 1
            if i <= j:
                for col in range(2):
                    tmp = pts[i, col]
                    pts[i, col] = pts[j, col]
                    pts[j, col] = tmp
                tmp_i = idx[i]
                idx[i] = idx[j]
                idx[j] = tmp_i
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return


if njit is not None:
    _select = njit(cache=True)(_select_python)
else:
    _select = _select_python


def _kd_match_python(target, males):
    """Pair each target point with a nearby male point, one to one.

    Splits both ``(M, 2)`` point sets at the same rank on alternating axes
    until every cell holds one point of each, so cell ``k`` of ``target`` and
    of ``males`` cover the same quantile region. Both sets should share one
    distribution with near-independent axes (whiten them first).

    Returns:
        ``perm`` with ``males[perm[i]]`` matched to ``target[i]``.
    """
    M = target.shape[0]
    pt = target.copy()
    pm = males.copy()
    it = np.arange(M, dtype=np.int64)
    im = np.arange(M, dtype=np.int64)
    stack = [(np.int64(0), np.int64(M), np.int64(0))]
    while len(stack) > 0:
        lo, hi, axis = stack.pop()
        if hi - lo < 2:
            continue
        mid = lo + (hi - lo) // 2
        _select(pt, it, axis, lo, hi, mid)
        _select(pm, im, axis, lo, hi, mid)
        stack.append((lo, mid, 1 - axis))
        stack.append((mid, hi, 1 - axis))
    perm = np.empty(M, dtype=np.int64)
    perm[it] = im
    return perm


if njit is not None:
    _kd_match = njit(cache=True)(_kd_match_python)
else:
    _kd_match = _kd_match_python


#: Smallest standard deviation a trait may have among one sex's mating slots in
#: two-trait assortment; below it the trait is treated as constant.
_MIN_TRAIT_SD = 1e-12

#: Largest |within-sex trait correlation| two-trait assortment accepts; above it
#: the two traits are treated as collinear and ``R_ff`` / ``R_mm`` as singular.
_MAX_TRAIT_CORR = 1.0 - 1e-6

#: Roundoff tolerance on the conditional covariance's smallest eigenvalue: within
#: it the eigenvalue is clipped to 0, beyond it the requested ``R_mf`` is infeasible.
_CONDITIONAL_EIG_TOL = 1e-8


def _standardized_traits(liab1: np.ndarray, liab2: np.ndarray, sex: str) -> np.ndarray:
    """Z-score both liabilities of one sex's slots; raise if either is constant."""
    columns = []
    for trait, liab in ((1, liab1), (2, liab2)):
        sd = liab.std()
        if not sd > _MIN_TRAIT_SD:
            raise ValueError(
                f"Two-trait assortative mating: {sex} trait-{trait} liability is constant across "
                f"mating slots (SD {sd:.3g}); the within-sex trait correlation is undefined. "
                f"Use single-trait assortment instead."
            )
        columns.append((liab - liab.mean()) / sd)
    return np.column_stack(columns)


def _realised_trait_corr(z: np.ndarray, sex: str) -> np.ndarray:
    """The realised 2x2 trait correlation of one sex's slots; raise if singular."""
    r = np.corrcoef(z, rowvar=False)[0, 1]
    if not abs(r) <= _MAX_TRAIT_CORR:
        raise ValueError(
            f"Two-trait assortative mating: the realised {sex} within-person trait correlation "
            f"is {r:.6f}; the traits are collinear across mating slots, so the conditional mate "
            f"draw is undefined. Use single-trait assortment instead."
        )
    return np.array([[1.0, r], [r, 1.0]])


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
    pairing_log: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Create mating pairs with assortative mating.

    Single-trait case (one of assort1/assort2 zero): bivariate Gaussian copula
    on the active trait, then, after dedup, Metropolis father swaps that never
    create a duplicate couple until the trait's slot-level Pearson correlation
    is within ``METROPOLIS_TOL`` of its target (ADR 0032; the copula alone
    falls short, issue #74).

    Both-nonzero case: 4-variate Gaussian copula on the mate correlation
    structure of Border et al. (2022, Science). Each female's ideal mate is
    drawn from the Gaussian conditional given her two standardized
    liabilities, males are matched to those draws with ``_kd_match``, and
    Metropolis greedy swaps remove the remaining sampling error in the four
    Pearson correlations. Mate liabilities are jointly normal, so thresholded
    mate concordance matches ``R_mf`` (issue #34: the previous moment
    matching hit the Pearson targets with heavy joint tails).

    Args:
        rng: numpy random generator
        male_idxs: indices of males in the parental population
        male_counts: balanced mating slot counts per male
        female_idxs: indices of females in the parental population
        female_counts: balanced mating slot counts per female
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2] for parents
        assort1: target mate correlation on trait 1 liability, in [-1, 1]
        assort2: target mate correlation on trait 2 liability, in [-1, 1]
        rho_w: configured within-person cross-trait liability correlation;
            sets the default cross-trait mate correlation when
            ``assort_matrix`` is None
        assort_matrix: optional full 2x2 mate correlation matrix R_mf.
            When provided, overrides the auto-computed off-diagonals.
        pairing_log: receives the pairing record: ``mode`` (``single`` or
            ``both``), ``M``, ``proposals``, ``max_proposals``, the final
            post-dedup slot-level ``errors`` (correlation minus target),
            ``duplicates_left`` and ``converged``.

    Returns:
        ``(M, 2)`` array of ``[mother_idx, father_idx]``, in mother order,
        with any repeated couple's rows adjacent.

    Raises:
        ValueError: in the both-nonzero case with two or more slots, if a
            trait is constant or the two traits are collinear within either
            sex (ADR 0024: use single-trait assortment), or if ``R_mf`` is
            infeasible for the realised within-sex trait correlations.
    """
    # Imported here, not at module level: scipy.spatial costs ~160 ms to import
    # and only assortative mating needs the tree.
    from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]  # real class; scipy stubs omit it

    # 1. Expand slots
    female_slots = np.repeat(female_idxs, female_counts)
    male_slots = np.repeat(male_idxs, male_counts)
    M = len(female_slots)
    mode = "both" if assort1 != 0 and assort2 != 0 else "single"
    if M < 2:
        # Nothing to sort: zero or one possible pairing.
        matings = np.column_stack([female_slots, male_slots])
        if pairing_log is not None:
            pairing_log.append(_pairing_record(mode, M, 0, 0, {}, 0))
        return matings

    # 2. Compute liability per slot
    liab1_f = _liability(pheno, 1, female_slots)
    liab2_f = _liability(pheno, 2, female_slots)
    liab1_m = _liability(pheno, 1, male_slots)
    liab2_m = _liability(pheno, 2, male_slots)

    if mode == "both":
        # --- Both traits nonzero: 4-variate Gaussian copula on R_mf ---
        r1, r2 = assort1, assort2

        # Build full R_mf with cross-trait off-diagonals
        if assort_matrix is not None:
            R_mf = np.asarray(assort_matrix, dtype=np.float64)
        else:
            R_mf = _cross_am_matrix(r1, r2, rho_w)
        c_target = R_mf[0, 1]

        fz = _standardized_traits(liab1_f, liab2_f, "female")
        males_z = _standardized_traits(liab1_m, liab2_m, "male")

        # Draw each female's ideal mate from the Gaussian conditional
        # m | f ~ N(f R_ff^-1 R_mf, R_mm - R_mf' R_ff^-1 R_mf), then match
        # males to those draws. R_ff and R_mm are the realised within-sex
        # correlations, not rho_w: cross-trait AM moves the population's
        # trait correlation away from the configured value, and a mismatch
        # leaves the targets and the males with different distributions,
        # which bends the joint tails. Whitening gives the k-d split
        # independent axes.
        R_ff = _realised_trait_corr(fz, "female")
        R_mm = _realised_trait_corr(males_z, "male")
        B = np.linalg.solve(R_ff, R_mf)
        eigval, eigvec = np.linalg.eigh(R_mm - R_mf.T @ B)
        if eigval[0] < -_CONDITIONAL_EIG_TOL:
            raise ValueError(
                f"Requested mate correlation matrix R_mf={np.round(R_mf, 4).tolist()} is not attainable with the "
                f"realised within-sex trait correlations (female {R_ff[0, 1]:.4f}, male {R_mm[0, 1]:.4f}): "
                f"the conditional covariance R_mm - R_mf' R_ff^-1 R_mf has eigenvalue {eigval[0]:.3g} < 0. "
                f"Reduce the assortment or cross-trait mate correlations."
            )
        noise_factor = eigvec * np.sqrt(np.clip(eigval, 0.0, None))
        target = fz @ B + rng.standard_normal((M, 2)) @ noise_factor.T
        whiten = np.linalg.inv(np.linalg.cholesky(R_mm)).T
        male_perm = _kd_match(target @ whiten, males_z @ whiten)

        # Metropolis swaps remove the sampling error of the draw so the
        # Pearson correlations land on R_mf.
        mz = males_z[male_perm]

        T1 = r1 * M
        T2 = r2 * M
        T12 = c_target * M
        T21 = c_target * M
        S1 = float(fz[:, 0] @ mz[:, 0])
        S2 = float(fz[:, 1] @ mz[:, 1])
        S12 = float(fz[:, 0] @ mz[:, 1])
        S21 = float(fz[:, 1] @ mz[:, 0])

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
            METROPOLIS_TOL,
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
        male_sorted = male_slots[order_m]

        r_eff = min(np.sqrt(assort1**2 + assort2**2), 1.0)

        cov = [[1.0, r_eff], [r_eff, 1.0]]
        z = rng.multivariate_normal([0.0, 0.0], cov, size=M)

        rank_f = np.argsort(np.argsort(z[:, 0]))
        rank_m = np.argsort(np.argsort(z[:, 1]))

        # Row k's mother is female slot slot_of_row[k]; dedup and the swaps move only fathers.
        slot_of_row = order_f[rank_f]
        matings = np.column_stack([female_slots[slot_of_row], male_sorted[rank_m]])

    n_swaps, duplicates_left = _dedup_matings(matings, liab1_f, liab2_f, cKDTree)

    if mode == "both":
        proposals = proposals_done
        targets = {"r1": (0, 0, r1), "r2": (1, 1, r2), "r12": (0, 1, c_target), "r21": (1, 0, c_target)}
        if n_swaps == 0:
            # No dedup swap: the Metropolis sums are the final slot-level correlations.
            sums = {"r1": S1, "r2": S2, "r12": S12, "r21": S21}
            errors = {name: sums[name] / M - r for name, (_, _, r) in targets.items()}
        else:
            errors = _slot_errors(pheno, matings, targets)
    else:
        # The copula's correlation is random around the target and dedup moves it
        # further (issue #74); swaps that never create a duplicate couple pin it.
        t = 0 if assort1 != 0 else 1
        target_r = assort1 if assort1 != 0 else assort2
        max_proposals = 8 * M
        female_liab = liab1_f if t == 0 else liab2_f
        refined = _refine_single_trait(rng, pheno, matings, slot_of_row, female_liab, t, target_r, max_proposals)
        # A constant active trait has no correlation to pin: no refinement and no errors.
        proposals, errors = (0, {}) if refined is None else (refined[0], {f"r{t + 1}": refined[1] / M - target_r})
        if duplicates_left:  # the swaps never add a duplicate couple, but they can remove one
            duplicates_left = int(_find_duplicate_pairs(matings).sum())

    record = _pairing_record(
        mode, M, proposals, max_proposals, {k: float(v) for k, v in errors.items()}, duplicates_left
    )
    if not errors:
        logger.warning(
            "Assortative mating (single): trait %d's liability is constant across mating slots; nothing was pinned",
            t + 1,
        )
    elif not record["converged"]:
        logger.warning(
            "Assortative mating (%s) missed its target after %d proposals: %s", mode, proposals, record["errors"]
        )
    if pairing_log is not None:
        pairing_log.append(record)
    if mode == "single":
        # Back to female-slot order, i.e. mother order (row k holds slot slot_of_row[k])
        in_slot_order = np.empty_like(matings)
        in_slot_order[slot_of_row] = matings
        matings = in_slot_order
    return _family_order(matings) if record["duplicates_left"] else matings


def _refine_single_trait(
    rng: np.random.Generator,
    pheno: np.ndarray,
    matings: np.ndarray,
    slot_of_row: np.ndarray,
    female_liab: np.ndarray,
    t: int,
    target_r: float,
    max_proposals: int,
) -> tuple[int, float] | None:
    """Swap fathers in ``matings`` (in place) until trait ``t``'s slot-level correlation is within tolerance.

    ``slot_of_row[k]`` is the female slot of row ``k``'s mother, and
    ``female_liab`` trait ``t``'s liability per female slot. Female slots are
    in female order (``np.repeat`` over ascending ids), so inverting
    ``slot_of_row`` groups the rows by mother without a sort.

    Returns:
        The number of proposals made and the final sum S = M · r of the
        slot-level correlation, or None when trait ``t`` is constant on
        either side, which leaves the correlation undefined and the pairing
        as the copula and dedup left it.
    """
    M = len(matings)
    f_liab, m_liab = female_liab[slot_of_row], _liability(pheno, t + 1, matings[:, 1])
    if not (f_liab.std() > _MIN_TRAIT_SD and m_liab.std() > _MIN_TRAIT_SD):
        return None
    fz, mz = _zscore(f_liab), _zscore(m_liab)
    fathers = matings[:, 1].astype(np.int64)
    mothers = matings[:, 0].astype(np.int64)
    # Rows grouped by mother (CSR), for the duplicate-couple check.
    rows = np.empty(M, dtype=np.int64)
    rows[slot_of_row] = np.arange(M, dtype=np.int64)
    rows_ptr = np.zeros(int(mothers.max()) + 2, dtype=np.int64)
    np.cumsum(np.bincount(mothers, minlength=len(rows_ptr) - 1), out=rows_ptr[1:])
    seed = int(rng.integers(2**63))
    S, proposals = _metropolis_single(
        fz, mz, fathers, mothers, rows_ptr, rows, float(fz @ mz), target_r * M, M, METROPOLIS_TOL, max_proposals, seed
    )
    matings[:, 1] = fathers
    return int(proposals), float(S)


def _pairing_record(
    mode: str, M: int, proposals: int, max_proposals: int, errors: dict[str, float], duplicates_left: int
) -> dict[str, Any]:
    """One assortative iteration's pairing record (``run_simulation``'s ``mating_log``).

    ``converged`` needs at least one error, so a record with nothing pinned
    (M < 2, or a constant active trait) does not read as converged.
    """
    return {
        "mode": mode,
        "M": int(M),
        "proposals": int(proposals),
        "max_proposals": int(max_proposals),
        "errors": errors,
        "duplicates_left": int(duplicates_left),
        "converged": bool(errors) and all(abs(e) < METROPOLIS_TOL for e in errors.values()),
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize with ddof 0, so a mean of products is a Pearson correlation."""
    return (x - x.mean()) / x.std()


def _slot_errors(
    pheno: np.ndarray, matings: np.ndarray, targets: dict[str, tuple[int, int, float]]
) -> dict[str, float]:
    """Slot-level Pearson correlation minus target, per ``name: (mother trait, father trait, target)`` (0-based)."""
    # One row gather per side, then each trait's liability from its columns.
    sides = [pheno[matings[:, side]] for side in (0, 1)]
    z = {(side, t): _zscore(_liability(sides[side], t + 1)) for side in (0, 1) for t in (0, 1)}
    return {name: float(np.mean(z[0, i] * z[1, j]) - r) for name, (i, j, r) in targets.items()}


def _dedup_matings(matings: np.ndarray, liab1_f: np.ndarray, liab2_f: np.ndarray, cKDTree: Any) -> tuple[int, int]:
    """Swap the father of each duplicate couple with a nearby female's (in place), for at most 5 passes.

    Returns:
        The number of father swaps made and the number of duplicate couples left.

    ``liab1_f`` / ``liab2_f`` are the female slots' liabilities in slot order.
    The single-trait copula reorders the rows, so there the partner is not a
    near female. Aligning them is worse: a duplicate's nearest row is then
    often the same woman's other slot, and the swap changes nothing.
    """
    # Deduplicate: swap with female-proximity partner to preserve correlations
    t_dedup = time.perf_counter()
    n_swaps = 0
    clean = False
    for _attempt in range(5):
        is_dup = _find_duplicate_pairs(matings)
        if not is_dup.any():
            clean = True
            break
        dup_idxs = np.where(is_dup)[0]
        non_dup_idxs = np.where(~is_dup)[0]
        if len(non_dup_idxs) == 0:
            break
        n_swaps += len(dup_idxs)

        # KD-tree nearest-neighbor lookup: O(M log M) build + O(dups × log M) query
        nondup_liabs = np.column_stack([liab1_f[non_dup_idxs], liab2_f[non_dup_idxs]])
        tree = cKDTree(nondup_liabs)
        dup_liabs = np.column_stack([liab1_f[dup_idxs], liab2_f[dup_idxs]])
        _, nn_idx = tree.query(dup_liabs, k=1)

        for i, d in enumerate(dup_idxs):
            swap_with = non_dup_idxs[nn_idx[i]]
            matings[d, 1], matings[swap_with, 1] = matings[swap_with, 1], matings[d, 1]
    logger.debug(
        "Dedup: %d father swaps in %.1fs",
        n_swaps,
        time.perf_counter() - t_dedup,
    )
    return n_swaps, 0 if clean else int(_find_duplicate_pairs(matings).sum())


def allocate_offspring(rng: np.random.Generator, n_matings: int, N: int) -> np.ndarray:
    """Distribute *N* offspring across *n_matings* matings via multinomial.

    Returns:
        Array of shape ``(n_matings,)`` summing to exactly *N*.
    """
    probs = np.ones(n_matings) / n_matings
    return rng.multinomial(N, probs)


def place_twins(rng: np.random.Generator, n_matings: int, N: int, p_mztwin: float) -> tuple[np.ndarray, np.ndarray]:
    """Allocate *N* offspring as births, a fraction *p_mztwin* of them MZ twin births, over *n_matings* couples.

    Draws ``K ~ Binomial(N, p / (1 + p))`` MZ pairs (at most ``N // 2``), so
    ``B = N - K`` births with ``E[K / B] = p``. The births are spread over
    the couples by :func:`allocate_offspring`, and ``K`` of them, uniformly
    at random, are twin births. Each couple's children are laid out in birth
    order, so co-twins are adjacent, at any position in the family, and a
    couple can have several MZ pairs.

    Returns:
        ``(offspring_counts, first_twins)``: children per couple, summing to
        *N*, and the sorted offspring positions (into the couples'
        contiguous blocks, in mating order) of each pair's first co-twin;
        the second is the next position.
    """
    K = min(int(rng.binomial(N, p_mztwin / (1 + p_mztwin))), N // 2) if p_mztwin > 0 else 0
    births = allocate_offspring(rng, n_matings, N - K)
    twin_births = np.sort(rng.choice(N - K, size=K, replace=False, shuffle=False))
    couple = np.searchsorted(np.cumsum(births), twin_births, side="right")
    # Each earlier twin birth shifts a birth's first child one position on.
    return births + np.bincount(couple, minlength=n_matings), twin_births + np.arange(K)


def _mating_standard(
    rng: np.random.Generator,
    parental_sex: np.ndarray,
    mating_lambda: float,
    p_mztwin: float,
    pheno: np.ndarray | None = None,
    assort1: float = 0.0,
    assort2: float = 0.0,
    rho_w: float = 0.0,
    assort_matrix: np.ndarray | None = None,
    pairing_log: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate parent-offspring pairings via a modular mating pipeline.

    1. Separate males/females and draw ZTP(mating_lambda) mating counts.
    2. Balance slots so both sexes have equal total.
    3. Pair partners randomly (or assortatively if assort1/assort2 != 0).
    4. Allocate N offspring as births across matings, some of them MZ twin
       births (:func:`place_twins`).
    5. Build output arrays.

    Args:
        rng: numpy random generator
        parental_sex: array of sex values (0=female, 1=male) for parents
        mating_lambda: Poisson lambda for zero-truncated mating count distribution
        p_mztwin: MZ twin rate per birth (see :func:`place_twins`)
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2]; required when
            assort1 or assort2 is nonzero
        assort1: target mate correlation on trait 1 liability
        assort2: target mate correlation on trait 2 liability
        rho_w: within-person cross-trait liability correlation, used to
            derive off-diagonal entries of the mate correlation matrix
        assort_matrix: optional full 2x2 mate correlation matrix R_mf
        pairing_log: receives this iteration's pairing record (``mode``
            ``"random"``, ``"single"`` or ``"both"``; for assortment also
            the final slot-level errors and whether they are within
            ``METROPOLIS_TOL``)

    Returns:
        parent_idxs: (N, 2) array of [mother_idx, father_idx] for each offspring
        twins: (m, 2) array of [twin1_idx, twin2_idx] pairs for MZ twins
        household_ids: (N,) array mapping each offspring to a household index

    Raises:
        ValueError: if assort is nonzero but pheno is None
        ValueError: if parental_sex contains zero males or zero females.
    """
    if (assort1 != 0 or assort2 != 0) and pheno is None:
        raise ValueError("pheno must be provided when assort1 or assort2 is nonzero")
    N = len(parental_sex)
    male_idxs = np.where(parental_sex == 1)[0]
    female_idxs = np.where(parental_sex == 0)[0]
    if len(female_idxs) == 0:
        raise ValueError("Standard mating: parent population has 0 female individuals; cannot sample mothers.")
    if len(male_idxs) == 0:
        raise ValueError("Standard mating: parent population has 0 male individuals; cannot sample fathers.")

    # 1. Draw mating counts per individual
    male_counts = draw_mating_counts(rng, len(male_idxs), mating_lambda)
    female_counts = draw_mating_counts(rng, len(female_idxs), mating_lambda)

    # 2. Balance slots
    male_counts, female_counts = balance_mating_slots(rng, male_counts, female_counts)

    # 3. Pair partners -> (M, 2) of [mother_idx, father_idx], in mother order with
    # repeated couples adjacent. Each pairing function keeps that order itself:
    # a lexsort here cost 8-14% of an assortative mating call (issue #78).
    if assort1 != 0 or assort2 != 0:
        assert pheno is not None  # required whenever assortment is active (see docstring)
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
            pairing_log=pairing_log,
        )
    else:
        matings = pair_partners(rng, male_idxs, male_counts, female_idxs, female_counts)
        if pairing_log is not None:
            pairing_log.append({"mode": "random", "M": len(matings)})
    M = len(matings)

    # 4. Allocate offspring as births, some of them MZ twin births
    offspring_counts, first_twins = place_twins(rng, M, N, p_mztwin)
    twins = np.column_stack([first_twins, first_twins + 1])

    # 5. Build output arrays: each couple's children form one block, in birth order
    parent_idxs = np.repeat(matings, offspring_counts, axis=0)

    # Household: all offspring of the same mother share a household
    _, household_ids = np.unique(parent_idxs[:, 0], return_inverse=True)

    return parent_idxs, twins, household_ids


def _mating_wf(
    rng: np.random.Generator,
    parental_sex: np.ndarray,
    N: int,
    generation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sex-structured idealized Wright-Fisher mating for one generation.

    For each of ``N`` offspring, draw one mother uniformly at random from the
    prior generation's females and one father uniformly at random from the
    males (both with replacement; selfing is impossible because parents are
    constrained to opposite sexes).  Multinomial offspring counts per parent;
    no persistent mating-pair structure; no MZ twins.

    Args:
        rng: numpy random generator.
        parental_sex: array of sex values (0=female, 1=male) for parents.
        N: number of offspring to produce.
        generation: simulation iteration index (used only for error messages
            so an empty-pool failure names the offending generation).

    Returns:
        parent_idxs: ``(N, 2)`` array of ``[mother_idx, father_idx]``.
        twins: ``(0, 2)`` empty int array (WF has no MZ twins).
        household_ids: ``(N,)`` contiguous ``[0, n_mothers)`` mapping each
            offspring to a maternal household, matching the standard path's
            grouping convention.

    Raises:
        ValueError: if ``parental_sex`` contains zero males or zero females.
    """
    male_idxs = np.where(parental_sex == 1)[0]
    female_idxs = np.where(parental_sex == 0)[0]
    if len(female_idxs) == 0:
        raise ValueError(f"WF mating: parent generation {generation} has 0 female individuals; cannot sample mothers.")
    if len(male_idxs) == 0:
        raise ValueError(f"WF mating: parent generation {generation} has 0 male individuals; cannot sample fathers.")

    mother_idx = rng.choice(female_idxs, size=N, replace=True)
    father_idx = rng.choice(male_idxs, size=N, replace=True)
    parent_idxs = np.column_stack([mother_idx, father_idx])

    twins = np.empty((0, 2), dtype=int)
    _, household_ids = np.unique(mother_idx, return_inverse=True)

    return parent_idxs, twins, household_ids


def mating(
    rng: np.random.Generator,
    parental_sex: np.ndarray,
    mating_lambda: float = 0.5,
    p_mztwin: float = 0.0,
    pheno: np.ndarray | None = None,
    assort1: float = 0.0,
    assort2: float = 0.0,
    rho_w: float = 0.0,
    assort_matrix: np.ndarray | None = None,
    *,
    mating_model: str = "standard",
    generation: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the configured mating implementation.

    Public entry point used by tests and any external caller; ``run_simulation``
    branches directly to ``_mating_standard`` / ``_mating_wf`` from inside the
    generation loop because the standard path needs per-iter AM variables that
    don't exist under WF.

    Args:
        rng: numpy random generator.
        parental_sex: array of sex values (0=female, 1=male) for parents.
        mating_lambda: Poisson lambda for the ZTP mating-count distribution.
            Standard-model only; ignored under WF.
        p_mztwin: MZ twin rate per birth.
            Standard-model only; ignored under WF.
        pheno: ``(n, 6)`` phenotype matrix used for assortative mating.
            Standard-model only; ignored under WF.
        assort1: Mate correlation on trait-1 liability.  Standard-model
            only; ignored under WF.
        assort2: Mate correlation on trait-2 liability.  Standard-model
            only; ignored under WF.
        rho_w: Within-person cross-trait liability correlation.
            Standard-model only; ignored under WF.
        assort_matrix: Optional 2x2 mate-correlation matrix.
            Standard-model only; ignored under WF.
        mating_model: ``"standard"`` (default) or ``"wright_fisher"``.
        generation: Passed to ``_mating_wf`` for error-message provenance.
            Ignored under the standard path.

    Returns:
        ``(parent_idxs, twins, household_ids)`` as produced by the chosen
        backend.
    """
    if mating_model == "wright_fisher":
        return _mating_wf(rng, parental_sex, len(parental_sex), generation)
    if mating_model == "standard":
        return _mating_standard(
            rng,
            parental_sex,
            mating_lambda,
            p_mztwin,
            pheno=pheno,
            assort1=assort1,
            assort2=assort2,
            rho_w=rho_w,
            assort_matrix=assort_matrix,
        )
    raise ValueError(f"Unknown mating_model={mating_model!r}; expected 'standard' or 'wright_fisher'.")


#: A negative eigenvalue of the fresh-draw covariance within this many standard
#: errors of the transmitted covariance is sampling noise, clipped to 0.
#: Calibrated on constant C with b_m = 1 (no structural slack, N = 1000), where
#: each clip leaves Var(C) slightly above target and the next generation copies
#: it, so the excess accumulates: the largest ratio was 5.96 over 1000 seeds at
#: G_sim = 8, 7.23 over 300 at G_sim = 20, and 8.71 over 200 at G_sim = 50.
CLIP_SE_MULTIPLIER = 12.0

#: The mates' C cross-covariance term counts as structural (assortative
#: mating), not sampling noise, beyond this many of its standard errors.
CROSS_SE_MULTIPLIER = 4.0


def _transmitted_c(
    rng: np.random.Generator,
    pheno: np.ndarray,
    parents: np.ndarray,
    household_ids: np.ndarray,
    n_hh: int,
    target: np.ndarray,
    c_transmission: CTransmission,
    clips: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return offspring (C1, C2) = transmitted part + a household draw of covariance ``target - S``.

    ``S`` is the realized covariance of the transmitted parts over the
    offspring, so the configured ``target`` is the total C covariance in
    expectation. ``S`` splits exactly into the parents' own terms
    ``B_m V_m B_m + B_f V_f B_f`` and the mates' cross term
    ``B_m V_mf B_f + B_f V_fm B_m``. Where ``target - S`` has a negative
    eigenvalue (eigenvector ``v``):

    - a cross term above ``CROSS_SE_MULTIPLIER`` standard errors along ``v``
      (``SE = 2 sqrt(v'B_m V_m B_m v * v'B_f V_f B_f v / n_eff)``) is
      assortative mating pushing ``S`` past the target, and raises;
    - otherwise an excess within ``CLIP_SE_MULTIPLIER`` standard errors of the
      own terms (``SE = sqrt(2 / n_eff) * v'(own)v``) is sampling noise,
      clipped to 0 and appended to ``clips`` as ``{"excess", "tolerance",
      "direction"}``; a larger one raises.

    ``n_eff`` is the Kish effective number of households over offspring counts.
    """
    parent_c = pheno[:, [_C1, _C2]]
    mother_c, father_c = parent_c[parents[:, 0]], parent_c[parents[:, 1]]
    B_m, B_f = np.diag(c_transmission.maternal), np.diag(c_transmission.paternal)
    cov = np.cov(np.hstack([mother_c, father_c]), rowvar=False)
    own_m, own_f = B_m @ cov[:2, :2] @ B_m, B_f @ cov[2:, 2:] @ B_f
    half_cross = B_m @ cov[:2, 2:] @ B_f
    cross = half_cross + half_cross.T
    realized = own_m + own_f + cross
    eigvals, eigvecs = np.linalg.eigh(target - realized)
    weights = np.bincount(household_ids, minlength=n_hh) / len(household_ids)
    inv_n_eff = float(np.sum(weights**2))
    for j in np.flatnonzero(eigvals < 0):
        v, excess = eigvecs[:, j], float(-eigvals[j])
        cross_v = float(v @ cross @ v)
        cross_se = 2.0 * np.sqrt(float(v @ own_m @ v) * float(v @ own_f @ v) * inv_n_eff)
        if cross_v > max(CROSS_SE_MULTIPLIER * cross_se, 1e-12):
            raise ValueError(
                f"C transmission: mates' correlated C (assortative mating) adds {cross_v:.4g} to the "
                f"transmitted C covariance {realized.round(6).tolist()}, past the configured "
                f"{target.round(6).tolist()} by {excess:.4g}; lower the coefficients or the assortment (ADR 0025)"
            )
        tolerance = max(CLIP_SE_MULTIPLIER * np.sqrt(2.0 * inv_n_eff) * float(v @ (own_m + own_f) @ v), 1e-12)
        if excess > tolerance:
            raise ValueError(
                f"C transmission: the transmitted C covariance {realized.round(6).tolist()} exceeds the "
                f"configured {target.round(6).tolist()} (fresh-draw eigenvalue {-excess:.4g}, sampling "
                f"tolerance {tolerance:.4g}); lower the coefficients or raise C (ADR 0025)"
            )
        logger.warning(
            "C transmission: clipped fresh-draw eigenvalue %.4g to 0 (tolerance %.4g); realized Var(C) exceeds the "
            "configured C by that much",
            -excess,
            tolerance,
        )
        if clips is not None:
            clips.append({"excess": excess, "tolerance": float(tolerance), "direction": v.round(6).tolist()})
    loadings = eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))
    fresh = rng.standard_normal((n_hh, 2)) @ loadings.T
    c = mother_c @ B_m + father_c @ B_f + fresh[household_ids]
    return c[:, 0], c[:, 1]


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
    c_transmission: CTransmission | None = None,
    clips: list[dict[str, Any]] | None = None,
    mendelian_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate offspring phenotypes from parents for two correlated traits.

    Additive genetic values are inherited as midparent + Mendelian noise.
    Common environment (C) is drawn per household (one household per
    mother). Without C transmission that draw is the whole of C, so
    householdmates share C and parents and children do not. With it, C is
    ``b_m * C_mother + b_f * C_father`` plus the household draw, sized so the
    configured C stays the total (see :func:`_transmitted_c`, ADR 0025).
    Unique environment (E) is drawn independently per individual.

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
        c_transmission: C transmission coefficients; None or all zero is
            the fresh household draw, identical to the code before ADR 0025.
        clips: under C transmission, collects one record per fresh-draw
            eigenvalue clipped to 0 (see :func:`_transmitted_c`).
        mendelian_scale: per-offspring ``sqrt(1 - F_bar)`` applied to the
            Mendelian noise (ADR 0033), or None for the unscaled draw.

    Returns:
        offspring: (n, 6) array of [A1, C1, E1, A2, C2, E2]
        sex_offspring: (n,) array of sex values (0=female, 1=male)

    Raises:
        ValueError: under C transmission, when the realized transmitted
            covariance exceeds the configured C by more than sampling allows.
    """
    n = len(parents)

    # Sex assignment
    sex_offspring = rng.binomial(size=n, n=1, p=0.5)

    # Additive genetic: midparent + correlated Mendelian noise
    mp1 = _midparent(pheno[:, _A1], parents)
    mp2 = _midparent(pheno[:, _A2], parents)

    noise1, noise2 = generate_mendelian_noise(rng, n, sd_A1, sd_A2, rA, mendelian_scale)
    a1_offspring = mp1 + noise1
    a2_offspring = mp2 + noise2

    # Common environment: a draw per household (household_ids are 0-based
    # contiguous from mating()), plus under C transmission the parents' C.
    n_hh = int(household_ids.max()) + 1
    if c_transmission is None or not c_transmission.active:
        hh_c1, hh_c2 = generate_correlated_components(rng, n_hh, sd_C1, sd_C2, rC)
        c1_offspring = hh_c1[household_ids]
        c2_offspring = hh_c2[household_ids]
    else:
        c1_offspring, c2_offspring = _transmitted_c(
            rng, pheno, parents, household_ids, n_hh, c_covariance(sd_C1**2, sd_C2**2, rC), c_transmission, clips
        )

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

    offspring = np.stack(  # column order: _A1 .. _E2
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


def _ped_columns(arrays: dict[str, np.ndarray]) -> list[str]:
    """The pedigree's columns: :data:`_PED_COLUMNS`, with ``F`` after ``household_id`` when tracked (ADR 0033)."""
    if "F" not in arrays:
        return _PED_COLUMNS
    at = _PED_COLUMNS.index("household_id") + 1
    return [*_PED_COLUMNS[:at], "F", *_PED_COLUMNS[at:]]


def _init_pedigree_arrays(total_rows: int, inbreeding: bool = False) -> dict[str, np.ndarray]:
    """Pre-allocate numpy arrays for the pedigree.

    Dtype choices for memory efficiency at large N:
    - int32 for IDs (supports up to 2.1B individuals)
    - int8 for sex (0/1)
    - float32 for ACE variance components (~7 significant digits)
    - float64 for liabilities (full precision for phenotype models)
    - float32 for the inbreeding coefficient ``F``, allocated only when
      ``inbreeding`` is set
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
    if inbreeding:
        _dtypes["F"] = np.float32
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
    F: np.ndarray | None = None,
) -> None:
    """Fill a slice of pre-allocated pedigree arrays for one generation (``F`` into the ``F`` array when given)."""
    n = len(pheno)
    s = slice(offset, offset + n)

    arrays["id"][s] = np.arange(n, dtype=np.int32) + np.int32(id_offset)
    arrays["sex"][s] = sex
    arrays["generation"][s] = generation
    arrays["household_id"][s] = (household_ids + household_offset).astype(np.int32)
    if F is not None:
        arrays["F"][s] = F

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
    for trait, (a, c, e) in _TRAIT_COLUMNS.items():
        arrays[f"A{trait}"][s] = pheno[:, a]
        arrays[f"C{trait}"][s] = pheno[:, c]
        arrays[f"E{trait}"][s] = pheno[:, e]
        arrays[f"liability{trait}"][s] = _liability(pheno, trait)


def _arrays_to_dataframe(arrays: dict[str, np.ndarray], total_rows: int) -> pl.DataFrame:
    """Convert pre-allocated arrays to a pedigree DataFrame."""
    return pl.DataFrame({col: arrays[col][:total_rows] for col in _ped_columns(arrays)})


def add_to_pedigree(
    pheno: np.ndarray,
    sex: np.ndarray,
    parents: np.ndarray,
    twins: np.ndarray,
    household_ids: np.ndarray,
    generation: int,
    pedigree: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Add a generation to the pedigree DataFrame (backward-compatible wrapper).

    The internal simulation loop uses pre-allocated arrays for performance.
    This function is retained for external callers and tests.
    """
    n = len(pheno)
    is_founder = pedigree is None
    id_offset = 0 if is_founder else len(pedigree)
    parent_offset = id_offset - n if not is_founder else 0
    household_offset = 0 if is_founder else int(pedigree["household_id"][-1]) + 1

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
        return pl.concat([pedigree, df], how="vertical")
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
    E1: float | dict[int, float],
    E2: float | dict[int, float],
    G_sim: int | None = None,
    assort1: float | dict[int, float] = 0.0,
    assort2: float | dict[int, float] = 0.0,
    assort_matrix: list[list[float]] | np.ndarray | None = None,
    mating_model: str = "standard",
    Ctm_maternal1: float = 0.0,
    Ctm_paternal1: float = 0.0,
    Ctm_maternal2: float = 0.0,
    Ctm_paternal2: float = 0.0,
    inbreeding_mode: str = "off",
    inbreeding_depth: int = DEFAULT_DEPTH,
    clip_log: list[dict[str, Any]] | None = None,
    mating_log: list[dict[str, Any]] | None = None,
) -> pl.DataFrame:
    """Run the full ACE simulation for two correlated traits.

    Variance components A (additive genetic), C (shared environment), and
    E (unique environment) are specified as absolute variances.  A and C are
    constant across generations; E may be specified per-generation via a dict
    mapping generation index to value (forward-filled for missing keys).

    Args:
        seed: Random seed
        N: Population size per generation (positive integer)
        G_ped: Number of generations to record in pedigree (integer >= 1)
        mating_lambda: Poisson lambda for zero-truncated mating count distribution (> 0)
        p_mztwin: MZ twin rate per birth, in [0, 1)
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
            Scalar (constant across generations) or dict mapping raw simulation
            iteration → value (forward-filled, same coordinate system as E/C).
            **Mutually exclusive with ``assort_matrix``** — pass scalar AM if
            you want a fixed 2x2 mate-correlation matrix.
        assort2: Same format as assort1, for trait 2.
        assort_matrix: Optional full 2x2 mate correlation matrix R_mf.
            Overrides assort1/assort2 diagonal with matrix diagonal. Only
            valid when both assort1 and assort2 are scalars.
        mating_model: ``"standard"`` (default) or ``"wright_fisher"``.
            Under ``"wright_fisher"``, ``mating_lambda``, ``p_mztwin``,
            ``assort1``, ``assort2``, and ``assort_matrix`` are silently
            ignored at runtime — the WF path samples each offspring's
            mother and father independently from the prior generation's
            females / males, with multinomial offspring counts and no MZ
            twins.  See ``docs/adr/0002-wright-fisher-mating-model.md``.
        Ctm_maternal1: Trait 1 maternal C transmission coefficient ``b_m``
            in [0, 1] (ADR 0025). Default 0, the fresh household draw.
        Ctm_paternal1: Trait 1 paternal C transmission coefficient ``b_f``.
        Ctm_maternal2: Trait 2 maternal C transmission coefficient.
        Ctm_paternal2: Trait 2 paternal C transmission coefficient.
        inbreeding_mode: ``"off"`` (default), ``"approx"`` or ``"exact"``:
            whether and how each offspring's Mendelian residual is scaled by
            ``sqrt(1 - F_bar)``, ``F_bar`` its parents' mean inbreeding
            coefficient (ADR 0033).
        inbreeding_depth: Generations ``approx`` traces exactly (>= 1).
        clip_log: Under C transmission, receives a record per fresh-draw
            eigenvalue clipped to 0, each with its ``generation`` in
            recorded coordinates (negative during burn-in), ``excess``,
            ``tolerance`` and ``direction``.
        mating_log: Receives one pairing record per mating iteration, with
            its ``iteration``, its offspring's ``generation`` in recorded
            coordinates, and ``mode`` (``random``, ``single``, ``both`` or
            ``wright_fisher``). An assortative record also has ``M``,
            ``proposals``, ``max_proposals``, the final slot-level
            ``errors`` against the targets, ``duplicates_left`` and
            ``converged`` (at least one error, and every error within
            ``METROPOLIS_TOL``; a record that pinned nothing is not
            converged).

    Returns:
        Pedigree DataFrame with columns id, sex, mother, father, twin,
        generation, household_id, A1, C1, E1, liability1, A2, C2, E2,
        liability2.

    Raises:
        ValueError: if any parameter is outside its valid range
    """
    # Validate + normalize inputs (incl. assort_matrix → assort1/assort2/R_mf).
    # run_simulation is also a public test API, so validation cannot rely on
    # config-load having run. See simace.simulation.params.SimulationParams.
    params = SimulationParams.create(
        seed=seed,
        N=N,
        G_ped=G_ped,
        mating_lambda=mating_lambda,
        p_mztwin=p_mztwin,
        A1=A1,
        C1=C1,
        A2=A2,
        C2=C2,
        rA=rA,
        rC=rC,
        rE=rE,
        E1=E1,
        E2=E2,
        G_sim=G_sim,
        assort1=assort1,
        assort2=assort2,
        assort_matrix=assort_matrix,
        mating_model=mating_model,
        c_transmission=CTransmission(maternal=(Ctm_maternal1, Ctm_maternal2), paternal=(Ctm_paternal1, Ctm_paternal2)),
        inbreeding_mode=inbreeding_mode,
        inbreeding_depth=inbreeding_depth,
    )
    # Rebind the values normalization changed; everything else is unchanged.
    G_sim = params.G_sim
    assort1 = params.assort1
    assort2 = params.assort2
    R_mf = params.R_mf

    logger.info("Starting simulation: N=%d, G_ped=%d, seed=%d", N, G_ped, seed)
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    # Resolve per-generation C and E variance components.  A and C are
    # scalars (SimulationParams rejects maps) and E may be a per-gen dict.
    # These run for both mating models since reproduce() needs them.
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

    # Standard-model AM plan: per-gen assort resolution, rho_w, and the
    # |rho_w| < 1 + 4x4 PSD checks are validated up front inside build().
    # WF skips this entirely — no AM, no rho_w-driven cross-AM.
    plan: AssortmentPlan | None = None
    if mating_model == "standard":
        plan = AssortmentPlan.build(
            assort1=assort1,
            assort2=assort2,
            R_mf_user=R_mf,
            rA=rA,
            rC=rC,
            rE=rE,
            A1=A1,
            A2=A2,
            C1_per_gen=C1_per_gen,
            C2_per_gen=C2_per_gen,
            E1_per_gen=E1_per_gen,
            E2_per_gen=E2_per_gen,
            G_sim=G_sim,
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

    pheno = np.stack([a1, c1, e1, a2, c2, e2], axis=-1)  # column order: _A1 .. _E2

    # Simulate generations
    burnin = G_sim - G_ped

    # Pre-allocate pedigree arrays (avoids frame concat per generation)
    total_individuals = N * G_ped
    if total_individuals > np.iinfo(np.int32).max:
        raise ValueError(
            f"Total pedigree size {total_individuals:,} exceeds int32 max "
            f"({np.iinfo(np.int32).max:,}). Reduce N or G_ped."
        )
    ped_arrays = _init_pedigree_arrays(total_individuals, inbreeding=params.inbreeding.mode != "off")
    ped_offset = 0
    id_offset = 0
    household_offset = 0

    # Inbreeding (ADR 0033): iteration i's parents are iteration i - 1's
    # offspring, so the Mendelian scale reads the previous step's F.
    inbreeding: Inbreeding | None = None
    F_offspring: np.ndarray | None = None
    if params.inbreeding.mode != "off":
        from simace.simulation.inbreeding import Inbreeding

        inbreeding = Inbreeding(params.inbreeding, N, G_sim, mother_sorted=mating_model == "standard")
        F_offspring = np.zeros(N)

    for i in range(G_sim):
        t_gen = time.perf_counter()

        if mating_model == "standard":
            assert plan is not None  # built above under the same guard
            a1_i, a2_i, rho_w_i, R_mf_i = plan.for_generation(i)
            pairing: list[dict[str, Any]] = []
            try:
                parents, twins, household_ids = _mating_standard(
                    rng,
                    sex,
                    mating_lambda,
                    p_mztwin,
                    pheno=pheno,
                    assort1=a1_i,
                    assort2=a2_i,
                    rho_w=rho_w_i,
                    assort_matrix=R_mf_i,
                    pairing_log=pairing,
                )
            except ValueError as exc:
                raise ValueError(f"Mating iteration {i}: {exc}") from exc
        else:  # wright_fisher
            parents, twins, household_ids = _mating_wf(rng, sex, N, generation=i)
            pairing = [{"mode": "wright_fisher"}]
        if mating_log is not None:
            mating_log.extend({"iteration": i, "generation": i - burnin, **record} for record in pairing)
        t_mate = time.perf_counter()
        mendelian_scale = None
        if inbreeding is not None:
            assert F_offspring is not None
            mendelian_scale = np.sqrt(1.0 - (F_offspring[parents[:, 0]] + F_offspring[parents[:, 1]]) / 2)
            F_offspring = inbreeding.step(parents, twins)
        t_inbreeding = time.perf_counter()
        generation_clips: list[dict[str, Any]] = []
        try:
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
                params.c_transmission,
                generation_clips,
                mendelian_scale,
            )
        except ValueError as exc:
            raise ValueError(f"Generation {i}: {exc}") from exc
        if clip_log is not None:
            clip_log.extend({"generation": i - burnin, **clip} for clip in generation_clips)
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
                F=F_offspring,
            )
            household_offset += int(household_ids.max()) + 1
            id_offset += n
            ped_offset += n
        t_fill = time.perf_counter()

        # Per-generation data shape checkpoints
        fam_sizes = np.bincount(household_ids)
        logger.info(
            "Generation %d: %d twins, mean family size %.2f [mating=%.1fs, inbreeding=%.1fs, reproduce=%.1fs, "
            "fill=%.1fs, total=%.1fs]",
            i,
            len(twins) * 2,
            fam_sizes.mean(),
            t_mate - t_gen,
            t_inbreeding - t_mate,
            t_repro - t_inbreeding,
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


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line interface for running ACE simulations."""
    import json

    from simace.core.cli_base import add_logging_args, add_version_arg, float_or_generation_map, init_logging
    from simace.core.publish import publish

    parser = argparse.ArgumentParser(prog=prog, description="Run ACE pedigree simulation")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--N", type=int, default=1000, help="Founder population size")
    parser.add_argument("--G-ped", type=int, default=3, help="Number of pedigree generations")
    parser.add_argument("--G-sim", type=int, default=None, help="Number of burn-in generations (default: G_ped)")
    parser.add_argument(
        "--mating-model",
        type=str,
        default="standard",
        choices=("standard", "wright_fisher"),
        help="Mating model: standard (ZTP fecundity, paired) or wright_fisher (sex-structured idealized WF).",
    )
    parser.add_argument(
        "--mating-lambda", type=float, default=0.5, help="ZTP mating count lambda (standard model only)"
    )
    parser.add_argument("--p-mztwin", type=float, default=0.004, help="MZ twin rate per birth (standard model only)")
    parser.add_argument("--A1", type=float, default=0.5, help="Additive genetic variance for trait 1")
    parser.add_argument("--C1", type=float, default=0.2, help="Shared environment variance for trait 1")
    parser.add_argument(
        "--E1",
        type=float_or_generation_map,
        required=True,
        help="Unique environment variance for trait 1: a scalar or a JSON per-generation map",
    )
    parser.add_argument("--A2", type=float, default=0.5, help="Additive genetic variance for trait 2")
    parser.add_argument("--C2", type=float, default=0.2, help="Shared environment variance for trait 2")
    parser.add_argument(
        "--E2",
        type=float_or_generation_map,
        required=True,
        help="Unique environment variance for trait 2: a scalar or a JSON per-generation map",
    )
    parser.add_argument("--rA", type=float, default=0.5, help="Cross-trait genetic correlation")
    parser.add_argument("--rC", type=float, default=0.3, help="Cross-trait shared environment correlation")
    parser.add_argument("--rE", type=float, default=0.0, help="Cross-trait unique environment correlation")
    parser.add_argument(
        "--assort1",
        type=float_or_generation_map,
        default=0.0,
        help="Mate correlation on trait 1 liability: a scalar or a JSON per-generation map",
    )
    parser.add_argument(
        "--assort2",
        type=float_or_generation_map,
        default=0.0,
        help="Mate correlation on trait 2 liability: a scalar or a JSON per-generation map",
    )
    parser.add_argument(
        "--assort-matrix",
        type=json.loads,
        default=None,
        help="Optional 2x2 mate correlation matrix as JSON, e.g. '[[0.3, 0.05], [0.05, 0.15]]'",
    )
    for side, parent in (("maternal", "mother"), ("paternal", "father")):
        for t in (1, 2):
            parser.add_argument(
                f"--Ctm-{side}{t}",
                type=float,
                default=0.0,
                help=f"Share of the {parent}'s C{t} an offspring inherits, in [0, 1] (ADR 0025)",
            )
    parser.add_argument(
        "--inbreeding-mode",
        default="off",
        choices=INBREEDING_MODES,
        help="Scale the Mendelian residual by sqrt(1 - parental F): off, approx or exact (ADR 0033)",
    )
    parser.add_argument("--inbreeding-depth", type=int, default=DEFAULT_DEPTH, help="Generations approx traces exactly")
    parser.add_argument("--output-pedigree", required=True, help="Output pedigree parquet path")
    parser.add_argument("--output-mating", default=None, help="Output path of the pairing records (mating.yaml)")
    args = parser.parse_args(argv)

    init_logging(args)

    c_transmission = CTransmission(
        maternal=(args.Ctm_maternal1, args.Ctm_maternal2), paternal=(args.Ctm_paternal1, args.Ctm_paternal2)
    )
    clip_log: list[dict[str, Any]] | None = [] if c_transmission.active else None
    mating_log: list[dict[str, Any]] = []
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
        assort_matrix=args.assort_matrix,
        mating_model=args.mating_model,
        Ctm_maternal1=args.Ctm_maternal1,
        Ctm_paternal1=args.Ctm_paternal1,
        Ctm_maternal2=args.Ctm_maternal2,
        Ctm_paternal2=args.Ctm_paternal2,
        inbreeding_mode=args.inbreeding_mode,
        inbreeding_depth=args.inbreeding_depth,
        clip_log=clip_log,
        mating_log=mating_log,
    )

    inbreeding = None
    if args.inbreeding_mode != "off":
        inbreeding = resolve(args.inbreeding_mode, args.inbreeding_depth, args.N, args.G_sim or args.G_ped)
    with publish(args.output_pedigree) as (tmp,):
        write_pedigree(pedigree, tmp, c_transmission_clips=clip_log, inbreeding=inbreeding)
    if args.output_mating is not None:
        write_mating_log(mating_log, args.output_mating)
