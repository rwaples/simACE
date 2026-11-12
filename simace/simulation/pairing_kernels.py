"""Numba kernels for pairing mates: midparent values, Metropolis refinement, and k-d matching."""

from typing import TYPE_CHECKING

import numpy as np
from numba.extending import register_jitable

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, TypeVar

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
    from numba import njit


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


_midparent = njit(cache=True)(_midparent_python)


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


# Plain Python when called from Python, so a _python kernel stays deterministic
# under its own np.random.seed; compiled inline when a jitted kernel calls it.
_draw_pairs = register_jitable(_draw_pairs_python)


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


_metropolis_full = njit(cache=True, fastmath=True)(_metropolis_full_python)


def _has_couple_python(fathers, rows_ptr, rows, female, father):
    """Whether ``female`` (a CSR group of ``rows``) is already paired with ``father`` in one of her rows."""
    for k in range(rows_ptr[female], rows_ptr[female + 1]):  # noqa: SIM110  # njit compiles no any() over a generator
        if fathers[rows[k]] == father:
            return True
    return False


_has_couple = njit(cache=True)(_has_couple_python)


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


_metropolis_single = njit(cache=True, fastmath=True)(_metropolis_single_python)


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


_select = njit(cache=True)(_select_python)


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


_kd_match = njit(cache=True)(_kd_match_python)
