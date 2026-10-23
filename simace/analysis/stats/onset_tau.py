"""Onset τ of the frailty model's raw onsets, and its realized expectation (ADR 0027).

Onset τ is Kendall's τ-a between the raw onsets of the two members of every
relationship pair of one type, or between a person's two raw onsets
(within-person). Given the linear predictors η, the frailty model's onsets
order as independent exponentials ``V_i = E_i / λ_i`` with ``λ_i = e^{η_i}``,
so every comparison the statistic counts has a closed-form expectation:
``tanh((η_x − η_x')/2) · tanh((η_y − η_y')/2)`` when the two entries have four
distinct members, and a Plackett–Luce sum when they share a person. The
average over all comparisons is the realized onset-τ expectation, which does
not depend on the baseline hazard.

A cell's entries are oriented pairs ``(x_e, y_e)``: both orientations of a
symmetric type's pair (each entry's comparison with its mirror is excluded),
the (offspring, parent) orientation of ``MO``/``FO``, and ``(p, p)`` on
(``t1``, ``t2``) within-person. The SE treats η as fixed: only the onset draws
are random, independent across people, so influences are clustered by the
connected components of the cell's pair graph.

:func:`compute_onset_tau` runs on the phenotype stage's float64 output, which
:func:`simace.analysis.validate.onset_tau.run_onset_tau` regenerates for a
finished rep and then scores (``simace validate``).
"""

from __future__ import annotations

__all__ = ["CELL_CATEGORIES", "compute_onset_tau", "onset_tau_cell", "tau_oracle"]

import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

logger = logging.getLogger(__name__)

CELL_CATEGORIES: tuple[str, ...] = ("MZ", "FS", "MHS", "PHS", "MO", "FO")
_SYMMETRIC = frozenset({"MZ", "FS", "MHS", "PHS"})

# Cells with at most this many entries sum their expectation directly.
EXACT_MAX_ENTRIES = 2000
# Taylor grid: start, cap, and the share of the SE its error bound may take.
GRID_START = 64
GRID_MAX = 1024
GRID_BOUND_SE_SHARE = 0.01
# Above this many shared-person comparisons the cell's target is not computed.
SHARED_WORKLOAD_MAX = 5e8
# A cell over this many entries is scored on a seeded subset of whole components
# of about this size, which bounds the check's cost at any population size.
ENTRY_CAP = 100_000

# Lagrange remainder of the second-order expansion of k(s)·k(t), k(t) = tanh(t/2),
# with |Δs|, |Δt| <= h: (2·M3 + 6·M2·M1)/6 · h³, where M1, M2, M3 bound |k'|, |k''|, |k'''|.
_M1, _M2, _M3 = 0.5, 1.0 / (3.0 * math.sqrt(3.0)), 0.25
TAYLOR_C3 = (2.0 * _M3 + 6.0 * _M2 * _M1) / 6.0

_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite_e.hermegauss(150)
_HERMITE_WEIGHTS = _HERMITE_WEIGHTS / _HERMITE_WEIGHTS.sum()
_HERMITE_MAX_SCALE = 3.0


def tau_oracle(s1: float, s2: float, r: float) -> float:
    """Onset τ of bivariate normal linear predictors: ``E[tanh(s1·X/2)·tanh(s2·Y/2)]``.

    ``(X, Y)`` is bivariate normal with variances 2 and correlation ``r``: the
    differences of two independent pairs' standardized predictors. ``s1`` and
    ``s2`` are each side's standard deviation of η. The limits are 0 at
    ``s = 0`` and ``(2/π)·arcsin(r)`` as ``s1, s2 → ∞``.
    """
    from scipy import integrate

    if s1 == 0.0 or s2 == 0.0:
        return 0.0
    r = min(1.0, max(-1.0, r))
    q = math.sqrt(max(0.0, 1.0 - r * r))
    root2 = math.sqrt(2.0)
    norm = 1.0 / math.sqrt(2.0 * math.pi)

    if max(abs(s1), abs(s2)) <= _HERMITE_MAX_SCALE:
        # A tensor Gauss–Hermite rule: within 1e-8 of the adaptive quadrature up to s = 3.
        u, v = np.meshgrid(_HERMITE_NODES, _HERMITE_NODES, indexing="ij")
        integrand = np.tanh(s1 * u / root2) * np.tanh(s2 * (r * u + q * v) / root2)
        return float(_HERMITE_WEIGHTS @ integrand @ _HERMITE_WEIGHTS)

    if q < 1e-12:
        sign = 1.0 if r > 0 else -1.0

        def edge(u: float) -> float:
            return math.tanh(s1 * u / root2) * math.tanh(sign * s2 * u / root2) * math.exp(-u * u / 2) * norm

        return integrate.quad(edge, -math.inf, math.inf, epsabs=1e-13, limit=400)[0]

    def inner(u: float) -> float:
        def f(v: float) -> float:
            return math.tanh(s2 * (r * u + q * v) / root2) * math.exp(-v * v / 2)

        return integrate.quad(f, -math.inf, math.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0] * norm

    def outer(u: float) -> float:
        return math.tanh(s1 * u / root2) * inner(u) * math.exp(-u * u / 2) * norm

    return integrate.quad(outer, -math.inf, math.inf, epsabs=1e-12, epsrel=1e-12, limit=400)[0]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _bit_add(tree: np.ndarray, i: int) -> None:
    while i < len(tree):
        tree[i] += 1
        i += i & -i


@njit(cache=True)
def _bit_sum(tree: np.ndarray, i: int) -> int:
    s = 0
    while i > 0:
        s += tree[i]
        i -= i & -i
    return s


def _dense_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The stable sort order of *values* and their 1-based dense ranks (equal values share a rank)."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.cumsum(np.r_[True, ordered[1:] != ordered[:-1]])
    return order, ranks


@njit(cache=True)
def _concordance_kernel(order: np.ndarray, rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """Concordance rows from x's sort order and both sides' dense ranks, by a Fenwick tree over y-ranks."""
    n = len(order)
    c = np.zeros(n, np.int64)
    tree = np.zeros(ry.max() + 1 if n else 1, np.int64)
    # Partners with strictly smaller x, then partners with strictly larger x.
    k = 0
    seen = 0
    while k < n:
        j = k
        while j < n and rx[order[j]] == rx[order[k]]:
            j += 1
        for t in range(k, j):
            e = order[t]
            below = _bit_sum(tree, ry[e] - 1)
            above = seen - _bit_sum(tree, ry[e])
            c[e] += below - above
        for t in range(k, j):
            _bit_add(tree, ry[order[t]])
        seen += j - k
        k = j
    tree[:] = 0
    k = n - 1
    seen = 0
    while k >= 0:
        j = k
        while j >= 0 and rx[order[j]] == rx[order[k]]:
            j -= 1
        for t in range(j + 1, k + 1):
            e = order[t]
            below = _bit_sum(tree, ry[e] - 1)
            above = seen - _bit_sum(tree, ry[e])
            c[e] += above - below
        for t in range(j + 1, k + 1):
            _bit_add(tree, ry[order[t]])
        seen += k - j
        k = j
    return c


@njit(cache=True)
def _rates(eta: np.ndarray) -> np.ndarray:
    """``e^η`` up to one common factor, centred on η's midrange so no rate over- or underflows.

    The kernels use only ratios of rates (tanh as ``(λ_a − λ_b)/(λ_a + λ_b)``
    and the Plackett–Luce sums), which a common factor leaves unchanged; a
    span of η up to about 1400 stays finite.
    """
    return np.exp(eta - 0.5 * (eta.max() + eta.min()))


@njit(cache=True, error_model="numpy")  # 0/0 is NaN (all rates underflowed): the cell guard reports it
def _exact_rowsum(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``Σ_f tanh((u[e] − u[f])/2) · tanh((v[e] − v[f])/2)`` for every e, directly.

    The term is symmetric in (e, f), so each pair is evaluated once, through
    ``tanh((a − b)/2) = (e^a − e^b)/(e^a + e^b)``, with each side's exponent
    centred (:func:`_rates`).
    """
    n = len(u)
    lu, lv = _rates(u), _rates(v)
    out = np.zeros(n)
    for e in range(n):
        for f in range(e + 1, n):
            term = (lu[e] - lu[f]) / (lu[e] + lu[f]) * (lv[e] - lv[f]) / (lv[e] + lv[f])
            out[e] += term
            out[f] += term
    return out


@njit(cache=True)
def _middle(lam_mid: float, lam_a: float, lam_b: float) -> float:
    """``E[sign(V_m − V_a) · sign(V_b − V_m)] = 2·P(V_m is the middle one) − 1`` for independent exponentials.

    Plackett–Luce: ``P(a first, m second) = λ_a/T · λ_m/(λ_m + λ_b)``, with
    ``T = λ_m + λ_a + λ_b``; likewise with b first. Each second-place
    denominator is summed from the two remaining rates, never as ``T − λ_a``,
    which cancels to 0 when one rate dominates (``beta_sex = 40``). A
    remaining pair of zero rates means the middle person cannot come second.
    """
    total = lam_mid + lam_a + lam_b
    after_a = lam_mid + lam_b
    after_b = lam_mid + lam_a
    middle = 0.0
    if after_a > 0.0:
        middle += lam_a / total * lam_mid / after_a
    if after_b > 0.0:
        middle += lam_b / total * lam_mid / after_b
    return 2.0 * middle - 1.0


@njit(cache=True, error_model="numpy")  # 0/0 is NaN (all rates underflowed): the cell guard reports it
def _shared_corrections(
    X: np.ndarray, Y: np.ndarray, lam: np.ndarray, ptr: np.ndarray, ent: np.ndarray, mirror: np.ndarray
) -> np.ndarray:
    """Per entry, Σ over partners sharing a person of (exact expectation − tanh product).

    Each comparison is visited from row e once for every person it shares
    with f, through that person's incidence list, and weighted by
    1 / (people shared). Work is ``Σ_p deg_p²``. With ``λ = e^η``,
    ``tanh((η_a − η_b)/2) = (λ_a − λ_b)/(λ_a + λ_b)``, so no tanh is evaluated.
    """
    corr = np.zeros(len(X))
    for p in range(len(ptr) - 1):
        lo, hi = ptr[p], ptr[p + 1]
        for i in range(lo, hi):
            e = ent[i]
            x, y = X[e], Y[e]
            for j in range(lo, hi):
                f = ent[j]
                if f == e or f == mirror[e]:
                    continue
                xf, yf = X[f], Y[f]
                n_shared = (x == xf) + (x == yf) + (y == xf) + (y == yf)
                approx = (lam[x] - lam[xf]) / (lam[x] + lam[xf]) * (lam[y] - lam[yf]) / (lam[y] + lam[yf])
                if x == xf or y == yf:
                    exact = 0.0
                elif x == yf and y == xf:  # the reversed pair: sign(T_x − T_y) · sign(T_y − T_x)
                    exact = -1.0
                elif x == yf:  # sign(T_x − T_xf) · sign(T_y − T_x): x in the middle
                    exact = _middle(lam[x], lam[xf], lam[y])
                else:  # y == xf: sign(T_x − T_y) · sign(T_y − T_yf): y in the middle
                    exact = _middle(lam[y], lam[yf], lam[x])
                corr[e] += (exact - approx) / n_shared
    return corr


def _taylor_rowsum(u: np.ndarray, v: np.ndarray, grid: int) -> tuple[np.ndarray, float]:
    """``Σ_f tanh((u_e − u_f)/2)·tanh((v_e − v_f)/2)`` for every e, and the per-term error bound.

    Entries are binned on one ``grid × grid`` lattice of width h. Each term is
    expanded to second order in the query's and the source's offsets from
    their bin centres, using per-bin moments of the source offsets, so every
    term's remainder is at most ``TAYLOR_C3 · h³``.
    """
    lo = float(min(u.min(), v.min()))
    span = max(float(max(u.max(), v.max())) - lo, 1e-12)
    h = span / grid
    gu = np.minimum(((u - lo) / h).astype(np.int64), grid - 1)
    gv = np.minimum(((v - lo) / h).astype(np.int64), grid - 1)
    centres = lo + h * (np.arange(grid) + 0.5)
    du = u - centres[gu]
    dv = v - centres[gv]
    flat = gu * grid + gv

    def hist(weights: np.ndarray | None) -> np.ndarray:
        return np.bincount(flat, weights=weights, minlength=grid * grid).reshape(grid, grid)

    count, a1, b1 = hist(None), hist(du), hist(dv)
    a2, b2, ab = hist(du * du), hist(dv * dv), hist(du * dv)
    diff = centres[:, None] - centres[None, :]
    th = np.tanh(diff / 2)
    k0 = th
    k1 = 0.5 * (1.0 - th * th)
    k2 = -0.5 * th * (1.0 - th * th)
    del diff, th

    # Ten left products K_d · moments, then ten right products grouped by their right factor.
    # (FFT convolution of these Toeplitz products measured 1.6x slower than BLAS at grid <= 512.)
    k0n, k1n, k2n = k0 @ count, k1 @ count, k2 @ count
    k1a1, k2a1, k2a2 = k1 @ a1, k2 @ a1, k2 @ a2
    k0b1, k1b1, k1ab, k0b2 = k0 @ b1, k1 @ b1, k1 @ ab, k0 @ b2

    def at(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (left @ right.T)[gu, gv]

    out = at(k0n - k1a1 + 0.5 * k2a2, k0) + at(k1ab - k0b1, k1) + at(0.5 * k0b2, k2)
    out += du * (at(k1n - k2a1, k0) - at(k1b1, k1))
    out += dv * (at(k0n - k1a1, k1) - at(k0b1, k2))
    out += 0.5 * du * du * at(k2n, k0) + du * dv * at(k1n, k1) + 0.5 * dv * dv * at(k0n, k2)
    return out, TAYLOR_C3 * h**3


def _tied_comparisons(ranks: np.ndarray, persons: np.ndarray) -> int:
    """Pairs of entries whose values are equal (equal dense *ranks*) while their people differ."""
    runs = np.bincount(ranks)
    equal_value = int((runs * (runs - 1) // 2).sum())
    per_person = np.bincount(persons)
    same_person = int((per_person * (per_person - 1) // 2).sum())
    return equal_value - same_person


def _incidence(X: np.ndarray, Y: np.ndarray, n_people: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR lists of the entries containing each person."""
    ent = np.arange(len(X))
    other = X != Y
    persons = np.concatenate([X, Y[other]])
    entries = np.concatenate([ent, ent[other]])
    order = np.argsort(persons, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(persons, minlength=n_people))]
    return ptr, entries[order]


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


def _cell_entries(
    first: np.ndarray, second: np.ndarray, symmetric: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Entries ``(X, Y)``, each entry's mirror (-1 for none), and the comparisons per entry."""
    n_pairs = len(first)
    if symmetric:
        X, Y = np.concatenate([first, second]), np.concatenate([second, first])
        mirror = np.concatenate([np.arange(n_pairs, 2 * n_pairs), np.arange(n_pairs)])
        return X, Y, mirror, len(X) - 2
    return first, second, np.full(n_pairs, -1, dtype=np.int64), n_pairs - 1


def _ranked_rows(
    t_first: np.ndarray, t_second: np.ndarray, symmetric: bool, per: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed rows and both sides' dense ranks, which the tie count reuses."""
    order, rx = _dense_ranks(t_first)
    ry = _dense_ranks(t_second)[1]
    rows = _concordance_kernel(order, rx, ry).astype(np.float64)
    if symmetric:
        # The mirror comparison is −1 when the pair's onsets differ and 0 when they are tied.
        rows += t_first != t_second
    return rows / per, rx, ry


def _shared_terms(
    ex: np.ndarray, X: np.ndarray, Y: np.ndarray, mirror: np.ndarray, n_people: int
) -> tuple[np.ndarray | None, int]:
    """Per-entry corrections for comparisons that share a person, and their workload ``Σ_p deg_p²``."""
    ptr, ent = _incidence(X, Y, n_people)
    workload = int((np.diff(ptr) ** 2).sum())
    if workload > SHARED_WORKLOAD_MAX:
        return None, workload
    return _shared_corrections(X, Y, _rates(ex), ptr, ent, mirror), workload


def _expected_rows(u: np.ndarray, v: np.ndarray, extra: np.ndarray, per: int, grid: int) -> tuple[np.ndarray, float]:
    """``E[observed row | η]`` per entry and its error bound; *grid* 0 sums directly.

    *extra* holds the exact adjustments to the all-entries tanh sum: the
    mirror term and the shared-person corrections.
    """
    if grid == 0:
        sums, bound = _exact_rowsum(u, v), 0.0
    else:
        sums, per_term = _taylor_rowsum(u, v, grid)
        bound = per_term * len(u) / per
    return (sums + extra) / per, bound


def _clustered_se(observed: np.ndarray, expected: np.ndarray, comp: np.ndarray) -> float:
    """SE of τ̂ given η: influences ``(2/N)·(observed − expected)`` summed per component, squared, added."""
    influence = 2.0 / len(observed) * (observed - expected)
    return float(math.sqrt((np.bincount(comp, weights=influence) ** 2).sum()))


def onset_tau_cell(
    tx: np.ndarray,
    ty: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    component: np.ndarray,
    *,
    symmetric: bool,
    within_person: bool = False,
) -> dict[str, Any]:
    """Observed onset τ of one cell, its realized expectation, the error bound and the SE.

    Args:
        tx: Per-person raw onsets on the first side (float64, ``+inf`` for a null).
        ty: Per-person raw onsets on the second side; the same array as *tx*
            for a relationship cell.
        ex: Per-person linear predictors matching *tx*.
        ey: Per-person linear predictors matching *ty*.
        first: The cell's first members, one per relationship pair.
        second: The cell's second members.
        component: Per-person connected-component label of the cell's pair graph.
        symmetric: Whether each pair enters in both orientations.
        within_person: Whether *first* and *second* are the same people on two
            traits; then no two entries share a person.

    Returns:
        The cell's statistics. ``expected`` and ``se`` are None, with a
        ``note``, when they cannot be computed.
    """
    X, Y, mirror, per = _cell_entries(first, second, symmetric)
    n = len(X)
    comp = component[X]
    out: dict[str, Any] = {"n_pairs": len(first), "n_entries": n}
    _component_summary(out, comp, component, X, Y)
    if per < 1:
        out.update(observed=None, expected=None, se=None, note="fewer than two comparable entries")
        return out

    u, v = ex[X], ey[Y]
    out["s1"], out["s2"] = float(u.std()), float(v.std())
    out["r_eta"] = float(np.corrcoef(u, v)[0, 1]) if out["s1"] > 0 and out["s2"] > 0 else None
    observed, rx, ry = _ranked_rows(tx[X], ty[Y], symmetric, per)
    out["observed"] = float(observed.mean())
    ties = _tied_comparisons(rx, X) + _tied_comparisons(ry, Y)
    out["tie_share"] = ties / (n * per / 2)

    if within_person:
        shared, workload = np.zeros(n), 0
    else:
        shared, workload = _shared_terms(ex, X, Y, mirror, len(tx))
    out["shared_workload"] = workload
    if shared is None:
        out.update(expected=None, se=None, note=f"shared-person workload {workload:.3g} > {SHARED_WORKLOAD_MAX:.0e}")
        return out
    extra = shared + (np.tanh((u - v) / 2) ** 2 if symmetric else 0.0)

    grid = 0 if n <= EXACT_MAX_ENTRIES else GRID_START
    expected, bound = _expected_rows(u, v, extra, per, grid)
    se = _clustered_se(observed, expected, comp)
    while grid and bound > GRID_BOUND_SE_SHARE * se and grid < GRID_MAX:
        # The bound falls as grid⁻³: jump to the power of two the pilot SE asks for.
        needed = grid * (bound / (GRID_BOUND_SE_SHARE * se)) ** (1 / 3)
        grid = min(GRID_MAX, 2 ** math.ceil(math.log2(needed)))
        expected, bound = _expected_rows(u, v, extra, per, grid)
        se = _clustered_se(observed, expected, comp)
    if not (np.isfinite(expected).all() and math.isfinite(se)):
        out.update(grid=grid, expected=None, se=None, note="non-finite expectation (linear predictors out of range)")
        return out
    out.update(grid=grid, expected=float(expected.mean()), numerical_bound=bound, se=se)
    return out


def _component_summary(
    out: dict[str, Any], comp: np.ndarray, component: np.ndarray, X: np.ndarray, Y: np.ndarray
) -> None:
    """Entry-weighted component counts: ``n_eff = (Σ w_c)² / Σ w_c²`` with ``w_c`` the entries in c."""
    entries = np.bincount(comp)
    entries = entries[entries > 0].astype(np.float64)
    in_cell = np.zeros(len(component), dtype=bool)
    in_cell[X] = True
    in_cell[Y] = True
    people = np.bincount(component[in_cell])
    out["n_people"] = int(in_cell.sum())
    out["n_components"] = len(entries)
    out["n_eff"] = float(entries.sum() ** 2 / (entries**2).sum()) if len(entries) else 0.0
    out["max_component_share"] = float(entries.max() / entries.sum()) if len(entries) else None
    out["max_component_people"] = int(people.max()) if len(entries) else 0


# ---------------------------------------------------------------------------
# The phenotyped population
# ---------------------------------------------------------------------------


def _with_oracle(cell: dict[str, Any]) -> dict[str, Any]:
    """Add ``tau_oracle(s1, s2, r_eta)``, the parametric value for the cell's η moments (information only)."""
    r = cell.get("r_eta")
    cell["tau_oracle"] = tau_oracle(cell["s1"], cell["s2"], r) if r is not None else None
    return cell


def _linear_predictor(
    model: Any, liability: np.ndarray, sex: np.ndarray, generation: np.ndarray, standardize: Any
) -> np.ndarray:
    """η = b·(L − m) + beta_sex·sex, with the trait's resolved standardization (the phenotype code's own)."""
    from simace.phenotype.hazards import resolve_hazard_mode, standardize_beta

    mode = resolve_hazard_mode(standardize, model.standardize_hazard)
    mean, scaled = standardize_beta(liability, model.beta, mode, generation)
    return scaled * (liability - mean) + model.beta_sex * sex


@njit(cache=True)
def _component_labels(first: np.ndarray, second: np.ndarray, n_people: int) -> np.ndarray:
    """Each person's connected-component root in the graph of the given pairs (union-find, path halving)."""
    parent = np.arange(n_people, dtype=np.int32)
    for k in range(len(first)):
        a, b = first[k], second[k]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            parent[max(a, b)] = min(a, b)
    for i in range(n_people):
        root = i
        while parent[root] != root:
            root = parent[root]
        parent[i] = root
    return parent


def _cap_components(
    first: np.ndarray, second: np.ndarray, component: np.ndarray, n_entries: int, *, seed: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the pairs of a seeded random subset of whole components, about ``ENTRY_CAP`` entries in all.

    Given η, components are independent of one another, so the kept set's
    target and SE are exact for the kept set. Cells at or under the cap are
    returned whole.
    """
    if n_entries <= ENTRY_CAP:
        return first, second
    keep = np.random.default_rng(seed).random(int(component.max()) + 1) < ENTRY_CAP / n_entries
    mask = keep[component[first]]
    return first[mask], second[mask]


def _lineage_roots(parent_row: np.ndarray) -> np.ndarray:
    """Each person's topmost ancestor along *parent_row* links (-1: no parent row), one generation a step."""
    root = np.arange(len(parent_row))
    while True:
        up = parent_row[root]
        if not (up >= 0).any():
            return root
        root = np.where(up >= 0, up, root)


def _family_units(
    ids: np.ndarray, mother: np.ndarray, father: np.ndarray, twin: np.ndarray
) -> dict[str, tuple[np.ndarray, int]]:
    """Per category, each person's sampling unit (-1: in no pair) and the category's expected entries.

    A unit always holds whole components of the category's pair graph, so
    sampling units samples whole components: the children of one mother for
    ``FS``, ``MZ`` (twins only) and ``MHS``, of one father for ``PHS``, and a
    maternal or paternal lineage (labelled by its topmost row) for ``MO`` and
    ``FO``. The expected entries only set the sampling fraction; which pairs
    exist is still decided by pedigree-graph.
    """
    n = len(ids)
    row_of = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    row_of[ids] = np.arange(n)

    def parent_rows(parent: np.ndarray) -> np.ndarray:
        known = (parent >= 0) & (parent < len(row_of))
        out = np.full(n, -1, dtype=np.int64)
        out[known] = row_of[parent[known]]
        return out

    def ordered_pairs(group: np.ndarray) -> int:
        sizes = np.bincount(group[group >= 0])
        return int((sizes * (sizes - 1)).sum())

    # Ordered full-sib pairs from one sort of the (mother, father) keys of children with both known.
    both = (mother >= 0) & (father >= 0)
    keys = np.sort(mother[both].astype(np.int64) * (int(ids.max()) + 1) + father[both])
    runs = np.diff(np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True]))
    full_sib = int((runs * (runs - 1)).sum())

    mother_unit = np.where(mother >= 0, mother, -1).astype(np.int64)
    father_unit = np.where(father >= 0, father, -1).astype(np.int64)
    units: dict[str, tuple[np.ndarray, int]] = {
        # Co-twins are MZ, not FS: drop their two ordered entries per pair.
        "FS": (mother_unit, full_sib - int(((twin >= 0) & both).sum())),
        "MZ": (np.where(twin >= 0, mother_unit, -1), int((twin >= 0).sum())),
        # Ordered same-parent pairs, less those that share the other parent too (full sibs).
        "MHS": (mother_unit, ordered_pairs(mother_unit) - full_sib),
        "PHS": (father_unit, ordered_pairs(father_unit) - full_sib),
    }
    for code, parent in (("MO", mother), ("FO", father)):
        up = parent_rows(parent)
        units[code] = (_lineage_roots(up), int((up >= 0).sum()))
    return units


def _unit_pairs(
    family: pl.DataFrame, code: str, unit: np.ndarray, fraction: float, *, seed: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """*code*'s pairs among a seeded sample of its units, in population rows.

    The kept units' people form a subgraph (parents outside it stay as
    external ids, which is how pedigree-graph already identifies siblings);
    a unit holds whole components, so the subgraph has exactly the
    category's pairs among them.
    """
    from pedigree_graph import PedigreeGraph

    keep = np.random.default_rng(seed).random(int(unit.max()) + 1) < fraction
    rows = np.flatnonzero((unit >= 0) & keep[np.maximum(unit, 0)])
    block = PedigreeGraph.from_frame(family[rows]).relationship_pairs(categories=[code])[code]
    return rows[block.first_rows], rows[block.second_rows]


def _compact(
    first: np.ndarray, second: np.ndarray, component: np.ndarray, n_people: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Renumber a cell's people ``0..k-1`` so its arrays scale with the cell, not the population.

    Returns the people's population rows, the pairs in the new numbering
    (int64), and each person's dense component code.
    """
    present = np.zeros(n_people, dtype=bool)
    present[first] = True
    present[second] = True
    people = np.flatnonzero(present)
    index = np.full(n_people, -1, dtype=np.int64)
    index[people] = np.arange(len(people))
    return people, index[first], index[second], _dense_ranks(component[people])[1] - 1


def compute_onset_tau(
    raw: pl.DataFrame, pedigree: pl.DataFrame, phenotype: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    """Onset-τ cells of the phenotyped population, for each frailty trait.

    Args:
        raw: The phenotype stage's output (``id``, ``t1``, ``t2``), before
            Parquet normalization, so onsets keep float64.
        pedigree: The recorded pedigree (``id``, ``mother``, ``father``,
            ``twin``, ``sex``, ``generation``, ``liability{t}``).
        phenotype: ``run_phenotype``'s keyword arguments other than ``seed``.
        seed: The rep seed; it picks the kept components of a cell over
            ``ENTRY_CAP`` entries.

    Returns:
        ``{"cells": {"trait{t}": {code: cell}, "within_person": cell},
        "skipped_traits": {"trait{t}": reason}, "entry_cap": ENTRY_CAP}``,
        where each cell is :func:`onset_tau_cell`'s result plus its
        ``sampled_fraction`` and ``n_entries_population``; ``within_person``
        only when both traits are frailty.
    """
    from pedigree_graph import PedigreeGraph

    from simace.phenotype.models import MODELS

    start = time.perf_counter()
    columns = ["id", "mother", "father", "twin", "sex", "generation", "liability1", "liability2"]
    rows = raw.select("id", "t1", "t2").join(pedigree.select(columns), on="id", how="left", maintain_order="left")
    sex = rows["sex"].to_numpy().astype(np.float64)
    generation = rows["generation"].to_numpy()
    onsets: dict[int, np.ndarray] = {}
    eta: dict[int, np.ndarray] = {}
    result: dict[str, Any] = {"cells": {}, "skipped_traits": {}}
    for t in (1, 2):
        name = phenotype[f"phenotype_model{t}"]
        if name != "frailty":
            result["skipped_traits"][f"trait{t}"] = f"model {name} is not frailty"
            continue
        model = MODELS[name].from_config(
            {
                f"phenotype_params{t}": phenotype[f"phenotype_params{t}"],
                f"beta{t}": phenotype[f"beta{t}"],
                f"beta_sex{t}": phenotype[f"beta_sex{t}"],
            },
            t,
        )
        liability = rows[f"liability{t}"].to_numpy().astype(np.float64)
        eta[t] = _linear_predictor(model, liability, sex, generation, phenotype["standardize"])
        onsets[t] = rows[f"t{t}"].fill_null(math.inf).to_numpy().astype(np.float64)
    if not eta:
        return result

    n_people = len(rows)
    result["entry_cap"] = ENTRY_CAP
    family = rows.select("id", "mother", "father", "twin", "sex")
    del rows, sex, generation
    units = _family_units(*(family[c].to_numpy() for c in ("id", "mother", "father", "twin")))
    expected_entries = {code: units[code][1] for code in CELL_CATEGORIES}
    # A category expected over the cap is extracted only from a seeded sample of its units, on
    # a subgraph of their rows; pedigree-graph still decides which pairs exist.
    unit_fraction = {code: min(1.0, 1.25 * ENTRY_CAP / max(expected_entries[code], 1)) for code in CELL_CATEGORIES}
    whole = [code for code in CELL_CATEGORIES if unit_fraction[code] == 1.0]
    # The categories extracted whole share one engine pass (four times faster than six passes
    # at 100k founders): over everyone when all six are whole, else over the union of their
    # units' rows, which holds every pair of each. The result owns all its blocks until dropped.
    pairs, whole_rows = None, None
    if whole:
        if len(whole) < len(CELL_CATEGORIES):
            in_units = np.zeros(n_people, dtype=bool)
            for code in whole:
                in_units |= units[code][0] >= 0
            whole_rows = np.flatnonzero(in_units)
        frame = family if whole_rows is None else family[whole_rows]
        pairs = PedigreeGraph.from_frame(frame).relationship_pairs(categories=whole)
    for index, code in enumerate(CELL_CATEGORIES):
        if pairs is not None and code in whole:
            first, second = pairs[code].first_rows, pairs[code].second_rows  # int32; only the cap copies
            if whole_rows is not None:
                first, second = whole_rows[first], whole_rows[second]
        else:
            first, second = _unit_pairs(family, code, units[code][0], unit_fraction[code], seed=(seed, index, 0))
        symmetric = code in _SYMMETRIC
        component = _component_labels(first, second, n_people)
        n_entries = len(first) * (2 if symmetric else 1)
        fraction = unit_fraction[code] * min(1.0, ENTRY_CAP / max(n_entries, 1))
        first, second = _cap_components(first, second, component, n_entries, seed=(seed, index))
        people, first, second, component = _compact(first, second, component, n_people)
        sampling = {
            "sampled_fraction": fraction,
            # Exact when every pair was extracted (the cap alone may still have applied).
            "n_entries_population": n_entries if unit_fraction[code] == 1.0 else expected_entries[code],
        }
        for t in eta:
            cell_start = time.perf_counter()
            tx, ex = onsets[t][people], eta[t][people]
            cell = onset_tau_cell(tx, tx, ex, ex, first, second, component, symmetric=symmetric)
            result["cells"].setdefault(f"trait{t}", {})[code] = _with_oracle({**sampling, **cell})
            logger.info("onset tau trait%d %s: %d pairs, %.2fs", t, code, len(first), time.perf_counter() - cell_start)
    del pairs, first, second, component, family
    if len(eta) == 2:
        everyone = np.arange(n_people)
        kept, _ = _cap_components(everyone, everyone, everyone, n_people, seed=(seed, len(CELL_CATEGORIES)))
        own = np.arange(len(kept))
        cell = onset_tau_cell(
            onsets[1][kept],
            onsets[2][kept],
            eta[1][kept],
            eta[2][kept],
            own,
            own,
            own,
            symmetric=False,
            within_person=True,
        )
        result["cells"]["within_person"] = _with_oracle(
            {"sampled_fraction": min(1.0, ENTRY_CAP / n_people), "n_entries_population": n_people, **cell}
        )
    logger.info("onset tau: %d people, %.2fs", n_people, time.perf_counter() - start)
    return result
