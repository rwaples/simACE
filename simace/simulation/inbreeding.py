"""Per-generation inbreeding coefficients for the simulate loop (ADR 0033).

:class:`Inbreeding` is stepped once per mating iteration with that iteration's
``parents`` (rows of the previous iteration's offspring) and MZ ``twins``, and
returns the offspring's inbreeding coefficients over the whole simulated
ancestry, founders at F = 0. The engine is fixed by an
:class:`~simace.simulation.inbreeding_plan.InbreedingPlan`:

- ``densep``: the dense kinship recurrence ``K_t = P K_{t-1} P'`` in float32,
  two N x N arrays (``notes/issue-76-recommendation/densep.py``).
- ``walkd2``: per distinct mating, breadth-first gene-contribution walks of
  mother and father over the whole history, ``F = 0.5 sum_k c_m c_f D_k``
  (``notes/issue-76-recommendation/walkd2.py`` at ``L = G_sim``, so no
  boundary closure). History is int32 parents and float64 D and F.

Every kernel is bitwise thread-invariant: parallel loops write disjoint
outputs and no reduction depends on the thread count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

if TYPE_CHECKING:
    from simace.simulation.inbreeding_plan import InbreedingPlan

    prange = range
else:
    from numba import prange

__all__ = ["Inbreeding"]


# ---------------------------------------------------------------- densep


@numba.njit(parallel=True, cache=True)
def _rows(K, m, f, Kp):
    N = K.shape[0]
    for i in prange(N):
        a, b = m[i], f[i]
        for j in range(N):
            Kp[i, j] = K[a, j] + K[b, j]


@numba.njit(parallel=True, cache=True)
def _cols(Kp, m, f, K):
    N = K.shape[0]
    for i in prange(N):
        for j in range(N):
            K[i, j] = (Kp[i, m[j]] + Kp[i, f[j]]) * 0.25


@numba.njit(parallel=True, cache=True)
def _diag_F(K, m, f, F):
    for i in prange(K.shape[0]):
        F[i] = K[m[i], f[i]]


class _Densep:
    """Dense float32 kinship of the current generation.

    Each sum of two values is rounded once to float32 and ``* 0.25`` is exact,
    so F matches the float64 recurrence to float32 rounding. F of an offspring
    is its parents' kinship; MZ co-twins' kinship is copied from the diagonal.
    """

    def __init__(self, N: int) -> None:
        self.K = np.zeros((N, N), np.float32)
        self.K[np.diag_indices(N)] = 0.5
        self.Kp = np.empty_like(self.K)

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        K = self.K
        m = np.ascontiguousarray(parents[:, 0], np.int64)
        f = np.ascontiguousarray(parents[:, 1], np.int64)
        Fk = np.empty(len(m), K.dtype)
        _diag_F(K, m, f, Fk)
        F = Fk.astype(np.float64)
        _rows(K, m, f, self.Kp)
        _cols(self.Kp, m, f, K)
        K[np.diag_indices(len(m))] = 0.5 * (1 + F)
        if len(twins):
            a, b = twins[:, 0], twins[:, 1]
            K[a, b] = K[b, a] = K[a, a]
        return F


# ---------------------------------------------------------------- walkd2


@numba.njit(cache=True)
def _bfs(root, par, stopkey, sc, q):
    """Breadth-first gene-contribution coefficients of root into sc, distinct ancestors in q.

    Generations are discrete, so the queue is ordered by generation and an
    entry is final before it is expanded. Founders (``key < stopkey``) are not
    expanded.
    """
    q[0] = root
    sc[root] = 1.0
    head = 0
    tail = 1
    while head < tail:
        a = q[head]
        head += 1
        if a >= stopkey:
            h = 0.5 * sc[a]
            for s in range(2):
                pk = par[a, s]
                if sc[pk] == 0.0:
                    q[tail] = pk
                    tail += 1
                sc[pk] += h
    return tail


@numba.njit(cache=True)
def _father_dot(root, par, stopkey, D, scm, scf, q):
    """The father's breadth-first walk fused with the dot product.

    An entry's coefficient is final when dequeued, so it is used, expanded and
    zeroed in one visit (later pushes only reach older generations).
    """
    q[0] = root
    scf[root] = 1.0
    head = 0
    tail = 1
    near = 0.0
    while head < tail:
        a = q[head]
        head += 1
        cf = scf[a]
        scf[a] = 0.0
        c = scm[a]
        if c != 0.0:
            near += np.float64(c) * np.float64(cf) * D[a]
        if a >= stopkey:
            h = 0.5 * cf
            for s in range(2):
                pk = par[a, s]
                if scf[pk] == 0.0:
                    q[tail] = pk
                    tail += 1
                scf[pk] += h
    return near


@numba.njit(cache=True)
def _walk_chunk(pm, pf, s, e, par, D, stopkey, scm, scf, qm, qf, out):
    p = s
    nm = 0
    prev = -1
    while p < e:
        m = pm[p]
        if prev >= 0 and par[m, 0] == par[prev, 0] and par[m, 1] == par[prev, 1]:
            scm[prev] = 0.0  # full sister of the previous mother: same ancestors, swap the self entry
            scm[m] = 1.0
            qm[0] = m
        else:
            for i in range(nm):
                scm[qm[i]] = 0.0
            nm = _bfs(m, par, stopkey, scm, qm)
        q = p
        while q < e and pm[q] == m:
            out[q] = 0.5 * _father_dot(pf[q], par, stopkey, D, scm, scf, qf)
            q += 1
        prev = m
        p = q
    for i in range(nm):
        scm[qm[i]] = 0.0


@numba.njit(parallel=True, cache=True)
def _walk(pm, pf, bounds, par, D, stopkey, scm, scf, qm, qf, out):
    for t in prange(bounds.shape[0] - 1):
        _walk_chunk(pm, pf, bounds[t], bounds[t + 1], par, D, stopkey, scm[t], scf[t], qm[t], qf[t], out)


def _chunks(pm: np.ndarray, T: int) -> np.ndarray:
    """T contiguous ranges of the mother-sorted pairs, cut at mother-group starts."""
    n = len(pm)
    cuts = [0]
    for t in range(1, T):
        i = int(np.searchsorted(pm, pm[min(n - 1, t * n // T)], "left"))
        cuts.append(max(i, cuts[-1]))
    cuts.append(n)
    return np.asarray(cuts, np.int64)


def _twin_alias(N: int, twins: np.ndarray) -> np.ndarray:
    """Each row's canonical genome row: a second co-twin maps to the first."""
    alias = np.arange(N, dtype=np.int64)
    if len(twins):
        alias[twins[:, 1]] = twins[:, 0]
    return alias


class _Walkd2:
    """Exact F by deduplicating path walks over the whole history (walkd2 at L = G_sim).

    Keys are ``generation * N + row``, founders generation 0. Each thread owns
    two scratch coefficient vectors over the whole history, float32 while the
    dyadic coefficients (``G_sim + 1`` bits) are exact in it, else float64, and
    two int32 queues. Work is split at mother-group starts and each pair's F
    is computed by one thread, so F does not depend on the thread count.
    """

    def __init__(self, N: int, G_sim: int) -> None:
        self.N = N
        self.g = 0
        tot = N * (G_sim + 1)
        self.par = np.zeros((tot, 2), np.int32)
        self.D = np.ones(tot)
        self.Fg = np.zeros(tot)
        self.last_alias = np.arange(N)
        self.T = numba.get_num_threads()
        win = tot
        sdt = np.float32 if G_sim <= 23 else np.float64
        self.scm = np.zeros((self.T, win), sdt)
        self.scf = np.zeros((self.T, win), sdt)
        qmax = sum(min(1 << k, N) for k in range(G_sim + 1))
        self.qm = np.empty((self.T, qmax), np.int32)
        self.qf = np.empty((self.T, qmax), np.int32)

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        N, g, last_alias = self.N, self.g, self.last_alias
        alias = _twin_alias(N, twins)
        rep = alias == np.arange(N)
        cr = np.flatnonzero(rep)
        mloc = last_alias[parents[cr, 0]]
        floc = last_alias[parents[cr, 1]]
        keys = mloc * N + floc
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        first = np.empty(len(ks), bool)
        first[0] = True
        np.not_equal(ks[1:], ks[:-1], out=first[1:])
        uniq = ks[first]
        inv = np.empty(len(ks), np.int64)
        inv[order] = np.cumsum(first) - 1
        pm = uniq // N + g * N
        pf = uniq % N + g * N
        out = np.empty(len(uniq))
        _walk(pm, pf, _chunks(pm, self.T), self.par, self.D, N, self.scm, self.scf, self.qm, self.qf, out)
        F = np.empty(N)
        F[cr] = out[inv]
        F[~rep] = F[alias[~rep]]
        lo = (g + 1) * N
        pmg = last_alias[parents[:, 0]] + g * N
        pfg = last_alias[parents[:, 1]] + g * N
        self.par[lo : lo + N, 0] = pmg
        self.par[lo : lo + N, 1] = pfg
        self.D[lo : lo + N] = 0.5 - (self.Fg[pmg] + self.Fg[pfg]) / 4
        self.Fg[lo : lo + N] = F
        self.last_alias = alias
        self.g += 1
        return F


# ---------------------------------------------------------------- stepper


class Inbreeding:
    """The offspring inbreeding coefficients of each mating iteration, under one plan.

    Args:
        plan: the resolved plan; its ``engine`` picks the computation.
        N: individuals per generation.
        G_sim: mating iterations the simulation runs.
        mother_sorted: whether every iteration's offspring rows are sorted by
            mother (the standard mating model), which xexact's parallel
            passes need.
    """

    def __init__(self, plan: InbreedingPlan, N: int, G_sim: int, mother_sorted: bool) -> None:
        self.plan = plan
        if plan.engine == "densep":
            self._engine: _Densep | _Walkd2 = _Densep(N)
        elif plan.engine == "walkd2":
            self._engine = _Walkd2(N, G_sim)
        else:
            raise NotImplementedError(f"inbreeding engine {plan.engine!r} is not implemented yet")

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        """Return the float64 inbreeding coefficients of the offspring of ``parents``, shape ``(N,)``.

        ``twins`` holds the ``(first, second)`` offspring rows of each MZ pair.
        """
        return self._engine.step(parents, np.asarray(twins, np.int64).reshape(-1, 2))
