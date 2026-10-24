"""Per-generation inbreeding coefficients for the simulate loop (ADR 0033).

:class:`Inbreeding` is stepped once per mating iteration with that iteration's
``parents`` (rows of the previous iteration's offspring) and MZ ``twins``, and
returns the offspring's inbreeding coefficients over the whole simulated
ancestry, founders at F = 0. The engine is fixed by an
:class:`~simace.simulation.inbreeding_plan.InbreedingPlan`:

- ``densep``: the dense kinship recurrence ``K_t = P K_{t-1} P'`` in float32,
  two N x N arrays (``notes/issue-76-recommendation/densep.py``).

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
            self._engine = _Densep(N)
        else:
            raise NotImplementedError(f"inbreeding engine {plan.engine!r} is not implemented yet")

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        """Return the float64 inbreeding coefficients of the offspring of ``parents``, shape ``(N,)``.

        ``twins`` holds the ``(first, second)`` offspring rows of each MZ pair.
        """
        return self._engine.step(parents, np.asarray(twins, np.int64).reshape(-1, 2))
