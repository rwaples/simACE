"""Assortative-mating additive-variance equilibrium under the infinitesimal model.

simACE inherits the additive genetic value as midparent + Mendelian sampling
noise of *fixed* variance ``A_base / 2`` (see ``reproduce`` in
:mod:`simace.simulation.simulate`). Under phenotypic assortative mating with
mate liability correlation ``r_ho``, the additive genetic variance inflates
across generations — the Bulmer (1971) effect — to an equilibrium ``a²``.

This module is the single source of truth for that equilibrium and for the
generation-by-generation trajectory. The validation check
(:mod:`simace.analysis.validate.am_equilibrium`) and the diagnostic plot
(:mod:`simace.plotting.plot_am_equilibrium`) both consume it — change one, change
the consumers.

Theory
------
Let ``V_t = Var(A)`` at generation ``t`` and ``V_env = C + E`` the (constant)
environmental variance. Mates assort on liability ``P = A + C + E`` with Pearson
correlation ``r_ho``, which induces a genetic correlation between mates of
``r_ho · h²_t`` with ``h²_t = V_t / (V_t + V_env)``. With the fixed Mendelian
sampling variance ``A_base / 2``, offspring variance follows

    V_{t+1} = ½·V_t·(1 + r_ho·h²_t) + ½·A_base.

Its fixed point is the positive root of

    (1 − r_ho)·V² + (V_env − A_base)·V − A_base·V_env = 0.

This equals the assortative-mating-only equilibrium additive variance ``a²`` of
Herzig et al. (2026, *Theor. Popul. Biol.* 170:26–35,
doi:10.1016/j.tpb.2026.06.003): substituting their random-mating gametic
variance ``g0² = A_base / 2`` and environmental variance ``e² = V_env`` into
their closed form makes the two discriminants algebraically identical. The
infinitesimal midparent model and their explicit gametic-disequilibrium model
therefore share the same AM equilibrium (the classic Bulmer result).
"""

from __future__ import annotations

__all__ = ["am_equilibrium_variance", "am_variance_trajectory"]

import numpy as np


def am_equilibrium_variance(A_base: float, V_env: float, r_ho: float) -> float:
    """Equilibrium additive genetic variance ``a²`` under phenotypic AM.

    Positive root of ``(1 − r_ho)·V² + (V_env − A_base)·V − A_base·V_env = 0``.
    Equal to the AM-only ``a²`` of Herzig et al. (2026) with their ``g0² =
    A_base/2``, ``e² = V_env``. At ``r_ho = 0`` this returns ``A_base`` (no
    inflation); ``r_ho < 0`` (disassortative) returns a deflated variance.

    Args:
        A_base: Founder additive genetic variance (constant Mendelian base).
        V_env: Total environmental variance ``C + E`` (assumed constant).
        r_ho: Mate liability (phenotypic) correlation.

    Returns:
        Equilibrium ``Var(A)``. ``inf`` if ``r_ho >= 1`` (variance diverges);
        ``0.0`` if ``A_base <= 0``.
    """
    A_base = float(A_base)
    V_env = float(V_env)
    r_ho = float(r_ho)
    if A_base <= 0.0:
        return 0.0
    if r_ho >= 1.0:
        return float("inf")
    disc = (V_env - A_base) ** 2 + 4.0 * (1.0 - r_ho) * A_base * V_env
    return ((A_base - V_env) + np.sqrt(disc)) / (2.0 * (1.0 - r_ho))


def am_variance_trajectory(A_base: float, V_env: float, r_ho: float, n_steps: int) -> np.ndarray:
    """Deterministic ``Var(A)`` after each of ``n_steps`` reproduce steps.

    Iterates ``V_{k+1} = ½·V_k·(1 + r_ho·h²_k) + ½·A_base`` from the founder
    variance ``V_0 = A_base``.

    Args:
        A_base: Founder additive genetic variance.
        V_env: Total environmental variance ``C + E`` (assumed constant).
        r_ho: Mate liability correlation.
        n_steps: Number of reproduce steps (generations) to iterate.

    Returns:
        Array of length ``n_steps + 1``; element ``k`` is ``Var(A)`` after ``k``
        reproduce steps (so ``V[0] == A_base`` are the founders).
    """
    A_base = float(A_base)
    V_env = float(V_env)
    r_ho = float(r_ho)
    n = int(n_steps)
    V = np.empty(n + 1, dtype=np.float64)
    V[0] = A_base
    for k in range(n):
        sigma2 = V[k] + V_env
        h2 = V[k] / sigma2 if sigma2 > 0.0 else 0.0
        V[k + 1] = 0.5 * V[k] * (1.0 + r_ho * h2) + 0.5 * A_base
    return V
