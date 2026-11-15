"""Assortative-mating additive-variance equilibrium validation.

Under phenotypic assortative mating the additive genetic variance inflates
across generations to the Bulmer equilibrium ``a²`` — identical to the
assortative-mating-only result of Herzig et al. (2026, *Theor. Popul. Biol.*
170:26–35, doi:10.1016/j.tpb.2026.06.003). This module checks that the simulated
``Var(A)`` at the final recorded generation matches the value the infinitesimal
recursion predicts after ``G_sim`` reproduce steps (see
:mod:`simace.simulation.am_equilibrium` for the theory).

The final recorded generation has undergone exactly ``G_sim`` reproduce steps
from the founders, so the recursion is evaluated at ``G_sim`` regardless of any
burn-in. The asymptotic equilibrium ``a²`` is reported for context but the
assertion is against the recursion value at the actual ``G_sim`` (valid whether
or not the run has converged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simace.simulation.am_equilibrium import am_equilibrium_variance, am_variance_trajectory

from .am_relatedness import am_relatedness_mode
from .check import FamilyResult, QualityCheck

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from simace.simulation.params import SimulationParams


@dataclass(frozen=True)
class AMEquilibriumMeasurements:
    """What the AM-equilibrium family measures.

    Attributes:
        final_var_A: ``Var(A{t})`` in the last recorded generation, per trait,
            where the univariate recursion applies to the trait's assortment
            (single-trait, constant, scalar C and E); ``None`` otherwise.
    """

    final_var_A: tuple[float | None, float | None]


_NOTHING_MEASURED = AMEquilibriumMeasurements(final_var_A=(None, None))


def validate_am_equilibrium(
    df: pd.DataFrame | pl.DataFrame, sim: SimulationParams
) -> FamilyResult[AMEquilibriumMeasurements]:
    """Check that ``Var(A)`` reaches the AM-inflated equilibrium per trait.

    Checks nothing when the mating schedule has no assortment (Wright-Fisher,
    or ``assort1``, ``assort2`` and the ``assort_matrix`` diagonal all zero).
    For an assorting trait, checks the observed final-generation ``Var(A)``
    against the infinitesimal recursion prediction, with the closed-form
    equilibrium ``a²`` in the message.

    Does not check a trait whose equilibrium is ill-defined or not modelled
    by the univariate recursion: AM that varies by generation, dict-valued
    C / E, both-trait assortment (cross-trait paths), or C transmission,
    whose A-C covariance the recursion omits (ADR 0025). Under C
    transmission the final ``Var(A)`` is still measured.

    Args:
        df: Pedigree DataFrame with ``id`` and ``A1``/``A2`` columns.
        sim: The rep's parameters.
    """
    assortment = sim.assortment
    if not assortment.any_active:
        return FamilyResult(_NOTHING_MEASURED, ())  # random mating -> nothing to validate

    G_sim = sim.span.G_sim
    gen_labels = df["id"].to_numpy() // sim.N
    last_mask = gen_labels == (sim.span.G_ped - 1)
    n_last = int(last_mask.sum())

    checks = []
    final_var_A: dict[int, float] = {}
    for t in (1, 2):
        if am_relatedness_mode(assortment, t) != "single":
            continue
        if not assortment.constant(t) or not sim.vc.scalar_E(t):
            continue
        A_base = float(sim.vc.A[t - 1])
        if A_base <= 0.0 or n_last < 2:
            continue

        V_env = float(sim.vc.C[t - 1]) + sim.vc.E_per_gen(t, 1)[0]
        r_ho = assortment.schedule[t - 1][0]

        V_pred = float(am_variance_trajectory(A_base, V_env, r_ho, int(G_sim))[-1])
        a2 = float(am_equilibrium_variance(A_base, V_env, r_ho))
        obs = float(np.var(df[f"A{t}"].to_numpy()[last_mask]))
        final_var_A[t] = obs
        if sim.c_transmission.coefficients(t) != (0.0, 0.0):
            continue

        # SE of a sample variance for ~Gaussian A: V·sqrt(2/(n-1)). The 5x
        # multiplier (floored at 0.03) absorbs the slight non-normality of A
        # under AM and accumulated process noise along the trajectory; a real
        # transmission bug shifts Var(A) far beyond this.
        se = V_pred * np.sqrt(2.0 / max(n_last - 1, 1))
        tol = max(0.03, 5.0 * se)
        checks.append(
            QualityCheck(
                f"am_equilibrium_A{t}",
                abs(obs - V_pred) < tol,
                f"AM Var(A{t}) final generation: {obs:.4f} (predicted {V_pred:.4f} "
                f"after {int(G_sim)} gens; equilibrium a²={a2:.4f}; tol {tol:.4f})",
                observed=obs,
                expected=V_pred,
            )
        )

    measurements = AMEquilibriumMeasurements(final_var_A=(final_var_A.get(1), final_var_A.get(2)))
    return FamilyResult(measurements, tuple(checks))
