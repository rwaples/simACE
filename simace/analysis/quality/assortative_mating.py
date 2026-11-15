"""Mate-correlation (assortative-mating) validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from simace.core.numerics import finite_or_none
from simace.simulation.assortment import parent_ce_gen
from simace.simulation.mate_correlation import expected_mate_corr_matrix

from .check import FamilyResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from simace.simulation.params import Assortment, SimulationParams, Span, VarianceComponents

    from ._common import MateCorrelations

__all__ = [
    "AssortativeMatingMeasurements",
    "MateCorrelation",
    "RecordedMating",
    "recorded_matings",
    "validate_assortative_mating",
]

#: The cross-trait cells (mothers' trait 1 with fathers' trait 2, and the reverse).
_CROSS_CELLS = ((0, 1), (1, 0))


@dataclass(frozen=True)
class RecordedMating:
    """The simulator's mate-correlation target for one mating whose parents are recorded.

    Attributes:
        parents_generation: The parents' ``generation`` in the pedigree.
        assort: The iteration's ``assort1`` and ``assort2``.
        expected: ``expected[i, j]`` is the target correlation of the
            mothers' ``liability{i+1}`` with the fathers' ``liability{j+1}``.
        ce: ``C1, C2, E1, E2`` of the parents' C/E generation.
    """

    parents_generation: int
    assort: tuple[float, float]
    expected: np.ndarray
    ce: tuple[float, float, float, float]

    @property
    def cross_scored(self) -> bool:
        """Whether the cross-trait cells are a target the simulator mates on, not induced."""
        return self.assort[0] != 0 and self.assort[1] != 0


def recorded_matings(span: Span, vc: VarianceComponents, assortment: Assortment) -> list[RecordedMating]:
    """One :class:`RecordedMating` per mating iteration whose parents are both pedigree rows.

    Iteration ``i`` records generation ``i - burnin``, so its parents are
    recorded generation ``i - burnin - 1`` and carry the C/E of generation
    ``parent_ce_gen(i)``. The targets follow the assortment schedule; a set
    ``assort_matrix`` is mated on only when both traits assort, and under
    Wright-Fisher every target is zero.
    """
    assort1, assort2 = assortment.schedule
    matrix = assortment.mated_matrix
    C1, C2 = float(vc.C[0]), float(vc.C[1])
    E = {t: vc.E_per_gen(t, span.G_sim) for t in (1, 2)}

    matings = []
    for i in range(span.burnin + 1, span.G_sim):
        a1, a2 = assort1[i], assort2[i]
        g = parent_ce_gen(i)
        expected = expected_mate_corr_matrix(
            a1,
            a2,
            rA=vc.rA,
            rC=vc.rC,
            A1=vc.A[0],
            C1=C1,
            A2=vc.A[1],
            C2=C2,
            assort_matrix=matrix if a1 != 0 and a2 != 0 else None,
            rE=vc.rE,
            E1=E[1][g],
            E2=E[2][g],
        )
        ce = (C1, C2, E[1][g], E[2][g])
        matings.append(RecordedMating(i - span.burnin - 1, (a1, a2), expected, ce))
    return matings


@dataclass(frozen=True)
class MateCorrelation:
    """One cell of the mate correlation matrix, pooled over every Mating Pair.

    Attributes:
        observed: The Pearson correlation; ``None`` where it is not computable.
        expected: The target, when one applies to every Mating Pair.
    """

    observed: float | None
    expected: float | None


@dataclass(frozen=True)
class AssortativeMatingMeasurements:
    """What the assortative-mating family measures; the correlations are ``None`` with no recorded mating pairs.

    Attributes:
        n_pairs: The Mating Pairs.
        liability: ``mate_corr_liability{t}``, ``liability[t - 1]``.
        cross: The cross-trait cells ``(cross_12, cross_21)``, only when some
            trait assorts in a recorded mating.
        genetic: The genetic mate correlation ``mu_A`` per trait.
    """

    n_pairs: int
    liability: tuple[MateCorrelation | None, MateCorrelation | None]
    cross: tuple[MateCorrelation, MateCorrelation] | None
    genetic: tuple[float | None, float | None]


def validate_assortative_mating(
    sim: SimulationParams, mates: MateCorrelations
) -> FamilyResult[AssortativeMatingMeasurements]:
    """Measure the pedigree's mate correlations; the family asserts no checks.

    Each is the Pearson correlation of mother and father values over every
    Mating Pair: the liability correlation per trait, the cross-trait cells
    when either trait assorts in some recorded mating, and the genetic
    ``mu_A``. A liability cell carries ``expected`` when its trait assorts
    in every recorded mating with the same target, or when no trait assorts
    anywhere (target 0); a cross cell when both traits assort in every
    recorded mating with the same target and the same C/E, since a pooled
    cross-trait Pearson mixes the generations' trait scales (ADR 0026).
    ``simace validate``'s ``mate_correlation`` test scores each parents'
    generation against its target (ADR 0031).

    Args:
        sim: The rep's parameters (``params.yaml``).
        mates: The pedigree's mate correlations
            (:func:`~simace.analysis.quality._common.mate_correlations`).
    """
    matings = recorded_matings(sim.span, sim.vc, sim.assortment)
    n_pairs = mates.n_pairs
    if n_pairs == 0 or not matings:
        return FamilyResult(AssortativeMatingMeasurements(n_pairs, (None, None), None, (None, None)), ())

    def cell(index: tuple[int, int], expected: float | None) -> MateCorrelation:
        return MateCorrelation(observed=finite_or_none(mates.liability[index]), expected=expected)

    # A cell the simulator does not mate on is induced, and its nominal
    # target is biased (ADR 0031), so it has none.
    assorts = any(m.assort != (0, 0) for m in matings)
    liability = tuple(
        cell(
            (t - 1, t - 1),
            _pooled_target(
                [m if m.assort[t - 1] != 0 or not assorts else None for m in matings], (t - 1, t - 1), same_ce=False
            ),
        )
        for t in (1, 2)
    )
    cross = None
    if assorts:
        scored = [m if m.cross_scored else None for m in matings]
        cross_12, cross_21 = (cell(index, _pooled_target(scored, index, same_ce=True)) for index in _CROSS_CELLS)
        cross = (cross_12, cross_21)

    # Genetic (A-component) mate correlation mu_A; consumed by the
    # AM-corrected relative-correlation reference lines (see am_relatedness
    # and plot_A_correlations). Reduces to ~0 with no assortment.
    genetic = (finite_or_none(mates.genetic[0]), finite_or_none(mates.genetic[1]))
    measurements = AssortativeMatingMeasurements(n_pairs, (liability[0], liability[1]), cross, genetic)
    return FamilyResult(measurements, ())


def _pooled_target(matings: Sequence[RecordedMating | None], cell: tuple[int, int], *, same_ce: bool) -> float | None:
    """The one target of ``cell`` across *matings*, or None when one is missing (None) or they differ."""
    if any(m is None for m in matings):
        return None
    values = [float(m.expected[cell]) for m in matings if m is not None]
    if max(values) - min(values) > 1e-12:
        return None
    if same_ce and len({m.ce for m in matings if m is not None}) > 1:
        return None
    return values[0]
