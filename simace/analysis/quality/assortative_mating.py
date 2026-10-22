"""Mate-correlation (assortative-mating) validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core.numerics import finite_or_none
from simace.simulation.assortment import _parent_ce_gen
from simace.simulation.mate_correlation import expected_mate_corr_matrix

from ._common import _info

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ._common import MateCorrelations

__all__ = [
    "RecordedMating",
    "assortment_schedule",
    "recorded_matings",
    "schedule_span",
    "validate_assortative_mating",
]

_CROSS_CELLS = (("cross_12", (0, 1)), ("cross_21", (1, 0)))


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


def schedule_span(params: Mapping[str, Any]) -> tuple[int, int]:
    """``(G_sim, burnin)``: iteration ``i`` records generation ``i - burnin``, so recorded generation 0 is iteration ``burnin``'s offspring."""
    G_sim = int(params.get("G_sim") or params.get("G_ped") or 1)
    G_ped = int(params.get("G_ped") or G_sim)
    return G_sim, G_sim - G_ped


def assortment_schedule(params: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """The ``assort1`` and ``assort2`` the simulator mates on in each of the ``G_sim`` mating iterations.

    A set ``assort_matrix`` replaces both by its diagonal, as
    ``SimulationParams`` does. The Wright-Fisher mating model pairs parents
    at random whatever assortment it inherits, so both are zero there.
    """
    from simace.simulation.simulate import resolve_per_gen_param

    G_sim, _ = schedule_span(params)
    if params.get("mating_model", "standard") != "standard":
        return [0.0] * G_sim, [0.0] * G_sim
    matrix = params.get("assort_matrix")
    if matrix is not None:
        matrix = np.asarray(matrix, dtype=np.float64)
        return [float(matrix[0, 0])] * G_sim, [float(matrix[1, 1])] * G_sim
    return (
        resolve_per_gen_param(params.get("assort1", 0.0), G_sim, name="assort1", allow_negative=True),
        resolve_per_gen_param(params.get("assort2", 0.0), G_sim, name="assort2", allow_negative=True),
    )


def recorded_matings(params: dict[str, Any]) -> list[RecordedMating]:
    """One :class:`RecordedMating` per mating iteration whose parents are both pedigree rows.

    Iteration ``i`` records generation ``i - burnin``, so its parents are
    recorded generation ``i - burnin - 1`` and carry the C/E of generation
    ``_parent_ce_gen(i)``. The targets follow :func:`assortment_schedule`;
    a set ``assort_matrix`` is mated on only when both traits assort, and
    under Wright-Fisher every target is zero.
    """
    from simace.simulation.simulate import resolve_per_gen_param

    G_sim, burnin = schedule_span(params)
    assort1, assort2 = assortment_schedule(params)
    matrix = params.get("assort_matrix") if params.get("mating_model", "standard") == "standard" else None
    if matrix is not None:
        matrix = np.asarray(matrix, dtype=np.float64)
    per_gen = {k: resolve_per_gen_param(params.get(k, 0), G_sim, name=k) for k in ("C1", "C2", "E1", "E2")}

    matings = []
    for i in range(burnin + 1, G_sim):
        a1, a2 = assort1[i], assort2[i]
        g = _parent_ce_gen(i)
        expected = expected_mate_corr_matrix(
            a1,
            a2,
            rA=params.get("rA", 0),
            rC=params.get("rC", 0),
            A1=params.get("A1", 0),
            C1=per_gen["C1"][g],
            A2=params.get("A2", 0),
            C2=per_gen["C2"][g],
            assort_matrix=matrix if a1 != 0 and a2 != 0 else None,
            rE=params.get("rE", 0),
            E1=per_gen["E1"][g],
            E2=per_gen["E2"][g],
        )
        ce = (per_gen["C1"][g], per_gen["C2"][g], per_gen["E1"][g], per_gen["E2"][g])
        matings.append(RecordedMating(i - burnin - 1, (a1, a2), expected, ce))
    return matings


def validate_assortative_mating(params: dict[str, Any], mates: MateCorrelations) -> dict[str, Any]:
    """Report the pedigree's mate correlations, as information with no pass/fail.

    Each is the Pearson correlation of mother and father values over every
    Mating Pair: ``mate_corr_liability{t}`` per trait, ``mate_corr_cross_12``
    / ``mate_corr_cross_21`` when either trait assorts in some recorded
    mating, and the genetic ``mate_corr_A{t}``. A liability entry carries
    ``expected`` when its trait assorts in every recorded mating with the
    same target, or when no trait assorts anywhere (target 0); a cross entry
    when both traits assort in every recorded mating with the
    same target and the same C/E, since a pooled cross-trait Pearson mixes
    the generations' trait scales (ADR 0026). ``simace validate``'s
    ``mate_correlation`` test scores each parents' generation against its
    target (ADR 0031).

    Args:
        params: Scenario parameters (``params.yaml``).
        mates: The pedigree's mate correlations
            (:func:`~simace.analysis.quality._common.mate_correlations`).

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}
    matings = recorded_matings(params)
    n_pairs = mates.n_pairs
    if n_pairs == 0 or not matings:
        results["mate_corr_liability1"] = _info("No mating pairs with both parents present")
        results["mate_corr_liability2"] = _info("No mating pairs with both parents present")
        return results

    # (name, cell, the target when one applies to every Mating Pair). A cell
    # the simulator does not mate on is induced, and its nominal target is
    # biased (ADR 0031), so it has none.
    assorts = any(m.assort != (0, 0) for m in matings)
    entries = [
        (
            f"mate_corr_liability{t}",
            (t - 1, t - 1),
            _pooled_target(
                [m if m.assort[t - 1] != 0 or not assorts else None for m in matings], (t - 1, t - 1), same_ce=False
            ),
        )
        for t in (1, 2)
    ]
    if assorts:
        cross = [m if m.cross_scored else None for m in matings]
        entries += [
            (f"mate_corr_{label}", cell, _pooled_target(cross, cell, same_ce=True)) for label, cell in _CROSS_CELLS
        ]

    for name, cell, expected in entries:
        observed = finite_or_none(mates.liability[cell])
        target = "" if expected is None else f" (target {expected:.4f})"
        observed_text = "not computable" if observed is None else f"{observed:.4f}"
        extra = {} if expected is None else {"expected": expected}
        results[name] = _info(f"{name}: {observed_text}{target}", observed=observed, n_pairs=n_pairs, **extra)

    # Genetic (A-component) mate correlation mu_A; consumed by the
    # AM-corrected relative-correlation reference lines (see am_relatedness
    # and plot_A_correlations). Reduces to ~0 with no assortment.
    for t in [1, 2]:
        obs_a = mates.genetic[t - 1]
        results[f"mate_corr_A{t}"] = _info(
            f"Genetic mate correlation A{t} (mu_A): {obs_a:.4f}",
            observed=finite_or_none(obs_a),
            n_pairs=n_pairs,
        )

    return results


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
