"""Mate-correlation (assortative-mating) validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from simace.analysis.mate_correlation import mate_correlation_by_generation
from simace.simulation.assortment import _parent_ce_gen
from simace.simulation.mate_correlation import expected_mate_corr_matrix

from ._common import _corr_se, _info, _result

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from ._common import MateCorrelations

__all__ = ["RecordedMating", "recorded_matings", "validate_assortative_mating"]

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


def recorded_matings(params: dict[str, Any]) -> list[RecordedMating]:
    """One :class:`RecordedMating` per mating iteration whose parents are both pedigree rows.

    Iteration ``i`` records generation ``i - burnin``, so its parents are
    recorded generation ``i - burnin - 1`` and carry the C/E of generation
    ``_parent_ce_gen(i)``. A set ``assort_matrix`` replaces ``assort1`` /
    ``assort2`` by its diagonal, as in the simulator's parameters, and is
    mated on only when both traits assort. The Wright-Fisher mating model
    pairs parents at random whatever assortment it inherits, so every
    target is zero.
    """
    from simace.simulation.simulate import resolve_per_gen_param

    G_sim = int(params.get("G_sim") or params.get("G_ped") or 1)
    G_ped = int(params.get("G_ped") or G_sim)
    burnin = G_sim - G_ped
    matrix = params.get("assort_matrix")
    if params.get("mating_model", "standard") != "standard":
        matrix = None
        assort = {"assort1": [0.0] * G_sim, "assort2": [0.0] * G_sim}
    elif matrix is not None:
        matrix = np.asarray(matrix, dtype=np.float64)
        assort = {"assort1": [float(matrix[0, 0])] * G_sim, "assort2": [float(matrix[1, 1])] * G_sim}
    else:
        assort = {
            k: resolve_per_gen_param(params.get(k, 0.0), G_sim, name=k, allow_negative=True)
            for k in ("assort1", "assort2")
        }
    per_gen = {k: resolve_per_gen_param(params.get(k, 0), G_sim, name=k) for k in ("C1", "C2", "E1", "E2")}

    matings = []
    for i in range(burnin + 1, G_sim):
        a1, a2 = assort["assort1"][i], assort["assort2"][i]
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


def validate_assortative_mating(
    params: dict[str, Any], mates: MateCorrelations, df: pd.DataFrame | pl.DataFrame
) -> dict[str, Any]:
    """Validate mate correlations on liability against the simulator's targets.

    Checks the Pearson correlation of mother and father liability over the
    Mating Pairs, per trait, and when both traits assort the two cross-trait
    correlations, against :func:`recorded_matings`. A check pools every
    Mating Pair when that is valid and otherwise becomes one check per
    parents' generation, ``{name}_gen{g}`` (ADR 0026):

    - ``mate_corr_liability{t}`` pools when its target is the same in every
      recorded mating.
    - ``mate_corr_cross_12`` / ``mate_corr_cross_21`` pool when both traits
      assort in every recorded mating with the same target and the same C/E,
      since a pooled cross-trait Pearson mixes the generations' trait scales.
      Split, they cover only the generations in which both traits assort.

    A split check keeps its pooled correlation under its own name, as
    information with no pass/fail.

    Args:
        params: Scenario parameters (``params.yaml``).
        mates: The pedigree's mate correlations
            (:func:`~simace.analysis.validate._common.mate_correlations`).
        df: The pedigree, for per-generation mate correlations.

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}
    matings = recorded_matings(params)
    n_pairs = mates.n_pairs
    if n_pairs == 0 or not matings:
        results["mate_corr_liability1"] = _result(True, "No mating pairs with both parents present")
        results["mate_corr_liability2"] = _result(True, "No mating pairs with both parents present")
        return results

    # (name, cell, the matings it covers, whether it pools)
    checks = [
        (f"mate_corr_liability{t}", (t - 1, t - 1), matings, _constant([m.expected[t - 1, t - 1] for m in matings]))
        for t in (1, 2)
    ]
    scored = [m for m in matings if m.cross_scored]
    if scored:
        pooled = (
            len(scored) == len(matings)
            and _constant([m.expected[0, 1] for m in scored])
            and all(m.ce == scored[0].ce for m in scored)
        )
        checks += [(f"mate_corr_{label}", cell, scored, pooled) for label, cell in _CROSS_CELLS]
    by_generation: dict[int, tuple[np.ndarray, int]] = {}
    if not all(pooled for *_, pooled in checks):
        generations = [m.parents_generation for m in matings]
        by_generation = mate_correlation_by_generation(df, ["liability1", "liability2"], generations)

    for name, cell, subset, pooled in checks:
        observed = float(mates.liability[cell])
        if pooled:
            results[name] = _check(name, observed, float(subset[0].expected[cell]), n_pairs)
            continue
        results[name] = _info(
            f"{name}: {observed:.4f} pooled over generations with different targets; checked per generation",
            observed=None if np.isnan(observed) else observed,
            n_pairs=n_pairs,
        )
        for m in subset:
            matrix, n = by_generation[m.parents_generation]
            key = f"{name}_gen{m.parents_generation}"
            results[key] = _check(key, float(matrix[cell]), float(m.expected[cell]), n)

    # Genetic (A-component) mate correlation mu_A — informational; consumed by
    # the AM-corrected relative-correlation reference lines (see am_relatedness
    # and plot_A_correlations). Reduces to ~0 with no assortment.
    for t in [1, 2]:
        obs_a = mates.genetic[t - 1]
        results[f"mate_corr_A{t}"] = _info(
            f"Genetic mate correlation A{t} (mu_A): {obs_a:.4f}",
            observed=None if np.isnan(obs_a) else float(obs_a),
            n_pairs=n_pairs,
        )

    return results


def _constant(values: list[float]) -> bool:
    return max(values) - min(values) <= 1e-12


def _check(name: str, observed: float, expected: float, n_pairs: int) -> dict[str, Any]:
    """Compare one mate correlation with its target, within ``max(0.1, 3 * se)``.

    No pairs fails: the simulator recorded matings there, so the pedigree
    lost them or its generations are not the recorded ones.
    """
    if n_pairs == 0:
        return _result(False, f"{name}: no Mating Pairs", expected=expected, n_pairs=0)
    if np.isnan(observed):
        return _result(
            True,
            f"Cannot compute {name} (zero variance or too few pairs)",
            expected=expected,
            observed=observed,
            n_pairs=n_pairs,
        )
    tol = max(0.1, 3 * _corr_se(expected, n_pairs))
    return _result(
        abs(observed - expected) < tol,
        f"{name}: {observed:.4f} (expected: {expected:.4f}, tol: {tol:.4f})",
        expected=expected,
        observed=observed,
        n_pairs=n_pairs,
    )
