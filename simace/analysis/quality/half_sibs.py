"""Half-sibling structure and variance-component correlation checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simace.core.numerics import unique_sorted

from ._common import _MIN_PAIRS_FOR_CORR, SIBLING_CATEGORIES, category_cell, pair_correlation
from .check import FamilyResult, PairCorrelation

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments

    from simace.simulation.params import SimulationParams

    from ._common import MateCorrelations


def household_sibling_counts(df: pd.DataFrame | pl.DataFrame) -> dict[str, int]:
    """Count offspring with a maternal sibling and with a maternal half-sib, from household sizes.

    A household is one mother's offspring (GLOSSARY.md), so the counts are
    O(N) from the ``mother``, ``father`` and ``twin`` columns:

    - ``n_offspring_with_sibs``: offspring with a known mother whose
      household holds another offspring besides themselves and their own
      co-twin. Only the individual's own co-twin is excluded.
    - ``n_offspring_with_maternal_half_sib``: offspring with a known mother
      whose household holds an offspring with a different father. An
      unknown father differs from every other father, including another
      unknown one, as the engine classifies such pairs as ``MHS``.

    Offspring with an unknown mother take part in neither count.
    """
    mothers = df["mother"].to_numpy().astype(np.int64)
    fathers = df["father"].to_numpy().astype(np.int64)
    twins = df["twin"].to_numpy().astype(np.int64)
    known = mothers != -1
    mothers, fathers, twins = mothers[known], fathers[known], twins[known]
    if mothers.size == 0:
        return {"n_offspring_with_sibs": 0, "n_offspring_with_maternal_half_sib": 0}

    households, household_of = unique_sorted(mothers, return_inverse=True)
    size = np.bincount(household_of, minlength=len(households))
    others = size[household_of] - 1 - (twins != -1)

    father_known = fathers != -1
    base = np.int64(max(int(fathers.max()), 0) + 1)
    known_pairs = unique_sorted(household_of[father_known] * base + fathers[father_known])
    n_fathers = np.bincount((known_pairs // base).astype(np.intp), minlength=len(households))
    n_unknown = np.bincount(household_of[~father_known], minlength=len(households))
    other_father = np.where(
        father_known,
        (n_fathers[household_of] >= 2) | (n_unknown[household_of] >= 1),
        others >= 1,
    )
    return {
        "n_offspring_with_sibs": int((others >= 1).sum()),
        "n_offspring_with_maternal_half_sib": int(other_father.sum()),
    }


@dataclass(frozen=True)
class HalfSibTrait:
    """One trait's half-sib correlations; each is ``None`` with fewer than ten pairs to correlate.

    Attributes:
        A_corr: The A correlation over pooled maternal and paternal half-sibs.
        liability_corr: The liability correlation over paternal half-sibs.
        shared_C: The C correlation over paternal half-sibs.
    """

    A_corr: PairCorrelation | None
    liability_corr: PairCorrelation | None
    shared_C: PairCorrelation | None


@dataclass(frozen=True)
class HalfSibMeasurements:
    """What the half-sib family measures.

    Attributes:
        pair_proportion: The maternal half-sib share of maternal sibling
            pairs; ``None`` with no maternal sibling pairs.
        offspring_with_half_sib: The share of offspring with maternal sibs
            who have a maternal half-sib; ``None`` with no such offspring.
        traits: Each trait's correlations, ``traits[t - 1]``.
    """

    pair_proportion: float | None
    offspring_with_half_sib: float | None
    traits: tuple[HalfSibTrait, HalfSibTrait]


def _half_sib_trait(
    mates: MateCorrelations, sibling_moments: RelationshipMoments, sim: SimulationParams, t: int
) -> HalfSibTrait:
    """Measure trait ``t``'s half-sib correlations for A, liability, and shared C, with their expectations.

    None is asserted: `simace validate`'s ``mendelian_residual`` and
    ``household_draw`` test the inheritance (ADR 0029).

    Pooling rule:
    - **A correlation** uses MHS ∪ PHS — both share kinship 0.25 for the
      additive component, so pooling is a sample-size win. Expected: 0.25
      (kinship) under random mating; under single-trait assortative mating it
      inflates to ``(1 + 2·mu_A + mu_A·r_ho)/4`` (see :mod:`.am_relatedness`).
      Both-trait AM has no expectation.
    - **Liability and shared-C correlations** use PHS only. Maternal
      half-sibs share households, so MHS liability corr = 0.25·A + 1·C and
      MHS shared_C ≠ 0; PHS gives the clean expected formulas (0.25·A and 0).
      Under paternal C transmission PHS share ``b_f²`` of C, so the shared-C
      expectation becomes ``b_f²`` where the closed form holds (ADR 0025).

    Every correlation is exact over all pairs (ADR 0022); ``n_pairs`` is the
    true pair count.
    """
    pooled = category_cell(sibling_moments, "MHS", "PHS")
    phs = category_cell(sibling_moments, "PHS")
    n_pooled = int(pooled.counts)
    n_phs = int(phs.counts)

    a_corr = None
    if n_pooled >= _MIN_PAIRS_FOR_CORR:
        a_corr = PairCorrelation(observed=float(pair_correlation(pooled, f"A{t}")), n_pairs=n_pooled)
    if n_phs < _MIN_PAIRS_FOR_CORR:
        return HalfSibTrait(A_corr=a_corr, liability_corr=None, shared_C=None)
    return HalfSibTrait(
        A_corr=a_corr,
        liability_corr=PairCorrelation(observed=float(pair_correlation(phs, f"P{t}")), n_pairs=n_phs),
        shared_C=PairCorrelation(observed=float(pair_correlation(phs, f"C{t}")), n_pairs=n_phs),
    )


def validate_half_sibs(
    df: pd.DataFrame | pl.DataFrame,
    sim: SimulationParams,
    sibling_moments: RelationshipMoments,
    mates: MateCorrelations,
) -> FamilyResult[HalfSibMeasurements]:
    """Measure half-sibling structure under the mating-pair model.

    Measures the maternal half-sib share of maternal sibling pairs and of
    offspring with maternal sibs. With a zero-truncated Poisson mating
    model, both maternal and paternal half-sibs arise naturally when
    individuals have multiple partners. Also measures half-sib
    variance-component correlations (A, liability, shared C); see
    ``_half_sib_trait`` for the pooling rules. Nothing here has a
    closed-form verdict, so the family asserts no checks.

    Args:
        df: Pedigree DataFrame with columns id, mother, father, twin.
        sim: The rep's parameters.
        sibling_moments: Relationship moments of ``df`` over ``FS``, ``MHS``
            and ``PHS`` with the ``A{t}``, ``C{t}`` and ``P{t}`` columns
            (:func:`~simace.analysis.quality._common.sibling_moments`).
        mates: The pedigree's mate correlations, for the AM-aware expected
            values (:func:`~simace.analysis.quality._common.mate_correlations`).
    """
    n_full, n_mat, _n_pat = (int(category_cell(sibling_moments, code).counts) for code in SIBLING_CATEGORIES)
    total_maternal_pairs = n_full + n_mat
    pair_proportion = n_mat / total_maternal_pairs if total_maternal_pairs > 0 else None

    household = household_sibling_counts(df)
    n_offspring_with_sibs = household["n_offspring_with_sibs"]
    n_offspring_with_hs = household["n_offspring_with_maternal_half_sib"]
    offspring_with_half_sib = n_offspring_with_hs / n_offspring_with_sibs if n_offspring_with_sibs > 0 else None

    measurements = HalfSibMeasurements(
        pair_proportion=pair_proportion,
        offspring_with_half_sib=offspring_with_half_sib,
        traits=(_half_sib_trait(mates, sibling_moments, sim, 1), _half_sib_trait(mates, sibling_moments, sim, 2)),
    )
    return FamilyResult(measurements, ())
