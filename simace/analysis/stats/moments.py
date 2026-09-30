"""Relationship moments for the descriptive statistics (ADR 0022).

One engine pass over the analysis sample replaces the relationship pair list:
:func:`relationship_moments_for` declares the factors and value columns the
report reads and returns the pedigree-graph
:class:`~pedigree_graph.RelationshipMoments` table, exact over every pair.
The correlation functions in :mod:`.correlations` reduce that table with
:class:`Stratum`: a selection (one generation, one sex on both sides, or the
whole sample) folded to one cell per relationship category, from which the
pair count, the 2×2 affection table and the liability correlation follow.

Factor layout, per D4.2 of the relationship-moments plan: the first member
carries ``generation`` (every distinct value, founders included; a frame
without the column gets one level), ``sex`` and ``affected{t}`` per trait;
the second member carries ``sex`` and ``affected{t}``. Level mapping is
pedigree-graph's, so a generation strata selects by its actual number.
"""

from __future__ import annotations

__all__ = ["Stratum", "relationship_moments_for"]

from typing import TYPE_CHECKING

import numpy as np

from simace.core.relationships import RELATIONSHIP_TYPES

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import PedigreeGraph, PedigreeView, RelationshipMoments

    type _Frame = pd.DataFrame | pl.DataFrame

TRAITS: tuple[int, ...] = (1, 2)


def relationship_moments_for(df: _Frame, source: PedigreeGraph | PedigreeView) -> RelationshipMoments:
    """Reduce every ``RELATIONSHIP_TYPES`` pair in *source* over the rows of *df*.

    *df* holds one row per receiver row of *source*, in the same order (the
    analysis sample hydrated with its pedigree columns, or the view of those
    ids). Rows need ``sex``, ``affected{t}`` and ``liability{t}`` for each
    trait in ``TRAITS``; ``generation`` is optional.

    Returns:
        A table over ``category``, ``first_generation``, ``first_sex``,
        ``first_affected{t}``, ``second_sex`` and ``second_affected{t}`` with
        value columns ``liability{t}`` and the ``first × second`` product of
        each.
    """
    n = len(df)
    generation = df["generation"].to_numpy() if "generation" in df.columns else np.zeros(n, dtype=np.int64)
    sex = df["sex"].to_numpy()
    affected = {f"affected{t}": df[f"affected{t}"].to_numpy().astype(bool) for t in TRAITS}
    return source.relationship_moments(
        categories=list(RELATIONSHIP_TYPES),
        first={"generation": generation, "sex": sex, **affected},
        second={"sex": sex, **affected},
        values={f"liability{t}": df[f"liability{t}"].to_numpy() for t in TRAITS},
    )


def select_levels(moments: RelationshipMoments, **levels: object) -> RelationshipMoments:
    """``moments.select`` where a level absent from an axis selects nothing rather than raising.

    A report generation with no rows in the frame, or a sex with no
    individuals, yields zero pairs everywhere, as the mask-based selection
    it replaces did.
    """
    wanted = {name: [value] if value in moments.axis(name).levels else [] for name, value in levels.items()}
    return moments.select(**wanted)


class Stratum:
    """One selection of the moments folded to a cell per relationship category."""

    def __init__(self, moments: RelationshipMoments) -> None:
        self._moments = moments
        self._folded = moments.sum(*(axis.name for axis in moments.axes if axis.name != "category"))
        self.categories: tuple[str, ...] = self._folded.categories

    @property
    def n_pairs(self) -> np.ndarray:
        """int64 pair count per category."""
        return self._folded.counts

    def index(self, category: str) -> int:
        """Position of *category* along the category axis."""
        return self.categories.index(category)

    def report_positions(self) -> list[tuple[str, int]]:
        """``(code, position)`` for each ``RELATIONSHIP_TYPES`` code, in that order."""
        return [(code, self.index(code)) for code in RELATIONSHIP_TYPES]

    def affection_table(self, first_trait: int, second_trait: int | None = None) -> np.ndarray:
        """int64 ``(n_categories, 2, 2)`` counts indexed ``[category, first affected, second affected]``.

        A trait affected for nobody (or everybody) has one level in the
        moments; the missing level's row or column is zero here.
        """
        axis_a = f"first_affected{first_trait}"
        axis_b = f"second_affected{first_trait if second_trait is None else second_trait}"
        raw = self._moments.table(axis_a, axis_b)
        out = np.zeros((raw.shape[0], 2, 2), dtype=np.int64)
        ia = self._moments.axis(axis_a).levels.astype(np.int64)
        ib = self._moments.axis(axis_b).levels.astype(np.int64)
        out[:, ia[:, None], ib[None, :]] = raw
        return out

    def liability_r(self, trait: int) -> np.ndarray:
        """Pearson liability correlation per category; ``0.0`` where a side is constant.

        ``0.0`` is what the single-pass ``_pearsonr_core`` kernel returns on
        a zero denominator, so the report keeps that value.
        """
        r = self._folded.pearson(f"first.liability{trait}", f"second.liability{trait}")
        return np.where(np.isnan(r), 0.0, r)
