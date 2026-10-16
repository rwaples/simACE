"""Midparent-offspring regressions, fitted by pg-phenotype.

Each caller keeps its own minimum-count rule: the per-rep stats report values
from ten children, the validation checks need more than a hundred, and the
scenario-comparison plots need ten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pg_phenotype.regression import parent_offspring

if TYPE_CHECKING:
    import numpy as np
    from pg_phenotype import Pedigree, Trait
    from pg_phenotype.regression import Fit, ParentOffspring

__all__ = ["midparent_offspring", "stratum_line"]


def midparent_offspring(pedigree: Pedigree, trait: Trait, stratum: np.ndarray | None = None) -> ParentOffspring:
    """The midparent-offspring line of *trait*, point estimates and OLS SE only.

    A child is placed in a stratum by its own *stratum* label, and its parents
    are read from every row of *pedigree*.
    """
    return parent_offspring(pedigree, trait, stratum=stratum, lines="midparent", inference=False)


def stratum_line(fits: ParentOffspring, label: int) -> Fit | None:
    """The midparent line of stratum *label* of stratified *fits*, ``None`` when no row has that label."""
    assert fits.strata is not None, "fits were not stratified"
    subset = fits.strata.get(label)
    return None if subset is None else subset.midparent
