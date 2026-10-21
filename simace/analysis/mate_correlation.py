"""Mate correlations of pedigree columns, computed by pg-phenotype.

A Mating Pair is a distinct mother and father, both rows of the pedigree, of
at least one child; each pair counts once however many children it has. A
child whose parent is severed (``-1``) or names no row makes no pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pg_phenotype import Pedigree, Trait
from pg_phenotype.assortative import mate_correlation

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import polars as pl

__all__ = ["mate_correlation_matrix", "mate_pedigree"]


def mate_pedigree(df: pd.DataFrame | pl.DataFrame) -> Pedigree:
    """The ``id``/``mother``/``father`` columns of *df*, validated once for several calls."""
    return Pedigree({c: df[c].to_numpy() for c in ("id", "mother", "father")})


def _known(values: np.ndarray) -> np.ndarray:
    """*values* as float64 with ``inf`` and ``-inf`` missing (NaN): pg-phenotype refuses them."""
    out = values.astype(np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def _varies(values: np.ndarray) -> bool:
    """Whether pg-phenotype accepts *values* as a trait: some known value, and not only one."""
    known = values[~np.isnan(values)]
    return known.size > 0 and known.min() != known.max()


def mate_correlation_matrix(
    pedigree: Pedigree, df: pd.DataFrame | pl.DataFrame, columns: Sequence[str]
) -> tuple[np.ndarray, int]:
    """Pearson correlation of the mothers' and fathers' *columns* over the Mating Pairs.

    Args:
        pedigree: :func:`mate_pedigree` of *df*.
        df: The pedigree rows, holding *columns*.
        columns: One or two continuous columns.

    Returns:
        ``(matrix, n_pairs)``: ``matrix[i, j]`` correlates the mothers'
        ``columns[i]`` with the fathers' ``columns[j]``, NaN where either column
        is constant or the pairs give no estimate (fewer than two, or one side
        constant among them). A NaN or infinite value is missing: a pair
        with one leaves the cells that read it.
    """
    values = [_known(df[c].to_numpy()) for c in columns]
    matrix = np.full((len(columns), len(columns)), np.nan)
    usable = [i for i, v in enumerate(values) if _varies(v)]
    if not usable:
        if len(df) < 2:
            return matrix, 0
        # Count the pairs with the row index, a trait that always varies.
        res = mate_correlation(pedigree, Trait(np.arange(len(df), dtype=np.float64), "continuous"), inference=False)
        return matrix, res.sample.n_total
    res = mate_correlation(pedigree, [Trait(values[i], "continuous") for i in usable], inference=False)
    for cell in res.cells:
        if cell.primary.value is not None:
            matrix[usable[cell.mother_trait], usable[cell.father_trait]] = cell.primary.value
    return matrix, res.sample.n_total
