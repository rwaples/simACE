"""Observed prevalence per trait, overall and by generation.

Kept outside :mod:`simace.analysis.stats` so the ``cohort`` stage can summarize
the phenotyped population without importing the stats package (numba,
pedigree-graph). :mod:`simace.analysis.stats.incidence` re-exports it.
"""

from __future__ import annotations

__all__ = ["compute_prevalence"]

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def compute_prevalence(df: pd.DataFrame | pl.DataFrame) -> dict[str, Any]:
    """Compute observed prevalence for each trait.

    Args:
        df: Phenotype DataFrame with ``affected1`` and ``affected2`` columns.
            If a ``generation`` column is present, per-generation prevalence
            is also reported under ``by_generation``.

    Returns:
        Dict with ``trait1`` and ``trait2`` marginal prevalence fractions, and
        (when ``generation`` is present) a ``by_generation`` subkey mapping
        ``int(generation) -> {"trait1": float, "trait2": float}``.
    """
    result: dict[str, Any] = {
        "trait1": float(df["affected1"].to_numpy().mean()),
        "trait2": float(df["affected2"].to_numpy().mean()),
    }
    if "generation" in df.columns:
        gens = df["generation"].to_numpy()
        a1 = df["affected1"].to_numpy().astype(bool)
        a2 = df["affected2"].to_numpy().astype(bool)
        result["by_generation"] = {
            int(g): {"trait1": float(a1[gens == g].mean()), "trait2": float(a2[gens == g].mean())}
            for g in np.unique(gens)
        }
    return result
