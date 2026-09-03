"""Filter a pedigree DataFrame to a set of observed IDs plus their ancestors."""

from __future__ import annotations

import numpy as np
import polars as pl

from simace.core.pedigree_arrays import PedigreeArrays

__all__ = ["filter_pedigree_to_observed"]


def filter_pedigree_to_observed(
    df_ped: pl.DataFrame,
    observed_ids: np.ndarray | pl.Series,
) -> pl.DataFrame:
    """Restrict ``df_ped`` to ``observed_ids`` plus all ancestors needed for kinship.

    Walks parent pointers in ``df_ped`` from the observed set until fixed point,
    one vectorised generation per step. Ancestors absent from ``df_ped`` (e.g.
    removed by pedigree dropout) are not added. Returns a copy of ``df_ped``
    filtered to the closure, preserving original row order.

    Args:
        df_ped: Pedigree with ``id``, ``mother``, ``father`` columns. Ids must be
            unique; missing parents must be encoded as ``-1``.
        observed_ids: IDs to seed the closure; duplicates are fine. Must be a
            subset of ``df_ped["id"]``; raises ``ValueError`` otherwise.

    Returns:
        Filtered ``df_ped`` containing rows for observed IDs and every ancestor
        reachable through parent pointers.
    """
    observed = np.asarray(observed_ids)
    ped = PedigreeArrays.from_frame(df_ped)
    in_ped = ped.contains(observed)
    if not in_ped.all():
        missing = np.unique(observed[~in_ped])
        preview = missing[:10].tolist()
        raise ValueError(
            f"filter_pedigree_to_observed: {len(missing)} observed id(s) not in "
            f"df_ped (first {min(len(missing), 10)}: {preview})"
        )

    mother = ped["mother"]
    father = ped["father"]
    keep = np.zeros(len(ped), dtype=bool)
    keep[ped.positions(observed)] = True
    frontier = np.flatnonzero(keep)
    while frontier.size:
        parent_ids = np.concatenate([mother[frontier], father[frontier]])
        parent_rows = ped.positions(parent_ids[ped.contains(parent_ids)])
        # Mask arithmetic dedupes the next generation; np.unique on the rows
        # was the largest cost of the walk on deep pedigrees.
        added = np.zeros(len(ped), dtype=bool)
        added[parent_rows] = True
        added &= ~keep
        keep |= added
        frontier = np.flatnonzero(added)

    return df_ped.filter(pl.Series(keep))
