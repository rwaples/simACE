"""Filter a pedigree DataFrame to a set of observed IDs plus their ancestors."""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["filter_pedigree_to_observed"]


def filter_pedigree_to_observed(
    df_ped: pl.DataFrame,
    observed_ids: np.ndarray | pl.Series,
) -> pl.DataFrame:
    """Restrict ``df_ped`` to ``observed_ids`` plus all ancestors needed for kinship.

    Iteratively walks parent pointers in ``df_ped`` from the observed set until
    fixed point, one vectorised generation per step.  Ancestors absent from ``df_ped`` (e.g. removed by pedigree
    dropout) are not added.  Returns a copy of ``df_ped`` filtered to the
    closure, preserving original row order.

    Args:
        df_ped: Pedigree with ``id``, ``mother``, ``father`` columns.  Missing
            parents must be encoded as ``-1``.
        observed_ids: IDs to seed the closure.  Must be a subset of
            ``df_ped["id"]``; raises ``ValueError`` otherwise.

    Returns:
        Filtered ``df_ped`` containing rows for observed IDs and every ancestor
        reachable through parent pointers.
    """
    observed = np.unique(np.asarray(observed_ids))
    all_ids = df_ped["id"].to_numpy()
    in_ped = np.isin(observed, all_ids)
    if not in_ped.all():
        missing = observed[~in_ped]
        preview = missing[:10].tolist()
        raise ValueError(
            f"filter_pedigree_to_observed: {len(missing)} observed id(s) not in "
            f"df_ped (first {min(len(missing), 10)}: {preview})"
        )

    order = np.argsort(all_ids, kind="stable")
    sorted_ids = all_ids[order]

    def rows_of(ids: np.ndarray) -> np.ndarray:
        """Row indices of ``ids`` present in ``df_ped``; absent ids are dropped."""
        if sorted_ids.size == 0:
            return np.empty(0, dtype=np.intp)
        pos = np.minimum(np.searchsorted(sorted_ids, ids), sorted_ids.size - 1)
        return order[pos[sorted_ids[pos] == ids]]

    mother = df_ped["mother"].to_numpy()
    father = df_ped["father"].to_numpy()
    keep = np.zeros(all_ids.size, dtype=bool)
    frontier = rows_of(observed)
    keep[frontier] = True
    while frontier.size:
        parent_ids = np.concatenate([mother[frontier], father[frontier]])
        parent_rows = rows_of(parent_ids[parent_ids >= 0])
        frontier = np.unique(parent_rows[~keep[parent_rows]])
        keep[frontier] = True

    return df_ped.filter(pl.Series(keep))
