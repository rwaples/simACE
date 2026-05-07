"""Filter a pedigree DataFrame to a set of observed IDs plus their ancestors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["filter_pedigree_to_observed"]


def filter_pedigree_to_observed(
    df_ped: pd.DataFrame,
    observed_ids: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Restrict ``df_ped`` to ``observed_ids`` plus all ancestors needed for kinship.

    Iteratively walks parent pointers in ``df_ped`` from the observed set until
    fixed point.  Ancestors absent from ``df_ped`` (e.g. removed by pedigree
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

    parents = dict(
        zip(
            all_ids,
            zip(df_ped["mother"].to_numpy(), df_ped["father"].to_numpy(), strict=True),
            strict=True,
        )
    )

    closure: set = set(observed.tolist())
    frontier = set(observed.tolist())
    while frontier:
        next_frontier: set = set()
        for ind_id in frontier:
            for parent_id in parents[ind_id]:
                if parent_id < 0 or parent_id in closure:
                    continue
                if parent_id in parents:
                    next_frontier.add(parent_id)
        closure.update(next_frontier)
        frontier = next_frontier

    keep_mask = np.isin(all_ids, list(closure))
    return df_ped.loc[keep_mask].copy()
