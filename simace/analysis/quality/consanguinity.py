"""Consanguineous-mating detection and grandparent-link reconciliation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._common import _info, _result

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from simace.core.pedigree_arrays import PedigreeArrays


def validate_consanguineous_matings(
    df: pd.DataFrame | pl.DataFrame, params: dict[str, Any], ped: PedigreeArrays
) -> dict[str, Any]:
    """Detect consanguineous matings and reconcile grandparent-link discrepancy.

    When ``pair_partners()`` randomly pairs individuals, half-siblings (or
    full siblings) may be matched.  Their offspring have fewer than 4
    distinct grandparents, which reduces the grandparent-grandchild pair
    count relative to the naive expectation of 4 × n_eligible.

    This check:
    1. Identifies all mating pairs where partners share one or both parents.
    2. Computes the expected and observed grandparent-grandchild pair counts.
    3. Verifies that the discrepancy is fully explained by consanguineous
       matings.

    Args:
        df: Pedigree DataFrame with columns id, mother, father.
        params: Scenario parameters (accepted for API consistency).
        ped: The same pedigree as id-addressable arrays.

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}

    ids = ped["id"]
    mothers = ped["mother"]
    fathers = ped["father"]

    # Both parents must be present. On a full pedigree this is exactly
    # "not a founder", since parents are always set together. On an
    # ascertained one it also excludes rows whose parent was severed to -1
    # (ascertainment severs mother and father independently), which the
    # previous id-indexed lookup silently read as the pedigree's last row.
    has_parents = ped.contains(mothers) & ped.contains(fathers)

    # Identify individuals in gen >= 2 (parents are non-founders, so grandparents exist)
    mothers_have_parents = np.zeros(len(ids), dtype=bool)
    mothers_have_parents[has_parents] = ped.gather("mother", mothers[has_parents]) != -1
    eligible = has_parents & mothers_have_parents  # gen >= 2

    eligible_ids = ids[eligible]
    eligible_mothers = mothers[eligible]
    eligible_fathers = fathers[eligible]

    if len(eligible_ids) == 0:
        results["consanguineous_count"] = _info("No individuals with grandparents in pedigree")
        return results

    # Look up all 4 grandparents for eligible individuals
    mgm = ped.gather("mother", eligible_mothers)  # maternal grandmother
    mgf = ped.gather("father", eligible_mothers)  # maternal grandfather
    fgm = ped.gather("mother", eligible_fathers)  # paternal grandmother
    fgf = ped.gather("father", eligible_fathers)  # paternal grandfather

    # Count distinct grandparents per individual (vectorized via sorted rows)
    gp_stack = np.column_stack([mgm, mgf, fgm, fgf])  # (n_eligible, 4)
    gp_sorted = np.sort(gp_stack, axis=1)
    n_distinct = 1 + (gp_sorted[:, 1:] != gp_sorted[:, :-1]).sum(axis=1)
    observed_gp_links = int(n_distinct.sum())
    expected_gp_links = len(eligible_ids) * 4
    total_missing = expected_gp_links - observed_gp_links

    # Identify consanguineous matings (vectorized)
    # Encode (mother, father) as single int64 key for fast np.unique on 1D array
    # int64 cast required: max_id² overflows int32
    max_id = int(ids.max()) + 1
    pair_keys = eligible_mothers.astype(np.int64) * max_id + eligible_fathers.astype(np.int64)
    unique_keys, _inverse, pair_counts = np.unique(pair_keys, return_inverse=True, return_counts=True)
    mp_m = unique_keys // max_id  # mothers in each mating pair
    mp_f = unique_keys % max_id  # fathers in each mating pair
    # Check which parent IDs are shared between mates
    share_mother = ped.gather("mother", mp_m) == ped.gather("mother", mp_f)
    share_father = ped.gather("father", mp_m) == ped.gather("father", mp_f)
    shared_count = share_mother.astype(np.int64) + share_father.astype(np.int64)
    is_consanguineous = shared_count > 0

    n_half_sib_matings = int((shared_count == 1).sum())
    n_full_sib_matings = int((shared_count == 2).sum())
    explained_missing = int((shared_count[is_consanguineous] * pair_counts[is_consanguineous]).sum())

    # Informational: report counts
    results["consanguineous_count"] = _info(
        f"Consanguineous matings: {n_half_sib_matings} half-sib, "
        f"{n_full_sib_matings} full-sib "
        f"(total missing GP links: {total_missing})",
        n_half_sib_matings=n_half_sib_matings,
        n_full_sib_matings=n_full_sib_matings,
        total_missing_gp_links=total_missing,
    )

    # Hard check: reconciliation
    reconciled = explained_missing == total_missing
    results["grandparent_reconciliation"] = _result(
        reconciled,
        f"Grandparent links: expected={expected_gp_links}, observed={observed_gp_links}, "
        f"explained_missing={explained_missing}, actual_missing={total_missing}",
        expected_gp_links=expected_gp_links,
        observed_gp_links=observed_gp_links,
        explained_missing=explained_missing,
    )

    return results
