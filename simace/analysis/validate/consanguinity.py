"""Consanguineous-mating detection and grandparent-link reconciliation."""

from typing import Any

import numpy as np
import pandas as pd

from ._common import _result


def validate_consanguineous_matings(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
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

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}

    ids = df["id"].values
    mothers = df["mother"].values
    fathers = df["father"].values

    # Build parent lookup arrays indexed by id (assumes contiguous 0-based ids)
    n = len(ids)
    mother_of = np.full(n, -1, dtype=np.int64)
    father_of = np.full(n, -1, dtype=np.int64)
    mother_of[ids] = mothers
    father_of[ids] = fathers

    # Identify individuals in gen >= 2 (parents are non-founders, so grandparents exist)
    has_parents = mothers != -1
    mothers_have_parents = np.where(has_parents, mother_of[mothers] != -1, False)
    eligible = has_parents & mothers_have_parents  # gen >= 2

    eligible_ids = ids[eligible]
    eligible_mothers = mothers[eligible]
    eligible_fathers = fathers[eligible]

    if len(eligible_ids) == 0:
        results["consanguineous_count"] = _result(True, "No individuals with grandparents in pedigree")
        return results

    # Look up all 4 grandparents for eligible individuals
    mgm = mother_of[eligible_mothers]  # maternal grandmother
    mgf = father_of[eligible_mothers]  # maternal grandfather
    fgm = mother_of[eligible_fathers]  # paternal grandmother
    fgf = father_of[eligible_fathers]  # paternal grandfather

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
    share_mother = mother_of[mp_m] == mother_of[mp_f]
    share_father = father_of[mp_m] == father_of[mp_f]
    shared_count = share_mother.astype(np.int64) + share_father.astype(np.int64)
    is_consanguineous = shared_count > 0

    n_half_sib_matings = int((shared_count == 1).sum())
    n_full_sib_matings = int((shared_count == 2).sum())
    explained_missing = int((shared_count[is_consanguineous] * pair_counts[is_consanguineous]).sum())

    # Informational: report counts
    results["consanguineous_count"] = _result(
        True,
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
