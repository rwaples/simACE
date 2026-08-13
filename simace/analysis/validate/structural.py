"""Structural-integrity checks for the pedigree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._common import _result

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from simace.core.pedigree_arrays import PedigreeArrays


def validate_structural(df: pd.DataFrame | pl.DataFrame, params: dict[str, Any], ped: PedigreeArrays) -> dict[str, Any]:
    """Validate structural integrity of the pedigree.

    Checks contiguous IDs, valid parent references, sex-parent consistency,
    and balanced sex ratio.

    Args:
        df: Pedigree DataFrame with columns id, sex, mother, father.
        params: Scenario parameters; requires keys ``N`` and ``G_ped``.
        ped: The same pedigree as id-addressable arrays.

    Returns:
        Dict of check-name to result dicts (keys: passed, details, …).
    """
    results = {}
    N = params["N"]
    ngen = params["G_ped"]
    expected_total = N * ngen

    # ID integrity
    ids = df["id"].to_numpy()
    expected_ids = np.arange(expected_total)
    ids_contiguous = np.array_equal(np.sort(ids), expected_ids)
    results["id_integrity"] = _result(
        ids_contiguous and len(df) == expected_total,
        f"Expected {expected_total} contiguous IDs, found {len(df)} individuals",
        expected_count=expected_total,
        observed_count=len(df),
    )

    # Parent references: valid IDs (0..expected_total-1) or -1 for founders
    mother_vals = df["mother"].to_numpy()
    father_vals = df["father"].to_numpy()
    mothers_valid = (((mother_vals >= 0) & (mother_vals < expected_total)) | (mother_vals == -1)).all()
    fathers_valid = (((father_vals >= 0) & (father_vals < expected_total)) | (father_vals == -1)).all()
    id_vals = df["id"].to_numpy()
    no_self_parent = bool(((mother_vals != id_vals) & (father_vals != id_vals)).all())
    results["parent_references"] = _result(
        bool(mothers_valid and fathers_valid and no_self_parent),
        f"Mothers valid: {mothers_valid}, Fathers valid: {fathers_valid}, No self-parenting: {no_self_parent}",
    )

    # Sex-parent consistency (only for non-founders)
    nf_mask = mother_vals != -1
    if nf_mask.any():
        mothers = mother_vals[nf_mask]
        fathers = father_vals[nf_mask]
        # Short-circuits on an absent parent, matching the reindex this
        # replaced: a missing id produced NaN, which failed the comparison.
        mothers_female = bool(ped.contains(mothers).all()) and bool((ped.gather("sex", mothers) == 0).all())
        fathers_male = bool(ped.contains(fathers).all()) and bool((ped.gather("sex", fathers) == 1).all())
        results["sex_parent_consistency"] = _result(
            bool(mothers_female and fathers_male),
            f"Mothers female: {mothers_female}, Fathers male: {fathers_male}",
        )
    else:
        results["sex_parent_consistency"] = _result(True, "No non-founders to check")

    # Sex distribution
    sex_ratio = df["sex"].mean()
    sex_balanced = 0.45 <= sex_ratio <= 0.55
    results["sex_distribution"] = _result(
        sex_balanced,
        f"Male ratio: {sex_ratio:.3f} (expected ~0.5)",
        observed_ratio=float(sex_ratio),
    )

    return results
