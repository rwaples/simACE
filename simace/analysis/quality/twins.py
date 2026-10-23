"""MZ-twin validation checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._common import _info, _result

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from simace.core.pedigree_arrays import PedigreeArrays


def validate_twins(df: pd.DataFrame | pl.DataFrame, params: dict[str, Any], ped: PedigreeArrays) -> dict[str, Any]:
    """Validate MZ twin properties for two-trait simulation.

    Checks bidirectional twin pointers, shared parents, identical A values
    and sex for MZ pairs, and that the observed twin-individual rate among
    non-founders matches ``2q``, ``q = p_mztwin / (1 + p_mztwin)``: each
    generation draws ``Binomial(N, q)`` MZ pairs (``place_twins``), so the
    non-founders' pairs are ``Binomial(n, q)``, scored within 4 SE.

    Under ``mating_model="wright_fisher"`` no MZ twins are produced by
    design (see ADR 0002), so ``expected_rate`` is ``0.0`` and any
    twin present in the pedigree is a violation that fails the rate
    check.  Structural checks still run when twins are present so a
    corrupted WF output is also flagged for malformed twin records.

    Args:
        df: Pedigree DataFrame.
        params: Scenario parameters; requires key ``p_mztwin`` (and optional
            ``mating_model``, defaulted to ``"standard"``).
        ped: The same pedigree as id-addressable arrays.

    Returns:
        Dict of check-name to result dicts.
    """
    results = {}
    mating_model = params.get("mating_model", "standard")
    p_mztwin = params["p_mztwin"]
    is_wf = mating_model == "wright_fisher"
    pair_share = 0.0 if is_wf else float(p_mztwin) / (1 + float(p_mztwin))
    expected_rate = 2 * pair_share

    twin_all = df["twin"].to_numpy()
    id_all = df["id"].to_numpy()
    twin_mask = twin_all != -1
    n_twins = int(twin_mask.sum())

    if n_twins == 0:
        no_twins_msg = "No twins expected under mating_model=wright_fisher" if is_wf else "No twins found"
        # No twin records, so the structural twin checks have nothing to test.
        results["twin_bidirectional"] = _info(no_twins_msg)
        results["twin_same_parents"] = _info(no_twins_msg)
        for t in [1, 2]:
            results[f"twin_same_A{t}"] = _info(no_twins_msg)
        results["twin_same_sex"] = _info(no_twins_msg)
        # The rate is still tested: zero twins is the WF invariant, and under
        # standard mating it fails unless p_mztwin is small enough that zero
        # observed twins is plausible.
        rate_pass = True if is_wf else p_mztwin < 0.01
        rate_msg = (
            "No twins expected under mating_model=wright_fisher"
            if is_wf
            else f"No twins found, expected rate: {p_mztwin}"
        )
        results["twin_rate"] = _result(
            rate_pass,
            rate_msg,
            expected_rate=expected_rate,
            observed_rate=0.0,
        )
        return results

    # Get unique twin pairs
    twin_ids = id_all[twin_mask]
    twin_partners = twin_all[twin_mask]
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]
    n_pairs = len(t1_arr)

    # Bidirectional check. Tolerates a partner missing from the pedigree the
    # way the reindex it replaces did: absent means the check simply fails.
    partners_present = ped.contains(t2_arr)
    bidirectional = bool(partners_present.all()) and np.all(ped.gather("twin", t2_arr) == t1_arr)
    results["twin_bidirectional"] = _result(
        bool(bidirectional),
        f"All {n_twins} twin references are bidirectional: {bidirectional}",
    )

    # Same parents
    t1_mother = ped.gather("mother", t1_arr)
    t2_mother = ped.gather("mother", t2_arr)
    t1_father = ped.gather("father", t1_arr)
    t2_father = ped.gather("father", t2_arr)
    same_parents = np.all((t1_mother == t2_mother) & (t1_father == t2_father))
    results["twin_same_parents"] = _result(
        bool(same_parents),
        f"All {n_pairs} twin pairs share parents: {same_parents}",
    )

    # Same A values and same sex - loop over traits for A
    for t in [1, 2]:
        col = f"A{t}"
        v1 = ped.gather(col, t1_arr)
        v2 = ped.gather(col, t2_arr)
        same = np.allclose(v1, v2)
        results[f"twin_same_{col}"] = _result(
            bool(same),
            f"All MZ twin pairs have identical {col} values: {same}",
        )

    # Same sex
    t1_sex = ped.gather("sex", t1_arr)
    t2_sex = ped.gather("sex", t2_arr)
    same_sex = np.all(t1_sex == t2_sex)
    results["twin_same_sex"] = _result(
        bool(same_sex),
        f"All MZ twin pairs have same sex: {same_sex}",
    )

    # Twin rate (count only non-founder twin pairs; founders have twins but no parents in pedigree)
    nf_mask = df["mother"].to_numpy() != -1
    if nf_mask.any():
        n_nf = int(nf_mask.sum())
        nf_twin_mask = nf_mask & twin_mask
        nf_twin_ids = id_all[nf_twin_mask]
        nf_twin_partners = twin_all[nf_twin_mask]
        nf_pairs = int(np.sum(nf_twin_ids < nf_twin_partners))
        observed_rate = nf_pairs * 2 / n_nf
        if is_wf:
            # WF documents zero twins (ADR 0002).  Any presence is a
            # violation of the no-MZ-twins invariant.
            rate_ok = nf_pairs == 0
            rate_msg = (
                f"Twin rate under mating_model=wright_fisher: {observed_rate:.4f} "
                f"(expected 0.0); pedigree contains {nf_pairs} unexpected twin pair(s)."
            )
        else:
            # +1 pair keeps the bound from rejecting on a single pair at tiny n.
            pair_tol = 4 * np.sqrt(n_nf * pair_share * (1 - pair_share)) + 1
            rate_ok = abs(nf_pairs - n_nf * pair_share) <= pair_tol
            rate_msg = (
                f"Twin rate in non-founders: {observed_rate:.4f} (expected {expected_rate:.4f} "
                f"from p_mztwin={p_mztwin:.4f}; {nf_pairs} pairs, tol ±{pair_tol:.1f})"
            )
        results["twin_rate"] = _result(
            rate_ok,
            rate_msg,
            expected_rate=expected_rate,
            observed_rate=float(observed_rate),
            twin_pairs=nf_pairs,
        )
    else:
        results["twin_rate"] = _info("No non-founders to check twin rate")

    return results
