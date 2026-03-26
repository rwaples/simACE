"""
ACE Simulation Validation

Validates simulation outputs for structural integrity, statistical properties,
and heritability estimates.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sim_ace.pedigree_graph import extract_sibling_pairs
from sim_ace.utils import safe_corrcoef, safe_linregress, to_native
from sim_ace.utils import validation_result as _result

logger = logging.getLogger(__name__)


def _check_variance(founders: pd.DataFrame, col: str, expected: float, tol: float = 0.1) -> dict[str, Any]:
    """Check that the variance of `col` in founders is close to `expected`."""
    var = founders[col].var()
    return _result(
        abs(var - expected) < tol,
        f"Var({col}) in founders: {var:.4f} (expected: {expected})",
        expected=expected,
        observed=float(var),
    )


def _midparent_regression(
    vals: np.ndarray, mother_idx: np.ndarray, father_idx: np.ndarray, offspring_idx: np.ndarray, label: str
) -> dict[str, Any]:
    """Run midparent-offspring regression and return result dict."""
    midparent = (vals[mother_idx] + vals[father_idx]) / 2
    offspring = vals[offspring_idx]
    reg = safe_linregress(midparent, offspring)
    if reg is not None:
        return {
            "slope": float(reg.slope),
            "intercept": float(reg.intercept),
            "r_squared": float(reg.rvalue**2),
            "details": f"Midparent-offspring {label} regression: slope={reg.slope:.4f}, R²={reg.rvalue**2:.4f}",
        }
    return {"details": f"Zero variance in midparent {label} values"}


def _count_sib_pairs_legacy(non_twin_sibs: pd.DataFrame) -> dict[str, int]:
    """Legacy implementation kept for golden testing. Use pedigree_graph.count_sib_pairs instead."""
    cols = non_twin_sibs[["id", "mother", "father"]]

    # Maternal grouping: self-merge on mother
    sib_counts = cols.groupby("mother").size()
    multi_mothers = sib_counts[sib_counts >= 2].index
    mat_sib = cols[cols["mother"].isin(multi_mothers)]

    n_full_sib = 0
    n_maternal_hs = 0
    n_offspring_with_sibs = len(mat_sib)

    if len(mat_sib) > 0:
        mat_pairs = mat_sib.merge(mat_sib, on="mother", suffixes=("_1", "_2"))
        mat_pairs = mat_pairs[mat_pairs["id_1"] < mat_pairs["id_2"]]
        same_father = mat_pairs["father_1"] == mat_pairs["father_2"]
        n_full_sib = int(same_father.sum())
        n_maternal_hs = int((~same_father).sum())

    # Count offspring with maternal half-sibs (families with >1 unique father)
    if len(mat_sib) > 0:
        n_fathers_per_mother = mat_sib.groupby("mother")["father"].nunique()
        mothers_with_hs = n_fathers_per_mother[n_fathers_per_mother > 1].index
        n_offspring_with_maternal_hs = int(mat_sib[mat_sib["mother"].isin(mothers_with_hs)].shape[0])
    else:
        n_offspring_with_maternal_hs = 0

    # Paternal grouping: self-merge on father, different mother
    pat_counts = cols.groupby("father").size()
    multi_fathers = pat_counts[pat_counts >= 2].index
    pat_sib = cols[cols["father"].isin(multi_fathers)]

    n_paternal_hs = 0
    if len(pat_sib) > 0:
        pat_pairs = pat_sib.merge(pat_sib, on="father", suffixes=("_1", "_2"))
        pat_pairs = pat_pairs[pat_pairs["id_1"] < pat_pairs["id_2"]]
        diff_mother = pat_pairs["mother_1"] != pat_pairs["mother_2"]
        n_paternal_hs = int(diff_mother.sum())

    return {
        "n_maternal_half_sib_pairs": n_maternal_hs,
        "n_paternal_half_sib_pairs": n_paternal_hs,
        "n_full_sib_pairs": n_full_sib,
        "n_offspring_with_maternal_half_sib": n_offspring_with_maternal_hs,
        "n_offspring_with_sibs": n_offspring_with_sibs,
    }


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_structural(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Validate structural integrity of the pedigree.

    Checks contiguous IDs, valid parent references, sex-parent consistency,
    and balanced sex ratio.

    Args:
        df: Pedigree DataFrame with columns id, sex, mother, father.
        params: Scenario parameters; requires keys ``N`` and ``G_ped``.

    Returns:
        Dict of check-name to result dicts (keys: passed, details, …).
    """
    results = {}
    N = params["N"]
    ngen = params["G_ped"]
    expected_total = N * ngen

    # ID integrity
    ids = df["id"].values
    expected_ids = np.arange(expected_total)
    ids_contiguous = np.array_equal(np.sort(ids), expected_ids)
    results["id_integrity"] = _result(
        ids_contiguous and len(df) == expected_total,
        f"Expected {expected_total} contiguous IDs, found {len(df)} individuals",
        expected_count=expected_total,
        observed_count=len(df),
    )

    # Parent references: valid IDs (0..expected_total-1) or -1 for founders
    mother_vals = df["mother"].values
    father_vals = df["father"].values
    mothers_valid = (((mother_vals >= 0) & (mother_vals < expected_total)) | (mother_vals == -1)).all()
    fathers_valid = (((father_vals >= 0) & (father_vals < expected_total)) | (father_vals == -1)).all()
    no_self_parent = ((df["mother"] != df["id"]) & (df["father"] != df["id"])).all()
    results["parent_references"] = _result(
        bool(mothers_valid and fathers_valid and no_self_parent),
        f"Mothers valid: {mothers_valid}, Fathers valid: {fathers_valid}, No self-parenting: {no_self_parent}",
    )

    # Sex-parent consistency (only for non-founders)
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        id_to_sex = df.set_index("id")["sex"]
        mother_sex = id_to_sex.reindex(non_founders["mother"]).values
        father_sex = id_to_sex.reindex(non_founders["father"]).values
        mothers_female = (mother_sex == 0).all()
        fathers_male = (father_sex == 1).all()
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


def validate_twins(df: pd.DataFrame, params: dict[str, Any], df_indexed: pd.DataFrame) -> dict[str, Any]:
    """Validate MZ twin properties for two-trait simulation.

    Checks bidirectional twin pointers, shared parents, identical A values
    and sex for MZ pairs, and that the observed twin rate matches the
    expected rate ``2 * p_mztwin * eligible_fraction``.

    Args:
        df: Pedigree DataFrame.
        params: Scenario parameters; requires key ``p_mztwin``.
        df_indexed: Pedigree DataFrame indexed by ``id`` for fast lookups.

    Returns:
        Dict of check-name to result dicts.
    """
    results = {}
    p_mztwin = params["p_mztwin"]

    twins_df = df[df["twin"] != -1]
    n_twins = len(twins_df)

    if n_twins == 0:
        results["twin_bidirectional"] = _result(True, "No twins found")
        results["twin_same_parents"] = _result(True, "No twins found")
        for t in [1, 2]:
            results[f"twin_same_A{t}"] = _result(True, "No twins found")
        results["twin_same_sex"] = _result(True, "No twins found")
        results["twin_rate"] = _result(
            p_mztwin < 0.01,
            f"No twins found, expected rate: {p_mztwin}",
            expected_rate=p_mztwin,
            observed_rate=0.0,
        )
        return results

    # Get unique twin pairs
    twin_ids = twins_df["id"].values
    twin_partners = twins_df["twin"].values
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]
    n_pairs = len(t1_arr)

    # Bidirectional check
    twin_col = df_indexed["twin"]
    reverse_check = twin_col.reindex(t2_arr).values
    bidirectional = np.all(reverse_check == t1_arr)
    results["twin_bidirectional"] = _result(
        bool(bidirectional),
        f"All {n_twins} twin references are bidirectional: {bidirectional}",
    )

    # Same parents
    t1_mother = df_indexed.loc[t1_arr, "mother"].values
    t2_mother = df_indexed.loc[t2_arr, "mother"].values
    t1_father = df_indexed.loc[t1_arr, "father"].values
    t2_father = df_indexed.loc[t2_arr, "father"].values
    same_parents = np.all((t1_mother == t2_mother) & (t1_father == t2_father))
    results["twin_same_parents"] = _result(
        bool(same_parents),
        f"All {n_pairs} twin pairs share parents: {same_parents}",
    )

    # Same A values and same sex - loop over traits for A
    for t in [1, 2]:
        col = f"A{t}"
        v1 = df_indexed.loc[t1_arr, col].values
        v2 = df_indexed.loc[t2_arr, col].values
        same = np.allclose(v1, v2)
        results[f"twin_same_{col}"] = _result(
            bool(same),
            f"All MZ twin pairs have identical {col} values: {same}",
        )

    # Same sex
    t1_sex = df_indexed.loc[t1_arr, "sex"].values
    t2_sex = df_indexed.loc[t2_arr, "sex"].values
    same_sex = np.all(t1_sex == t2_sex)
    results["twin_same_sex"] = _result(
        bool(same_sex),
        f"All MZ twin pairs have same sex: {same_sex}",
    )

    # Twin rate (count only non-founder twin pairs; founders have twins but no parents in pedigree)
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        n_nf = len(non_founders)
        nf_twins = non_founders[non_founders["twin"] != -1]
        nf_twin_ids = nf_twins["id"].values
        nf_twin_partners = nf_twins["twin"].values
        nf_pairs = int(np.sum(nf_twin_ids < nf_twin_partners))
        observed_rate = nf_pairs * 2 / n_nf
        # Under the mating-pair model, twins are assigned per mating with >=2
        # offspring. Use a generous range check since the expected rate depends
        # on the offspring allocation distribution.
        rate_tol = max(0.01, 3 * p_mztwin)
        rate_ok = observed_rate < rate_tol
        results["twin_rate"] = _result(
            rate_ok,
            f"Twin rate in non-founders: {observed_rate:.4f} (p_mztwin={p_mztwin:.4f}, tol: {rate_tol:.4f})",
            expected_rate=float(p_mztwin),
            observed_rate=float(observed_rate),
            twin_pairs=nf_pairs,
        )
    else:
        results["twin_rate"] = _result(True, "No non-founders to check twin rate")

    return results


def _corr_se(expected_r: float, n_pairs: int) -> float:
    """Approximate SE of Pearson correlation: (1 - r^2) / sqrt(n - 1)."""
    return (1 - expected_r**2) / np.sqrt(max(n_pairs - 1, 1))


def _corr_tolerance(expected_r: float, n_pairs: int, min_tol: float = 0.05, n_se: int = 4) -> float:
    """Compute SE-based tolerance for correlation checks."""
    se = _corr_se(expected_r, n_pairs)
    return max(n_se * se, min_tol)


def _sib_counts_from_pairs(
    sibling_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, int]:
    """Derive sibling counts from pre-extracted pair arrays."""
    full = sibling_pairs["Full sib"]
    mat = sibling_pairs["Maternal half sib"]
    pat = sibling_pairs["Paternal half sib"]
    n_full = len(full[0])
    n_mat = len(mat[0])
    n_pat = len(pat[0])

    # Individuals with any maternal sibling (full or half)
    maternal_parts: list[np.ndarray] = []
    if n_full > 0:
        maternal_parts.extend([full[0], full[1]])
    if n_mat > 0:
        maternal_parts.extend([mat[0], mat[1]])
    n_with_sibs = len(np.unique(np.concatenate(maternal_parts))) if maternal_parts else 0

    # Individuals with a maternal half-sib
    if n_mat > 0:
        n_with_mat_hs = len(np.unique(np.concatenate([mat[0], mat[1]])))
    else:
        n_with_mat_hs = 0

    return {
        "n_full_sib_pairs": n_full,
        "n_maternal_half_sib_pairs": n_mat,
        "n_paternal_half_sib_pairs": n_pat,
        "n_offspring_with_sibs": n_with_sibs,
        "n_offspring_with_maternal_half_sib": n_with_mat_hs,
    }


def validate_half_sibs(
    df: pd.DataFrame, params: dict[str, Any], sibling_pairs: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    """Validate half-sibling structure under the mating-pair model.

    Reports observed counts and proportions of full-sib, maternal half-sib,
    and paternal half-sib pairs as informational checks. With a
    zero-truncated Poisson mating model, both maternal and paternal
    half-sibs arise naturally when individuals have multiple partners.

    Args:
        df: Pedigree DataFrame with columns id, mother, father, twin.
        params: Scenario parameters; requires key ``mating_lambda``.
        sibling_pairs: Pre-extracted sibling pairs from extract_sibling_pairs().

    Returns:
        Dict of check-name to result dicts.
    """
    results = {}

    sib_info = _sib_counts_from_pairs(sibling_pairs)

    # Report sibling structure (informational — no closed-form expected value)
    total_maternal_pairs = sib_info["n_full_sib_pairs"] + sib_info["n_maternal_half_sib_pairs"]
    if total_maternal_pairs > 0:
        observed_half_sib_prop = sib_info["n_maternal_half_sib_pairs"] / total_maternal_pairs
        # Range check: at lambda=0.5, most people have 1 partner, so half-sibs
        # should be present but not dominant. Wide tolerance for any lambda.
        results["half_sib_pair_proportion"] = _result(
            True,
            f"Maternal half-sib pair proportion: {observed_half_sib_prop:.4f} "
            f"(full={sib_info['n_full_sib_pairs']}, mat_hs={sib_info['n_maternal_half_sib_pairs']}, "
            f"pat_hs={sib_info['n_paternal_half_sib_pairs']})",
            observed=float(observed_half_sib_prop),
            n_full_sib_pairs=int(sib_info["n_full_sib_pairs"]),
            n_maternal_half_sib_pairs=int(sib_info["n_maternal_half_sib_pairs"]),
            n_paternal_half_sib_pairs=int(sib_info["n_paternal_half_sib_pairs"]),
        )
    else:
        results["half_sib_pair_proportion"] = _result(True, "No maternal sibling pairs to check")

    # Offspring with maternal half-sib (informational)
    n_offspring_with_sibs = sib_info["n_offspring_with_sibs"]
    n_offspring_with_hs = sib_info["n_offspring_with_maternal_half_sib"]
    if n_offspring_with_sibs > 0:
        observed_frac = n_offspring_with_hs / n_offspring_with_sibs
        results["offspring_with_half_sib"] = _result(
            True,
            f"Offspring with maternal half-sib: {observed_frac:.4f} ({n_offspring_with_hs}/{n_offspring_with_sibs})",
            observed=float(observed_frac),
            n_offspring_with_half_sib=int(n_offspring_with_hs),
            n_offspring_with_sibs=int(n_offspring_with_sibs),
        )
    else:
        results["offspring_with_half_sib"] = _result(True, "No non-twin offspring with siblings to check")

    return results


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


def validate_statistical(df: pd.DataFrame, params: dict[str, Any], df_indexed: pd.DataFrame) -> dict[str, Any]:
    """Validate statistical properties of variance components for two traits.

    Checks founder variances for A, C, E against configured values, total
    variance close to 1.0, cross-trait correlations (rA, rC, rE), C sharing
    within households, and E independence between siblings.

    Args:
        df: Pedigree DataFrame with variance-component columns A1, C1, E1,
            A2, C2, E2.
        params: Scenario parameters; requires keys ``A1``, ``C1``, ``E1``,
            ``A2``, ``C2``, ``E2``, ``rA``, ``rC``.
        df_indexed: Pedigree DataFrame indexed by ``id``.

    Returns:
        Dict of check-name to result dicts.
    """
    results = {}

    rA_param = params.get("rA", 0)
    rC_param = params.get("rC", 0)

    founders = df[df["mother"] == -1]

    # Variance checks for both traits
    for t in [1, 2]:
        for comp in ["A", "C", "E"]:
            col = f"{comp}{t}"
            results[f"variance_{col}"] = _check_variance(founders, col, params[col])

    # Total variances
    for t in [1, 2]:
        total = sum(results[f"variance_{c}{t}"]["observed"] for c in ["A", "C", "E"])
        results[f"total_variance_trait{t}"] = _result(
            abs(total - 1.0) < 0.15,
            f"Total variance trait {t}: {total:.4f} (expected: 1.0)",
            expected=1.0,
            observed=float(total),
        )

    # Cross-trait correlations
    for comp, expected, label in [("A", rA_param, "A"), ("C", rC_param, "C")]:
        obs = safe_corrcoef(founders[f"{comp}1"].values, founders[f"{comp}2"].values)
        ok = abs(obs - expected) < 0.15 if not np.isnan(obs) else expected == 0
        results[f"cross_trait_r{label}"] = _result(
            ok,
            f"Cross-trait {label} correlation: {obs:.4f} (expected: {expected})",
            expected=expected,
            observed=float(obs),
        )

    rE_obs = safe_corrcoef(founders["E1"].values, founders["E2"].values)
    rE_ok = abs(rE_obs) < 0.1 if not np.isnan(rE_obs) else True
    results["cross_trait_rE"] = _result(
        rE_ok,
        f"Cross-trait E correlation: {rE_obs:.4f} (expected: ~0)",
        expected=0.0,
        observed=float(rE_obs),
    )

    # C inheritance: siblings should share C
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        for t in [1, 2]:
            col = f"C{t}"
            c_by_mother = non_founders.groupby("mother")[col].nunique()
            c_shared = (c_by_mother == 1).mean()
            results[f"c{t}_inheritance"] = _result(
                c_shared > 0.99,
                f"Proportion of families with shared {col}: {c_shared:.4f}",
                proportion_shared=float(c_shared),
            )
    else:
        for t in [1, 2]:
            results[f"c{t}_inheritance"] = _result(True, "No non-founders to check C inheritance")

    # E independence between siblings
    if len(non_founders) > 0:
        fam_sizes = non_founders.groupby("mother").size()
        multi_child_mothers = fam_sizes[fam_sizes >= 2].index

        if len(multi_child_mothers) > 10:
            # Vectorized: get first two E1 values per mother via groupby
            multi_child = non_founders[non_founders["mother"].isin(multi_child_mothers[:500])]
            grouped = multi_child.groupby("mother")["E1"]
            first = grouped.nth(0).values
            second = grouped.nth(1).values
            # nth returns NaN for groups with < 2 members; both arrays aligned by group
            valid = ~(np.isnan(first) | np.isnan(second))
            e1_pairs_arr = np.column_stack([first[valid], second[valid]])
            e1_pairs_arr = e1_pairs_arr[:1000]

            if len(e1_pairs_arr) > 10:
                e1, e2 = e1_pairs_arr[:, 0], e1_pairs_arr[:, 1]
                e_corr = safe_corrcoef(e1, e2)
                results["e1_independence"] = _result(
                    abs(e_corr) < 0.1,
                    f"E1 correlation between siblings: {e_corr:.4f} (expected: ~0)",
                    observed_correlation=float(e_corr),
                )
            else:
                results["e1_independence"] = _result(True, "Not enough sibling pairs to check E independence")
        else:
            results["e1_independence"] = _result(True, "Not enough sibling groups to check E independence")
    else:
        results["e1_independence"] = _result(True, "No non-founders to check E independence")

    return results


def _validate_mz_correlations(
    df: pd.DataFrame,
    A_params: dict[int, float],
    comp_vals: dict[str, np.ndarray],
    id_to_idx: pd.Series,
    results: dict[str, Any],
) -> tuple[dict[int, float | None], int]:
    """Validate MZ twin correlations. Returns (mz_pheno_corr, n_mz_pairs)."""
    twins_df = df[df["twin"] != -1]
    twin_ids = twins_df["id"].values
    twin_partners = twins_df["twin"].values
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]

    mz_pheno_corr: dict[int, float | None] = {}
    if len(t1_arr) >= 10:
        idx1 = id_to_idx.reindex(t1_arr).values.astype(int)
        idx2 = id_to_idx.reindex(t2_arr).values.astype(int)

        for t in [1, 2]:
            col = f"A{t}"
            mz_v1, mz_v2 = comp_vals[col][idx1], comp_vals[col][idx2]
            mz_corr = safe_corrcoef(mz_v1, mz_v2)
            mz_ok = mz_corr > 0.99 if not np.isnan(mz_corr) else A_params[t] == 0
            results[f"mz_twin_{col}_correlation"] = _result(
                mz_ok,
                f"MZ twin {col} correlation: {mz_corr:.4f} (expected: 1.0)",
                expected=1.0,
                observed=float(mz_corr),
                n_pairs=len(t1_arr),
            )

            P1 = mz_v1 + comp_vals[f"C{t}"][idx1] + comp_vals[f"E{t}"][idx1]
            P2 = mz_v2 + comp_vals[f"C{t}"][idx2] + comp_vals[f"E{t}"][idx2]
            pheno_corr = safe_corrcoef(P1, P2)
            mz_pheno_corr[t] = pheno_corr
            results[f"mz_twin_liability{t}_correlation"] = {
                "observed": float(pheno_corr),
                "details": f"MZ twin liability{t} correlation: {pheno_corr:.4f}",
                "n_pairs": len(t1_arr),
            }
    else:
        for t in [1, 2]:
            results[f"mz_twin_A{t}_correlation"] = _result(
                True,
                f"Not enough MZ twin pairs ({len(t1_arr)}) to compute correlation",
            )
            mz_pheno_corr[t] = None

    return mz_pheno_corr, len(t1_arr)


def _validate_dz_correlations(
    params: dict[str, Any],
    A_params: dict[int, float],
    comp_vals: dict[str, np.ndarray],
    full_sib_pairs: tuple[np.ndarray, np.ndarray],
    results: dict[str, Any],
) -> tuple[dict[int, float | None], int]:
    """Validate DZ sibling correlations. Returns (dz_pheno_corr, n_dz_pairs)."""
    idx1, idx2 = full_sib_pairs

    dz_pheno_corr: dict[int, float | None] = {}
    n_dz_pairs = len(idx1)

    max_pairs = 5000
    if n_dz_pairs > max_pairs:
        rng = np.random.default_rng(params.get("seed", 42))
        sel = rng.choice(n_dz_pairs, max_pairs, replace=False)
        idx1 = idx1[sel]
        idx2 = idx2[sel]
        n_dz_pairs = max_pairs

    if n_dz_pairs >= 10:
        for t in [1, 2]:
            col = f"A{t}"
            dz_v1, dz_v2 = comp_vals[col][idx1], comp_vals[col][idx2]
            dz_corr = safe_corrcoef(dz_v1, dz_v2)
            expected_dz = 0.5
            dz_tol = _corr_tolerance(expected_dz, n_dz_pairs)
            if np.isnan(dz_corr):
                dz_ok = A_params[t] == 0
            else:
                dz_ok = abs(dz_corr - expected_dz) < dz_tol
            results[f"dz_sibling_{col}_correlation"] = _result(
                dz_ok,
                f"DZ sibling {col} correlation: {dz_corr:.4f} (expected: ~0.5, tol: {dz_tol:.4f})",
                expected=expected_dz,
                observed=float(dz_corr),
                n_pairs=n_dz_pairs,
            )

            P1 = dz_v1 + comp_vals[f"C{t}"][idx1] + comp_vals[f"E{t}"][idx1]
            P2 = dz_v2 + comp_vals[f"C{t}"][idx2] + comp_vals[f"E{t}"][idx2]
            pheno_corr = safe_corrcoef(P1, P2)
            dz_pheno_corr[t] = pheno_corr
            results[f"dz_sibling_liability{t}_correlation"] = {
                "observed": float(pheno_corr),
                "details": f"DZ sibling liability{t} correlation: {pheno_corr:.4f}",
                "n_pairs": n_dz_pairs,
            }

    if n_dz_pairs < 10:
        for t in [1, 2]:
            results[f"dz_sibling_A{t}_correlation"] = _result(
                True,
                f"Not enough DZ sibling pairs ({n_dz_pairs}) to compute correlation",
            )
            dz_pheno_corr[t] = None

    return dz_pheno_corr, n_dz_pairs


def _validate_falconer(
    A_params: dict[int, float],
    mz_pheno_corr: dict[int, float | None],
    dz_pheno_corr: dict[int, float | None],
    n_mz_pairs: int,
    n_dz_pairs: int,
    results: dict[str, Any],
) -> None:
    """Validate Falconer heritability estimates."""
    for t in [1, 2]:
        mz_c = mz_pheno_corr.get(t)
        dz_c = dz_pheno_corr.get(t)
        if mz_c is not None and dz_c is not None and not (np.isnan(mz_c) or np.isnan(dz_c)):
            falconer = 2 * (mz_c - dz_c)
            se_mz = _corr_se(mz_c, n_mz_pairs)
            se_dz = _corr_se(dz_c, n_dz_pairs)
            se_falconer = 2 * np.sqrt(se_mz**2 + se_dz**2)
            falconer_tol = max(4 * se_falconer, 0.05)
            results[f"falconer_estimate_trait{t}"] = _result(
                abs(falconer - A_params[t]) < falconer_tol,
                f"Falconer h²{chr(8320 + t)} = 2(r_MZ - r_DZ) = {falconer:.4f} "
                f"(expected: ~{A_params[t]}, tol: {falconer_tol:.4f})",
                expected=A_params[t],
                observed=float(falconer),
            )
        else:
            results[f"falconer_estimate_trait{t}"] = _result(
                True,
                "Cannot compute Falconer estimate without both MZ and DZ correlations",
            )


def _validate_parent_offspring(
    df: pd.DataFrame,
    comp_vals: dict[str, np.ndarray],
    id_to_idx: pd.Series,
    df_indexed: pd.DataFrame,
    results: dict[str, Any],
) -> None:
    """Validate parent-offspring regression."""
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 100:
        valid_offspring = non_founders[
            non_founders["mother"].isin(df_indexed.index) & non_founders["father"].isin(df_indexed.index)
        ]

        if len(valid_offspring) > 100:
            mother_idx = id_to_idx.reindex(valid_offspring["mother"]).values.astype(int)
            father_idx = id_to_idx.reindex(valid_offspring["father"]).values.astype(int)
            offspring_idx = id_to_idx.reindex(valid_offspring["id"]).values.astype(int)

            for t in [1, 2]:
                results[f"parent_offspring_A{t}_regression"] = _midparent_regression(
                    comp_vals[f"A{t}"],
                    mother_idx,
                    father_idx,
                    offspring_idx,
                    f"A{t}",
                )
                P_vals = comp_vals[f"A{t}"] + comp_vals[f"C{t}"] + comp_vals[f"E{t}"]
                results[f"parent_offspring_liability{t}_regression"] = _midparent_regression(
                    P_vals,
                    mother_idx,
                    father_idx,
                    offspring_idx,
                    f"liability{t}",
                )
        else:
            for t in [1, 2]:
                results[f"parent_offspring_A{t}_regression"] = {
                    "details": "Not enough offspring with both parents in data"
                }
                results[f"parent_offspring_liability{t}_regression"] = {
                    "details": "Not enough offspring with both parents in data"
                }
    else:
        for t in [1, 2]:
            results[f"parent_offspring_A{t}_regression"] = {"details": "Not enough non-founders for regression"}
            results[f"parent_offspring_liability{t}_regression"] = {"details": "Not enough non-founders for regression"}


def validate_heritability(
    df: pd.DataFrame,
    params: dict[str, Any],
    df_indexed: pd.DataFrame,
    sibling_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Validate heritability estimates for two-trait simulation.

    Computes MZ twin and DZ sibling liability correlations, Falconer
    heritability estimates ``h² = 2(r_MZ - r_DZ)``, and midparent-offspring
    regressions, comparing each to expected values derived from the
    configured A parameters.

    Args:
        df: Pedigree DataFrame.
        params: Scenario parameters; requires keys ``A1``, ``A2``, ``seed``.
        df_indexed: Pedigree DataFrame indexed by ``id``.
        sibling_pairs: Pre-extracted sibling pairs from extract_sibling_pairs().

    Returns:
        Dict of check-name to result dicts, including MZ/DZ correlations,
        Falconer estimates, and parent-offspring regression slopes.
    """
    results: dict[str, Any] = {}
    A_params = {1: params["A1"], 2: params["A2"]}

    comp_vals = {}
    for comp in ["A", "C", "E"]:
        for t in [1, 2]:
            comp_vals[f"{comp}{t}"] = df_indexed[f"{comp}{t}"].values
    id_to_idx = pd.Series(np.arange(len(df_indexed)), index=df_indexed.index)

    mz_pheno_corr, n_mz_pairs = _validate_mz_correlations(
        df,
        A_params,
        comp_vals,
        id_to_idx,
        results,
    )
    dz_pheno_corr, n_dz_pairs = _validate_dz_correlations(
        params,
        A_params,
        comp_vals,
        sibling_pairs["Full sib"],
        results,
    )
    _validate_falconer(
        A_params,
        mz_pheno_corr,
        dz_pheno_corr,
        n_mz_pairs,
        n_dz_pairs,
        results,
    )
    _validate_parent_offspring(df, comp_vals, id_to_idx, df_indexed, results)

    return results


def compute_per_generation_stats(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Compute per-generation statistics for two traits.

    For each generation, computes liability mean/variance/sd and per-component
    (A, C, E) mean/variance for both traits.

    Args:
        df: Pedigree DataFrame with columns id, A1, C1, E1, A2, C2, E2.
        params: Scenario parameters; requires keys ``N`` and ``G_ped``.

    Returns:
        Dict keyed by ``"generation_{g}"`` where each value is a dict of
        summary statistics (n, liability mean/variance/sd, component mean/var).
    """
    N = params["N"]
    ngen = params["G_ped"]

    # Assign generation labels once via integer division
    gen_labels = df["id"].values // N

    results = {}
    for gen in range(1, ngen + 1):
        gen_mask = gen_labels == (gen - 1)
        gen_df = df[gen_mask]

        gen_stats: dict[str, int | float] = {"n": int(gen_mask.sum())}
        for t in [1, 2]:
            a_vals = gen_df[f"A{t}"].values
            c_vals = gen_df[f"C{t}"].values
            e_vals = gen_df[f"E{t}"].values
            liability = a_vals + c_vals + e_vals
            gen_stats[f"liability{t}_mean"] = float(liability.mean())
            gen_stats[f"liability{t}_variance"] = float(liability.var())
            gen_stats[f"liability{t}_sd"] = float(liability.std())
            for comp, vals in [("A", a_vals), ("C", c_vals), ("E", e_vals)]:
                col = f"{comp}{t}"
                gen_stats[f"{col}_mean"] = float(vals.mean())
                gen_stats[f"{col}_var"] = float(vals.var())

        results[f"generation_{gen}"] = gen_stats

    return results


def validate_population(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Validate population-level properties.

    Checks that each generation has exactly ``N`` individuals, the number of
    generations equals ``G_ped``, and the mean offspring per mother is
    approximately ``N / n_females`` (always ~2.0 for balanced sex ratios).

    Args:
        df: Pedigree DataFrame with columns id and mother.
        params: Scenario parameters; requires keys ``N``, ``G_ped``.

    Returns:
        Dict of check-name to result dicts.
    """
    results = {}
    N = params["N"]
    ngen = params["G_ped"]

    gen_assignments = df["id"].values // N
    gen_sizes = np.bincount(gen_assignments, minlength=ngen)[:ngen].tolist()

    all_correct = all(s == N for s in gen_sizes)
    results["generation_sizes"] = _result(
        all_correct,
        f"Generation sizes: {gen_sizes} (expected: {N} each)",
        expected=N,
        observed=gen_sizes,
    )

    results["generation_count"] = _result(
        len(gen_sizes) == ngen,
        f"Number of generations: {len(gen_sizes)} (expected: {ngen})",
        expected=ngen,
        observed=len(gen_sizes),
    )

    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        family_sizes = non_founders.groupby("mother").size()
        mean_fam = family_sizes.mean()
        # Mean offspring per mother is ~N / n_mothers ~= 2.0 for balanced sex
        expected_mean = 2.0
        fam_ok = abs(mean_fam - expected_mean) < expected_mean * 0.5
        results["family_size"] = _result(
            fam_ok,
            f"Mean offspring per mother: {mean_fam:.2f} (expected: ~{expected_mean:.1f})",
            expected=expected_mean,
            observed=float(mean_fam),
        )
    else:
        results["family_size"] = _result(True, "No non-founders to check family size")

    return results


def compute_family_size_distribution(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Compute offspring count distributions per parent sex.

    Args:
        df: Pedigree DataFrame with columns mother and father.
        params: Scenario parameters (unused but accepted for API consistency).

    Returns:
        Dict with keys ``"mother"`` and ``"father"``, each mapping to a dict
        of summary statistics (mean, median, std, n_parents). Empty dict if
        no non-founders exist.
    """
    non_founders = df[df["mother"] != -1]
    if len(non_founders) == 0:
        return {}

    mother_counts = non_founders.groupby("mother").size()
    father_counts = non_founders.groupby("father").size()

    result = {}
    for label, counts in [("mother", mother_counts), ("father", father_counts)]:
        result[label] = {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "std": float(counts.std()),
            "n_parents": len(counts),
        }

    return result


def validate_assortative_mating(df: pd.DataFrame, params: dict[str, Any], df_indexed: pd.DataFrame) -> dict[str, Any]:
    """Validate mate correlation on liability when assortative mating is configured.

    Extracts unique mating pairs from non-founders, computes Pearson
    correlation of mother and father liability for each trait, and checks
    against the configured ``assort1`` / ``assort2`` parameters.

    Args:
        df: Pedigree DataFrame.
        params: Scenario parameters; uses keys ``assort1``, ``assort2``.
        df_indexed: Pedigree DataFrame indexed by ``id``.

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}
    assort1 = params.get("assort1", 0.0)
    assort2 = params.get("assort2", 0.0)

    non_founders = df[df["mother"] != -1]
    if len(non_founders) == 0:
        results["mate_corr_liability1"] = _result(True, "No non-founders to check")
        results["mate_corr_liability2"] = _result(True, "No non-founders to check")
        return results

    # Extract unique mating pairs
    pairs = non_founders[["mother", "father"]].drop_duplicates()
    mother_ids = pairs["mother"].values
    father_ids = pairs["father"].values
    n_pairs = len(pairs)

    for t, expected in [(1, assort1), (2, assort2)]:
        m_liab = df_indexed.loc[mother_ids, f"liability{t}"].values
        f_liab = df_indexed.loc[father_ids, f"liability{t}"].values
        obs = safe_corrcoef(m_liab, f_liab)

        if np.isnan(obs):
            results[f"mate_corr_liability{t}"] = _result(
                True,
                f"Cannot compute mate correlation for trait {t} (zero variance)",
                expected=float(expected),
                observed=float(obs),
            )
            continue

        se = _corr_se(expected, n_pairs)
        tol = max(0.1, 3 * se)
        ok = abs(obs - expected) < tol
        results[f"mate_corr_liability{t}"] = _result(
            ok,
            f"Mate correlation liability{t}: {obs:.4f} (expected: {expected}, tol: {tol:.4f})",
            expected=float(expected),
            observed=float(obs),
            n_pairs=n_pairs,
        )

    # Cross-trait validation (only when both traits assort)
    if assort1 != 0 and assort2 != 0:
        am = params.get("assort_matrix")
        if am is not None:
            c_expected = float(np.asarray(am)[0, 1])
        else:
            rho_w = params.get("rA", 0) * np.sqrt(params.get("A1", 0) * params.get("A2", 0)) + params.get(
                "rC", 0
            ) * np.sqrt(params.get("C1", 0) * params.get("C2", 0))
            c_expected = rho_w * np.sqrt(abs(assort1 * assort2)) * np.sign(assort1 * assort2)

        for label, fi, mi in [("cross_12", 1, 2), ("cross_21", 2, 1)]:
            m_liab = df_indexed.loc[mother_ids, f"liability{fi}"].values
            f_liab = df_indexed.loc[father_ids, f"liability{mi}"].values
            obs = safe_corrcoef(m_liab, f_liab)

            if np.isnan(obs):
                results[f"mate_corr_{label}"] = _result(
                    True,
                    f"Cannot compute mate correlation {label} (zero variance)",
                    expected=float(c_expected),
                    observed=float(obs),
                )
                continue

            se = _corr_se(c_expected, n_pairs)
            tol = max(0.1, 3 * se)
            ok = abs(obs - c_expected) < tol
            results[f"mate_corr_{label}"] = _result(
                ok,
                f"Mate correlation {label}: {obs:.4f} (expected: {c_expected:.4f}, tol: {tol:.4f})",
                expected=float(c_expected),
                observed=float(obs),
                n_pairs=n_pairs,
            )

    return results


def run_validation(pedigree_path: str, params_path: str) -> dict[str, Any]:
    """Run all validation checks and return results.

    Loads a pedigree and its parameters, then runs structural, twin,
    half-sibling, statistical, heritability, and population checks.

    Args:
        pedigree_path: Path to the pedigree parquet file.
        params_path: Path to the scenario parameters YAML file.

    Returns:
        Nested dict with keys ``"structural"``, ``"twins"``, ``"half_sibs"``,
        ``"statistical"``, ``"heritability"``, ``"population"``,
        ``"per_generation"``, ``"summary"``, ``"family_size_distribution"``,
        and ``"parameters"``. The ``"summary"`` sub-dict contains
        ``passed`` (bool), ``checks_passed``, ``checks_failed``, and
        ``checks_total`` counts.
    """
    logger.info("Validating pedigree: %s", pedigree_path)
    df = pd.read_parquet(pedigree_path)
    with open(params_path, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    df_indexed = df.set_index("id")

    sibling_pairs = extract_sibling_pairs(df)

    results = {
        "structural": validate_structural(df, params),
        "twins": validate_twins(df, params, df_indexed),
        "half_sibs": validate_half_sibs(df, params, sibling_pairs),
        "statistical": validate_statistical(df, params, df_indexed),
        "heritability": validate_heritability(df, params, df_indexed, sibling_pairs),
        "population": validate_population(df, params),
        "per_generation": compute_per_generation_stats(df, params),
        "assortative_mating": validate_assortative_mating(df, params, df_indexed),
        "consanguineous_matings": validate_consanguineous_matings(df, params),
    }

    checks_passed = 0
    checks_failed = 0

    for category, checks in results.items():
        if category == "per_generation":
            continue
        for check_name, check_result in checks.items():
            if "passed" in check_result:
                if check_result["passed"]:
                    checks_passed += 1
                else:
                    checks_failed += 1
                    logger.warning(
                        "FAILED %s.%s: %s",
                        category,
                        check_name,
                        check_result.get("details", ""),
                    )

    results["summary"] = {
        "passed": checks_failed == 0,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "checks_total": checks_passed + checks_failed,
    }

    results["family_size_distribution"] = compute_family_size_distribution(df, params)
    results["parameters"] = params

    logger.info(
        "Validation complete: %d/%d checks passed",
        checks_passed,
        checks_passed + checks_failed,
    )

    return results


def cli() -> None:
    """Command-line interface for running validation."""
    from sim_ace.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Validate ACE simulation output")
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True, help="Pedigree parquet path")
    parser.add_argument("--params", required=True, help="Params YAML path")
    parser.add_argument("--output", required=True, help="Output validation YAML path")
    args = parser.parse_args()

    init_logging(args)

    results = run_validation(args.pedigree, args.params)
    results = to_native(results)

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
