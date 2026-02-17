"""
ACE Simulation Validation

Validates simulation outputs for structural integrity, statistical properties,
and heritability estimates.
"""

from __future__ import annotations

import argparse
from typing import Any

import logging

import numpy as np
import pandas as pd
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning nan if either array has zero variance."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return np.corrcoef(x, y)[0, 1]


def safe_linregress(x: np.ndarray, y: np.ndarray) -> Any:
    """Run linear regression, returning None if x has zero variance."""
    if np.std(x) < 1e-10:
        return None
    return stats.linregress(x, y)


def to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for YAML serialization."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


# ---------------------------------------------------------------------------
# Helper functions to reduce repetition
# ---------------------------------------------------------------------------


def _result(passed: bool, details: str, **extra: Any) -> dict[str, Any]:
    """Build a result dict with passed/details and optional extra keys."""
    d = {"passed": passed, "details": details}
    d.update(extra)
    return d


def _check_variance(founders: pd.DataFrame, col: str, expected: float, tol: float = 0.1) -> dict[str, Any]:
    """Check that the variance of `col` in founders is close to `expected`."""
    var = founders[col].var()
    return _result(
        abs(var - expected) < tol,
        f"Var({col}) in founders: {var:.4f} (expected: {expected})",
        expected=expected,
        observed=float(var),
    )



def _midparent_regression(vals: np.ndarray, mother_idx: np.ndarray, father_idx: np.ndarray, offspring_idx: np.ndarray, label: str) -> dict[str, Any]:
    """Run midparent-offspring regression and return result dict."""
    midparent = (vals[mother_idx] + vals[father_idx]) / 2
    offspring = vals[offspring_idx]
    reg = safe_linregress(midparent, offspring)
    if reg is not None:
        return {
            "slope": float(reg.slope),
            "intercept": float(reg.intercept),
            "r_squared": float(reg.rvalue ** 2),
            "details": f"Midparent-offspring {label} regression: slope={reg.slope:.4f}, R²={reg.rvalue**2:.4f}",
        }
    return {"details": f"Zero variance in midparent {label} values"}


def _count_sib_pairs(non_twin_sibs: pd.DataFrame) -> dict[str, int]:
    """Count full-sib, maternal half-sib, and paternal half-sib pairs.

    Uses vectorized merge instead of Python loops for speed.
    Returns dict with counts only (no pair lists).
    """
    cols = non_twin_sibs[["id", "mother", "father"]]

    # Maternal grouping: self-merge on mother
    sib_counts = cols.groupby("mother").size()
    multi_mothers = sib_counts[sib_counts >= 2].index
    mat_sib = cols[cols["mother"].isin(multi_mothers)]

    n_full_sib = 0
    n_maternal_hs = 0
    n_offspring_with_sibs = int(len(mat_sib))

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
        n_offspring_with_maternal_hs = int(
            mat_sib[mat_sib["mother"].isin(mothers_with_hs)].shape[0]
        )
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
    """Validate structural integrity of the pedigree."""
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

    # Parent references: valid IDs or -1 for founders
    valid_ids = set(df["id"].values) | {-1}
    mothers_valid = df["mother"].isin(valid_ids).all()
    fathers_valid = df["father"].isin(valid_ids).all()
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
    """Validate MZ twin properties for two-trait simulation."""
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

    # Twin rate
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        observed_rate = n_pairs * 2 / len(non_founders)
        se_rate = np.sqrt(p_mztwin * (1 - p_mztwin) / len(non_founders))
        rate_tol = max(4 * se_rate, 0.005)
        rate_ok = abs(observed_rate - p_mztwin) < rate_tol
        results["twin_rate"] = _result(
            rate_ok,
            f"Twin rate in non-founders: {observed_rate:.4f} (param: {p_mztwin}, tol: {rate_tol:.4f})",
            expected_rate=p_mztwin,
            observed_rate=float(observed_rate),
            twin_pairs=n_pairs,
        )
    else:
        results["twin_rate"] = _result(True, "No non-founders to check twin rate")

    return results


def _corr_se(expected_r: float, n_pairs: int) -> float:
    """Approximate SE of Pearson correlation: (1 - r^2) / sqrt(n - 1)."""
    return (1 - expected_r ** 2) / np.sqrt(max(n_pairs - 1, 1))


def _corr_tolerance(expected_r: float, n_pairs: int, min_tol: float = 0.05, n_se: int = 4) -> float:
    """Compute SE-based tolerance for correlation checks."""
    se = _corr_se(expected_r, n_pairs)
    return max(n_se * se, min_tol)


def validate_half_sibs(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Validate half-sibling counts and proportions related to p_nonsocial_father."""
    results = {}
    p_nonsocial = params.get("p_nonsocial_father", 0)
    fam_size = params.get("fam_size", 2)

    expected_half_sib_prop = 1 - (1 - p_nonsocial) ** 2
    expected_frac_with_half_sib = 1 - (1 - p_nonsocial) * np.exp(
        -fam_size * p_nonsocial
    )

    non_founders = df[df["mother"] != -1]
    non_twin_sibs = non_founders[non_founders["twin"] == -1][["id", "mother", "father"]]

    if len(non_twin_sibs) < 2:
        results["half_sib_pair_proportion"] = _result(
            True, "Not enough non-twin siblings to check"
        )
        results["offspring_with_half_sib"] = _result(
            True, "Not enough non-twin siblings to check"
        )
        return results

    sib_info = _count_sib_pairs(non_twin_sibs)

    # Maternal half-sib pair proportion (validates p_nonsocial_father)
    total_maternal_pairs = sib_info["n_full_sib_pairs"] + sib_info["n_maternal_half_sib_pairs"]
    if total_maternal_pairs > 0:
        observed_half_sib_prop = sib_info["n_maternal_half_sib_pairs"] / total_maternal_pairs
        se_prop = np.sqrt(
            expected_half_sib_prop * (1 - expected_half_sib_prop)
            / max(total_maternal_pairs, 1)
        )
        tol = max(4 * se_prop, 0.02)
        half_sib_ok = abs(observed_half_sib_prop - expected_half_sib_prop) < tol
        results["half_sib_pair_proportion"] = _result(
            half_sib_ok,
            f"Maternal half-sib pair proportion: {observed_half_sib_prop:.4f} "
            f"(expected: {expected_half_sib_prop:.4f}, tol: {tol:.4f})",
            expected=float(expected_half_sib_prop),
            observed=float(observed_half_sib_prop),
            n_full_sib_pairs=int(sib_info["n_full_sib_pairs"]),
            n_maternal_half_sib_pairs=int(sib_info["n_maternal_half_sib_pairs"]),
            n_paternal_half_sib_pairs=int(sib_info["n_paternal_half_sib_pairs"]),
        )
    else:
        results["half_sib_pair_proportion"] = _result(
            True, "No maternal sibling pairs to check"
        )

    # Offspring with maternal half-sib
    n_offspring_with_sibs = sib_info["n_offspring_with_sibs"]
    n_offspring_with_hs = sib_info["n_offspring_with_maternal_half_sib"]
    if n_offspring_with_sibs > 0:
        observed_frac = n_offspring_with_hs / n_offspring_with_sibs
        se_frac = np.sqrt(
            expected_frac_with_half_sib * (1 - expected_frac_with_half_sib)
            / max(n_offspring_with_sibs, 1)
        )
        tol = max(4 * se_frac, 0.02)
        frac_ok = abs(observed_frac - expected_frac_with_half_sib) < tol
        results["offspring_with_half_sib"] = _result(
            frac_ok,
            f"Offspring with maternal half-sib: {observed_frac:.4f} "
            f"(expected: {expected_frac_with_half_sib:.4f}, tol: {tol:.4f})",
            expected=float(expected_frac_with_half_sib),
            observed=float(observed_frac),
            n_offspring_with_half_sib=int(n_offspring_with_hs),
            n_offspring_with_sibs=int(n_offspring_with_sibs),
        )
    else:
        results["offspring_with_half_sib"] = _result(
            True, "No non-twin offspring with siblings to check"
        )

    return results


def validate_statistical(df: pd.DataFrame, params: dict[str, Any], df_indexed: pd.DataFrame) -> dict[str, Any]:
    """Validate statistical properties of variance components for two traits."""
    results = {}

    rA_param = params.get("rA", 0)
    rC_param = params.get("rC", 0)

    founders = df[df["mother"] == -1]

    # Variance checks for both traits
    for t in [1, 2]:
        for comp in ["A", "C", "E"]:
            col = f"{comp}{t}"
            results[f"variance_{col}"] = _check_variance(
                founders, col, params[col]
            )

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
            results[f"c{t}_inheritance"] = _result(
                True, "No non-founders to check C inheritance"
            )

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
                results["e1_independence"] = _result(
                    True, "Not enough sibling pairs to check E independence"
                )
        else:
            results["e1_independence"] = _result(
                True, "Not enough sibling groups to check E independence"
            )
    else:
        results["e1_independence"] = _result(
            True, "No non-founders to check E independence"
        )

    return results


def validate_heritability(df: pd.DataFrame, params: dict[str, Any], df_indexed: pd.DataFrame) -> dict[str, Any]:
    """Validate heritability estimates for two-trait simulation."""
    results = {}
    A_params = {1: params["A1"], 2: params["A2"]}

    # Precompute arrays
    comp_vals = {}
    for comp in ["A", "C", "E"]:
        for t in [1, 2]:
            comp_vals[f"{comp}{t}"] = df_indexed[f"{comp}{t}"].values
    id_to_idx = pd.Series(np.arange(len(df_indexed)), index=df_indexed.index)

    # --- MZ twin correlations ---
    twins_df = df[df["twin"] != -1]
    twin_ids = twins_df["id"].values
    twin_partners = twins_df["twin"].values
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]

    mz_pheno_corr = {}
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

    # --- DZ sibling correlations via vectorized merge ---
    non_founders = df[df["mother"] != -1]
    non_twin_sibs = non_founders[non_founders["twin"] == -1][["id", "mother", "father"]].copy()
    non_twin_sibs["_row"] = id_to_idx.reindex(non_twin_sibs["id"]).values.astype(int)

    # Full-sib pairs: same mother AND same father
    sib_counts = non_twin_sibs.groupby("mother").size()
    multi_mothers = sib_counts[sib_counts >= 2].index
    mat_sib = non_twin_sibs[non_twin_sibs["mother"].isin(multi_mothers)]

    dz_pheno_corr = {}
    n_dz_pairs = 0
    if len(mat_sib) > 0:
        mat_pairs = mat_sib[["mother", "father", "_row"]].merge(
            mat_sib[["mother", "father", "_row"]],
            on="mother", suffixes=("_1", "_2"),
        )
        full_sib_pairs = mat_pairs[
            (mat_pairs["_row_1"] < mat_pairs["_row_2"])
            & (mat_pairs["father_1"] == mat_pairs["father_2"])
        ]

        # Subsample if too many pairs
        max_pairs = 5000
        if len(full_sib_pairs) > max_pairs:
            rng = np.random.default_rng(params.get("seed", 42))
            full_sib_pairs = full_sib_pairs.iloc[
                rng.choice(len(full_sib_pairs), max_pairs, replace=False)
            ]

        n_dz_pairs = len(full_sib_pairs)
        if n_dz_pairs >= 10:
            idx1 = full_sib_pairs["_row_1"].values
            idx2 = full_sib_pairs["_row_2"].values

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

    # --- Falconer's estimates ---
    for t in [1, 2]:
        mz_c = mz_pheno_corr.get(t)
        dz_c = dz_pheno_corr.get(t)
        if mz_c is not None and dz_c is not None and not (np.isnan(mz_c) or np.isnan(dz_c)):
            falconer = 2 * (mz_c - dz_c)
            n_mz = len(t1_arr)
            se_mz = _corr_se(mz_c, n_mz)
            se_dz = _corr_se(dz_c, n_dz_pairs)
            se_falconer = 2 * np.sqrt(se_mz ** 2 + se_dz ** 2)
            falconer_tol = max(4 * se_falconer, 0.05)
            results[f"falconer_estimate_trait{t}"] = _result(
                abs(falconer - A_params[t]) < falconer_tol,
                f"Falconer h²{chr(8320+t)} = 2(r_MZ - r_DZ) = {falconer:.4f} "
                f"(expected: ~{A_params[t]}, tol: {falconer_tol:.4f})",
                expected=A_params[t],
                observed=float(falconer),
            )
        else:
            results[f"falconer_estimate_trait{t}"] = _result(
                True,
                "Cannot compute Falconer estimate without both MZ and DZ correlations",
            )

    # --- Parent-offspring regression ---
    if len(non_founders) > 100:
        valid_offspring = non_founders[
            non_founders["mother"].isin(df_indexed.index)
            & non_founders["father"].isin(df_indexed.index)
        ]

        if len(valid_offspring) > 100:
            mother_idx = id_to_idx.reindex(valid_offspring["mother"]).values.astype(int)
            father_idx = id_to_idx.reindex(valid_offspring["father"]).values.astype(int)
            offspring_idx = id_to_idx.reindex(valid_offspring["id"]).values.astype(int)

            for t in [1, 2]:
                results[f"parent_offspring_A{t}_regression"] = _midparent_regression(
                    comp_vals[f"A{t}"], mother_idx, father_idx, offspring_idx,
                    f"A{t}",
                )
                P_vals = comp_vals[f"A{t}"] + comp_vals[f"C{t}"] + comp_vals[f"E{t}"]
                results[f"parent_offspring_liability{t}_regression"] = _midparent_regression(
                    P_vals, mother_idx, father_idx, offspring_idx,
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
            results[f"parent_offspring_A{t}_regression"] = {
                "details": "Not enough non-founders for regression"
            }
            results[f"parent_offspring_liability{t}_regression"] = {
                "details": "Not enough non-founders for regression"
            }

    return results


def compute_per_generation_stats(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Compute per-generation statistics for two traits."""
    N = params["N"]
    ngen = params["G_ped"]

    # Assign generation labels once via integer division
    gen_labels = df["id"].values // N

    results = {}
    for gen in range(1, ngen + 1):
        gen_mask = gen_labels == (gen - 1)
        gen_df = df[gen_mask]

        gen_stats = {"n": int(gen_mask.sum())}
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
    """Validate population-level properties."""
    results = {}
    N = params["N"]
    ngen = params["G_ped"]
    fam_size = params["fam_size"]

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
        fam_ok = abs(mean_fam - fam_size) < fam_size * 0.5
        results["family_size"] = _result(
            fam_ok,
            f"Mean family size: {mean_fam:.2f} (expected: ~{fam_size})",
            expected=fam_size,
            observed=float(mean_fam),
        )
    else:
        results["family_size"] = _result(
            True, "No non-founders to check family size"
        )

    return results


def compute_family_size_distribution(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Compute offspring count distributions per parent sex."""
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
            "n_parents": int(len(counts)),
        }

    return result


def run_validation(pedigree_path: str, params_path: str) -> dict[str, Any]:
    """Run all validation checks and return results."""
    logger.info("Validating pedigree: %s", pedigree_path)
    df = pd.read_parquet(pedigree_path)
    with open(params_path) as f:
        params = yaml.safe_load(f)

    df_indexed = df.set_index("id")

    results = {
        "structural": validate_structural(df, params),
        "twins": validate_twins(df, params, df_indexed),
        "half_sibs": validate_half_sibs(df, params),
        "statistical": validate_statistical(df, params, df_indexed),
        "heritability": validate_heritability(df, params, df_indexed),
        "population": validate_population(df, params),
        "per_generation": compute_per_generation_stats(df, params),
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
                        category, check_name, check_result.get("details", ""),
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
        checks_passed, checks_passed + checks_failed,
    )

    return results


def cli() -> None:
    """Command-line interface for running validation."""
    from sim_ace import setup_logging
    parser = argparse.ArgumentParser(description="Validate ACE simulation output")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG output")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING+ only")
    parser.add_argument("--pedigree", required=True, help="Pedigree parquet path")
    parser.add_argument("--params", required=True, help="Params YAML path")
    parser.add_argument("--output", required=True, help="Output validation YAML path")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=level)

    results = run_validation(args.pedigree, args.params)
    results = to_native(results)

    with open(args.output, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
