"""
ACE Simulation Validation

Validates simulation outputs for structural integrity, statistical properties,
and heritability estimates.
"""

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def safe_corrcoef(x, y):
    """Compute Pearson correlation, returning nan if either array has zero variance."""
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return np.corrcoef(x, y)[0, 1]


def safe_linregress(x, y):
    """Run linear regression, returning None if x has zero variance."""
    if np.std(x) == 0:
        return None
    return stats.linregress(x, y)


def to_native(obj):
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


def _result(passed, details, **extra):
    """Build a result dict with passed/details and optional extra keys."""
    d = {"passed": passed, "details": details}
    d.update(extra)
    return d


def _check_variance(founders, col, expected, tol=0.1):
    """Check that the variance of `col` in founders is close to `expected`."""
    var = founders[col].var()
    return _result(
        abs(var - expected) < tol,
        f"Var({col}) in founders: {var:.4f} (expected: {expected})",
        expected=expected,
        observed=float(var),
    )


def _check_pair_correlation(vals1, vals2, expected, label, tol, n_pairs,
                            check_fn=None):
    """Compute correlation between paired arrays and check against expected.

    check_fn: optional callable(corr) -> bool. If None, uses abs(corr - expected) < tol.
    """
    corr = safe_corrcoef(vals1, vals2)
    if check_fn is not None:
        ok = check_fn(corr)
    elif np.isnan(corr):
        ok = expected == 0
    else:
        ok = abs(corr - expected) < tol
    return _result(
        ok,
        f"{label}: {corr:.4f} (expected: {expected})",
        expected=expected,
        observed=float(corr),
        n_pairs=n_pairs,
    )


def _midparent_regression(vals, mother_idx, father_idx, offspring_idx, label):
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


def _sample_sib_pairs(multi_sib_families, max_pairs=5000):
    """Sample full-sib and half-sib pairs from families with 2+ non-twin siblings.

    Returns (half_sib_pairs, full_sib_pairs, n_half_sib_pairs, n_full_sib_pairs,
             n_offspring_with_half_sib, n_offspring_with_sibs).
    """
    n_half_sib_pairs = 0
    n_full_sib_pairs = 0
    n_offspring_with_half_sib = 0
    n_offspring_with_sibs = 0
    half_sib_pairs = []
    full_sib_pairs = []

    for mother, group in multi_sib_families.groupby("mother"):
        ids = group["id"].values
        fathers = group["father"].values
        n = len(ids)
        n_offspring_with_sibs += n

        father_matrix = fathers[:, None] == fathers[None, :]
        n_same_father = (np.triu(father_matrix, k=1)).sum()
        n_pairs_total = n * (n - 1) // 2
        n_diff_father = n_pairs_total - n_same_father

        n_full_sib_pairs += n_same_father
        n_half_sib_pairs += n_diff_father

        unique_fathers = np.unique(fathers)
        if len(unique_fathers) > 1:
            n_offspring_with_half_sib += n

        if len(half_sib_pairs) < max_pairs or len(full_sib_pairs) < max_pairs:
            for i in range(min(n, 5)):
                for j in range(i + 1, min(n, 5)):
                    if fathers[i] == fathers[j]:
                        if len(full_sib_pairs) < max_pairs:
                            full_sib_pairs.append((ids[i], ids[j]))
                    else:
                        if len(half_sib_pairs) < max_pairs:
                            half_sib_pairs.append((ids[i], ids[j]))

    return (half_sib_pairs, full_sib_pairs, n_half_sib_pairs, n_full_sib_pairs,
            n_offspring_with_half_sib, n_offspring_with_sibs)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_structural(df, params):
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


def validate_twins(df, params, df_indexed):
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
        rate_ok = observed_rate <= p_mztwin * 3 + 0.01
        results["twin_rate"] = _result(
            rate_ok,
            f"Twin rate in non-founders: {observed_rate:.4f} (param: {p_mztwin})",
            expected_rate=p_mztwin,
            observed_rate=float(observed_rate),
            twin_pairs=n_pairs,
        )
    else:
        results["twin_rate"] = _result(True, "No non-founders to check twin rate")

    return results


def validate_half_sibs(df, params, df_indexed):
    """Validate half-sibling properties related to p_nonsocial_father."""
    results = {}
    p_nonsocial = params.get("p_nonsocial_father", 0)
    fam_size = params.get("fam_size", 2)

    expected_half_sib_prop = 1 - (1 - p_nonsocial) ** 2
    expected_frac_with_half_sib = 1 - (1 - p_nonsocial) * np.exp(
        -fam_size * p_nonsocial
    )

    non_founders = df[df["mother"] != -1].copy()
    non_twin_mask = non_founders["twin"] == -1
    non_twin_sibs = non_founders[non_twin_mask][["id", "mother", "father"]].copy()

    sib_counts = non_twin_sibs.groupby("mother").size()
    mothers_with_multiple = sib_counts[sib_counts >= 2].index

    if len(mothers_with_multiple) == 0:
        results["half_sib_pair_proportion"] = _result(
            True, "No maternal sibling pairs to check"
        )
        results["offspring_with_half_sib"] = _result(
            True, "No non-twin offspring with siblings to check"
        )
        results["half_sib_A1_correlation"] = _result(
            True, "Not enough half-sib pairs to compute correlation"
        )
        results["half_sib_liability1_correlation"] = {
            "details": "Not enough half-sib pairs to compute correlation"
        }
        results["half_sib_shared_C1"] = _result(
            True, "Not enough half-sib pairs to check shared C"
        )
        return results

    multi_sib_families = non_twin_sibs[
        non_twin_sibs["mother"].isin(mothers_with_multiple)
    ]

    (half_sib_pairs_sample, full_sib_pairs_sample,
     n_half_sib_pairs, n_full_sib_pairs,
     n_offspring_with_half_sib, n_offspring_with_sibs) = _sample_sib_pairs(
        multi_sib_families
    )

    total_pairs = n_full_sib_pairs + n_half_sib_pairs

    # Half-sib pair proportion
    if total_pairs > 0:
        observed_half_sib_prop = n_half_sib_pairs / total_pairs
        half_sib_ok = abs(observed_half_sib_prop - expected_half_sib_prop) < 0.05
        results["half_sib_pair_proportion"] = _result(
            half_sib_ok,
            f"Half-sib pair proportion: {observed_half_sib_prop:.4f} (expected: {expected_half_sib_prop:.4f})",
            expected=float(expected_half_sib_prop),
            observed=float(observed_half_sib_prop),
            n_full_sib_pairs=int(n_full_sib_pairs),
            n_half_sib_pairs=int(n_half_sib_pairs),
        )
    else:
        results["half_sib_pair_proportion"] = _result(
            True, "No maternal sibling pairs to check"
        )

    # Offspring with half-sib
    if n_offspring_with_sibs > 0:
        observed_frac = n_offspring_with_half_sib / n_offspring_with_sibs
        frac_ok = abs(observed_frac - expected_frac_with_half_sib) < 0.05
        results["offspring_with_half_sib"] = _result(
            frac_ok,
            f"Offspring with half-sib: {observed_frac:.4f} (expected: {expected_frac_with_half_sib:.4f})",
            expected=float(expected_frac_with_half_sib),
            observed=float(observed_frac),
            n_offspring_with_half_sib=int(n_offspring_with_half_sib),
            n_offspring_with_sibs=int(n_offspring_with_sibs),
        )
    else:
        results["offspring_with_half_sib"] = _result(
            True, "No non-twin offspring with siblings to check"
        )

    # Half-sib correlations (using trait 1)
    if len(half_sib_pairs_sample) >= 10:
        pairs = np.array(half_sib_pairs_sample)
        t1, t2 = pairs[:, 0], pairs[:, 1]

        A1_vals = df_indexed["A1"].values
        C1_vals = df_indexed["C1"].values
        E1_vals = df_indexed["E1"].values
        id_to_idx = pd.Series(np.arange(len(df_indexed)), index=df_indexed.index)

        idx1 = id_to_idx.reindex(t1).values.astype(int)
        idx2 = id_to_idx.reindex(t2).values.astype(int)

        hs_A1_1, hs_A1_2 = A1_vals[idx1], A1_vals[idx2]

        hs_A1_corr = safe_corrcoef(hs_A1_1, hs_A1_2)
        hs_A1_ok = (0.1 <= hs_A1_corr <= 0.4) if not np.isnan(hs_A1_corr) else params.get("A1", 0) == 0
        results["half_sib_A1_correlation"] = _result(
            hs_A1_ok,
            f"Half-sib A1 correlation: {hs_A1_corr:.4f} (expected: ~0.25)",
            expected=0.25,
            observed=float(hs_A1_corr),
            n_pairs=len(half_sib_pairs_sample),
        )

        hs_C1_1, hs_C1_2 = C1_vals[idx1], C1_vals[idx2]
        hs_E1_1, hs_E1_2 = E1_vals[idx1], E1_vals[idx2]
        hs_P1 = hs_A1_1 + hs_C1_1 + hs_E1_1
        hs_P2 = hs_A1_2 + hs_C1_2 + hs_E1_2
        hs_pheno_corr = safe_corrcoef(hs_P1, hs_P2)
        results["half_sib_liability1_correlation"] = {
            "observed": float(hs_pheno_corr),
            "details": f"Half-sib liability1 correlation: {hs_pheno_corr:.4f}",
            "n_pairs": len(half_sib_pairs_sample),
        }

        c_shared_prop = np.mean(np.isclose(hs_C1_1, hs_C1_2))
        results["half_sib_shared_C1"] = _result(
            c_shared_prop > 0.99,
            f"Half-sibs sharing C1: {c_shared_prop:.4f} (expected: 1.0)",
            expected=1.0,
            observed=float(c_shared_prop),
            n_pairs=len(half_sib_pairs_sample),
        )
    else:
        n = len(half_sib_pairs_sample)
        results["half_sib_A1_correlation"] = _result(
            True, f"Not enough half-sib pairs ({n}) to compute correlation"
        )
        results["half_sib_liability1_correlation"] = {
            "details": f"Not enough half-sib pairs ({n}) to compute correlation"
        }
        results["half_sib_shared_C1"] = _result(
            True, f"Not enough half-sib pairs ({n}) to check shared C"
        )

    return results


def validate_statistical(df, params, df_indexed):
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
            e1_pairs = []
            for mother in multi_child_mothers[:500]:
                group = non_founders[non_founders["mother"] == mother]
                e_vals = group["E1"].values
                if len(e_vals) >= 2:
                    e1_pairs.append((e_vals[0], e_vals[1]))
                if len(e1_pairs) >= 1000:
                    break

            if len(e1_pairs) > 10:
                e1, e2 = zip(*e1_pairs)
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


def validate_heritability(df, params, df_indexed):
    """Validate heritability estimates for two-trait simulation."""
    results = {}
    A_params = {1: params["A1"], 2: params["A2"]}

    # Get twin pairs
    twins_df = df[df["twin"] != -1]
    twin_ids = twins_df["id"].values
    twin_partners = twins_df["twin"].values
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]

    # Precompute arrays
    comp_vals = {}
    for comp in ["A", "C", "E"]:
        for t in [1, 2]:
            comp_vals[f"{comp}{t}"] = df_indexed[f"{comp}{t}"].values
    id_to_idx = pd.Series(np.arange(len(df_indexed)), index=df_indexed.index)

    # MZ twin correlations
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

    # DZ sibling correlations
    non_founders = df[df["mother"] != -1]
    non_twin_sibs = non_founders[non_founders["twin"] == -1].copy()
    parent_keys = (
        non_twin_sibs["mother"].astype(str) + "_" + non_twin_sibs["father"].astype(str)
    )
    non_twin_sibs["parent_key"] = parent_keys

    dz_pairs = []
    for key, group in non_twin_sibs.groupby("parent_key"):
        if len(group) >= 2:
            ids = group["id"].values
            for i in range(min(len(ids), 3)):
                for j in range(i + 1, min(len(ids), 3)):
                    dz_pairs.append((ids[i], ids[j]))
        if len(dz_pairs) >= 5000:
            break

    dz_pheno_corr = {}
    if len(dz_pairs) >= 10:
        pairs = np.array(dz_pairs)
        idx1 = id_to_idx.reindex(pairs[:, 0]).values.astype(int)
        idx2 = id_to_idx.reindex(pairs[:, 1]).values.astype(int)

        for t in [1, 2]:
            col = f"A{t}"
            dz_v1, dz_v2 = comp_vals[col][idx1], comp_vals[col][idx2]
            dz_corr = safe_corrcoef(dz_v1, dz_v2)
            dz_ok = (0.3 <= dz_corr <= 0.7) if not np.isnan(dz_corr) else A_params[t] == 0
            results[f"dz_sibling_{col}_correlation"] = _result(
                dz_ok,
                f"DZ sibling {col} correlation: {dz_corr:.4f} (expected: ~0.5)",
                expected=0.5,
                observed=float(dz_corr),
                n_pairs=len(dz_pairs),
            )

            P1 = dz_v1 + comp_vals[f"C{t}"][idx1] + comp_vals[f"E{t}"][idx1]
            P2 = dz_v2 + comp_vals[f"C{t}"][idx2] + comp_vals[f"E{t}"][idx2]
            pheno_corr = safe_corrcoef(P1, P2)
            dz_pheno_corr[t] = pheno_corr
            results[f"dz_sibling_liability{t}_correlation"] = {
                "observed": float(pheno_corr),
                "details": f"DZ sibling liability{t} correlation: {pheno_corr:.4f}",
                "n_pairs": len(dz_pairs),
            }
    else:
        for t in [1, 2]:
            results[f"dz_sibling_A{t}_correlation"] = _result(
                True,
                f"Not enough DZ sibling pairs ({len(dz_pairs)}) to compute correlation",
            )
            dz_pheno_corr[t] = None

    # Falconer's estimates
    for t in [1, 2]:
        mz_c = mz_pheno_corr.get(t)
        dz_c = dz_pheno_corr.get(t)
        if mz_c is not None and dz_c is not None and not (np.isnan(mz_c) or np.isnan(dz_c)):
            falconer = 2 * (mz_c - dz_c)
            results[f"falconer_estimate_trait{t}"] = _result(
                abs(falconer - A_params[t]) < 0.2,
                f"Falconer h²{chr(8320+t)} = 2(r_MZ - r_DZ) = {falconer:.4f} (expected: ~{A_params[t]})",
                expected=A_params[t],
                observed=float(falconer),
            )
        else:
            results[f"falconer_estimate_trait{t}"] = _result(
                True,
                "Cannot compute Falconer estimate without both MZ and DZ correlations",
            )

    # Parent-offspring regression
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
                # A regression
                results[f"parent_offspring_A{t}_regression"] = _midparent_regression(
                    comp_vals[f"A{t}"], mother_idx, father_idx, offspring_idx,
                    f"A{t}",
                )
                # Liability regression
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


def compute_per_generation_stats(df, params):
    """Compute per-generation statistics for two traits."""
    N = params["N"]
    ngen = params["G_ped"]

    results = {}
    for gen in range(1, ngen + 1):
        start_id = (gen - 1) * N
        end_id = gen * N
        gen_df = df[(df["id"] >= start_id) & (df["id"] < end_id)]

        gen_stats = {"n": len(gen_df)}
        for t in [1, 2]:
            liability = gen_df[f"A{t}"] + gen_df[f"C{t}"] + gen_df[f"E{t}"]
            gen_stats[f"liability{t}_mean"] = float(liability.mean())
            gen_stats[f"liability{t}_variance"] = float(liability.var())
            gen_stats[f"liability{t}_sd"] = float(liability.std())
            for comp in ["A", "C", "E"]:
                col = f"{comp}{t}"
                gen_stats[f"{col}_mean"] = float(gen_df[col].mean())
                gen_stats[f"{col}_var"] = float(gen_df[col].var())

        results[f"generation_{gen}"] = gen_stats

    return results


def validate_population(df, params):
    """Validate population-level properties."""
    results = {}
    N = params["N"]
    ngen = params["G_ped"]
    fam_size = params["fam_size"]

    gen_assignments = df["id"] // N
    gen_sizes = [int((gen_assignments == i).sum()) for i in range(ngen)]

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


def run_validation(pedigree_path, params_path):
    """Run all validation checks and return results."""
    df = pd.read_parquet(pedigree_path)
    with open(params_path) as f:
        params = yaml.safe_load(f)

    df_indexed = df.set_index("id")

    results = {
        "structural": validate_structural(df, params),
        "twins": validate_twins(df, params, df_indexed),
        "half_sibs": validate_half_sibs(df, params, df_indexed),
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

    results["summary"] = {
        "passed": checks_failed == 0,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "checks_total": checks_passed + checks_failed,
    }

    results["parameters"] = params

    return results


if __name__ == "__main__":
    pedigree_path = snakemake.input.pedigree
    params_path = snakemake.input.params
    output_path = snakemake.output.report

    results = run_validation(pedigree_path, params_path)
    results = to_native(results)

    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
