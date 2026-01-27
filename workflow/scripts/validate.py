"""
ACE Simulation Validation

Validates simulation outputs for structural integrity, statistical properties,
and heritability estimates.
"""

import numpy as np
import pandas as pd
import yaml
from scipy import stats


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


def validate_structural(df, params):
    """Validate structural integrity of the pedigree."""
    results = {}
    N = params["N"]
    ngen = params["ngen"]
    expected_total = N * ngen

    # ID integrity
    ids = df["id"].values
    expected_ids = np.arange(expected_total)
    ids_contiguous = np.array_equal(np.sort(ids), expected_ids)
    results["id_integrity"] = {
        "passed": ids_contiguous and len(df) == expected_total,
        "details": f"Expected {expected_total} contiguous IDs, found {len(df)} individuals",
        "expected_count": expected_total,
        "observed_count": len(df),
    }

    # Parent references: valid IDs or -1 for founders
    valid_ids = set(df["id"].values) | {-1}
    mothers_valid = df["mother"].isin(valid_ids).all()
    fathers_valid = df["father"].isin(valid_ids).all()
    no_self_parent = ((df["mother"] != df["id"]) & (df["father"] != df["id"])).all()
    results["parent_references"] = {
        "passed": bool(mothers_valid and fathers_valid and no_self_parent),
        "details": f"Mothers valid: {mothers_valid}, Fathers valid: {fathers_valid}, No self-parenting: {no_self_parent}",
    }

    # Sex-parent consistency (only for non-founders)
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        id_to_sex = df.set_index("id")["sex"]
        mother_sex = id_to_sex.reindex(non_founders["mother"]).values
        father_sex = id_to_sex.reindex(non_founders["father"]).values
        mothers_female = (mother_sex == 0).all()
        fathers_male = (father_sex == 1).all()
        results["sex_parent_consistency"] = {
            "passed": bool(mothers_female and fathers_male),
            "details": f"Mothers female: {mothers_female}, Fathers male: {fathers_male}",
        }
    else:
        results["sex_parent_consistency"] = {
            "passed": True,
            "details": "No non-founders to check",
        }

    # Sex distribution
    sex_ratio = df["sex"].mean()
    sex_balanced = 0.45 <= sex_ratio <= 0.55
    results["sex_distribution"] = {
        "passed": sex_balanced,
        "details": f"Male ratio: {sex_ratio:.3f} (expected ~0.5)",
        "observed_ratio": float(sex_ratio),
    }

    return results


def validate_twins(df, params, df_indexed):
    """Validate MZ twin properties."""
    results = {}
    p_mztwin = params["p_mztwin"]

    # Find twins using vectorized operations
    twins_df = df[df["twin"] != -1]
    n_twins = len(twins_df)

    if n_twins == 0:
        results["twin_bidirectional"] = {"passed": True, "details": "No twins found"}
        results["twin_same_parents"] = {"passed": True, "details": "No twins found"}
        results["twin_same_A"] = {"passed": True, "details": "No twins found"}
        results["twin_same_sex"] = {"passed": True, "details": "No twins found"}
        results["twin_rate"] = {
            "passed": p_mztwin < 0.01,
            "details": f"No twins found, expected rate: {p_mztwin}",
            "expected_rate": p_mztwin,
            "observed_rate": 0.0,
        }
        return results

    # Get unique twin pairs (each pair appears twice, take lower id first)
    twin_ids = twins_df["id"].values
    twin_partners = twins_df["twin"].values
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]
    n_pairs = len(t1_arr)

    # Bidirectional check - vectorized
    twin_col = df_indexed["twin"]
    reverse_check = twin_col.reindex(t2_arr).values
    bidirectional = np.all(reverse_check == t1_arr)
    results["twin_bidirectional"] = {
        "passed": bool(bidirectional),
        "details": f"All {n_twins} twin references are bidirectional: {bidirectional}",
    }

    # Same parents - vectorized
    t1_mother = df_indexed.loc[t1_arr, "mother"].values
    t2_mother = df_indexed.loc[t2_arr, "mother"].values
    t1_father = df_indexed.loc[t1_arr, "father"].values
    t2_father = df_indexed.loc[t2_arr, "father"].values
    same_parents = np.all((t1_mother == t2_mother) & (t1_father == t2_father))
    results["twin_same_parents"] = {
        "passed": bool(same_parents),
        "details": f"All {n_pairs} twin pairs share parents: {same_parents}",
    }

    # Same A value - vectorized
    t1_A = df_indexed.loc[t1_arr, "A"].values
    t2_A = df_indexed.loc[t2_arr, "A"].values
    same_A = np.allclose(t1_A, t2_A)
    results["twin_same_A"] = {
        "passed": bool(same_A),
        "details": f"All MZ twin pairs have identical A values: {same_A}",
    }

    # Same sex - vectorized
    t1_sex = df_indexed.loc[t1_arr, "sex"].values
    t2_sex = df_indexed.loc[t2_arr, "sex"].values
    same_sex = np.all(t1_sex == t2_sex)
    results["twin_same_sex"] = {
        "passed": bool(same_sex),
        "details": f"All MZ twin pairs have same sex: {same_sex}",
    }

    # Twin rate
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        observed_rate = n_pairs * 2 / len(non_founders)
        rate_ok = observed_rate <= p_mztwin * 3 + 0.01
        results["twin_rate"] = {
            "passed": rate_ok,
            "details": f"Twin rate in non-founders: {observed_rate:.4f} (param: {p_mztwin})",
            "expected_rate": p_mztwin,
            "observed_rate": float(observed_rate),
            "twin_pairs": n_pairs,
        }
    else:
        results["twin_rate"] = {
            "passed": True,
            "details": "No non-founders to check twin rate",
        }

    return results


def validate_half_sibs(df, params, df_indexed):
    """Validate half-sibling properties related to p_nonsocial_father."""
    results = {}
    p_nonsocial = params.get("p_nonsocial_father", 0)
    fam_size = params.get("fam_size", 2)

    # Expected proportions
    expected_half_sib_prop = 1 - (1 - p_nonsocial) ** 2
    expected_frac_with_half_sib = 1 - (1 - p_nonsocial) * np.exp(
        -fam_size * p_nonsocial
    )

    non_founders = df[df["mother"] != -1].copy()

    # Get non-twin siblings grouped by mother - do this once
    non_twin_mask = non_founders["twin"] == -1
    non_twin_sibs = non_founders[non_twin_mask][["id", "mother", "father"]].copy()

    # Count siblings per mother for non-twins
    sib_counts = non_twin_sibs.groupby("mother").size()
    mothers_with_multiple = sib_counts[sib_counts >= 2].index

    if len(mothers_with_multiple) == 0:
        results["half_sib_pair_proportion"] = {
            "passed": True,
            "details": "No maternal sibling pairs to check",
        }
        results["offspring_with_half_sib"] = {
            "passed": True,
            "details": "No non-twin offspring with siblings to check",
        }
        results["half_sib_A_correlation"] = {
            "passed": True,
            "details": "Not enough half-sib pairs to compute correlation",
        }
        results["half_sib_phenotype_correlation"] = {
            "details": "Not enough half-sib pairs to compute correlation"
        }
        results["half_sib_shared_C"] = {
            "passed": True,
            "details": "Not enough half-sib pairs to check shared C",
        }
        return results

    # Filter to only families with 2+ non-twin siblings
    multi_sib_families = non_twin_sibs[
        non_twin_sibs["mother"].isin(mothers_with_multiple)
    ]

    # Count full-sib and half-sib pairs using vectorized groupby
    # For each family, count pairs with same father vs different father
    n_half_sib_pairs = 0
    n_full_sib_pairs = 0
    n_offspring_with_half_sib = 0
    n_offspring_with_sibs = 0

    # Also collect some pairs for correlation analysis
    half_sib_pairs_sample = []
    full_sib_pairs_sample = []
    max_pairs = 5000

    for mother, group in multi_sib_families.groupby("mother"):
        ids = group["id"].values
        fathers = group["father"].values
        n = len(ids)
        n_offspring_with_sibs += n

        # Count pairs - use numpy broadcasting for speed
        father_matrix = fathers[:, None] == fathers[None, :]
        # Upper triangle only (excluding diagonal)
        n_same_father = (np.triu(father_matrix, k=1)).sum()
        n_pairs_total = n * (n - 1) // 2
        n_diff_father = n_pairs_total - n_same_father

        n_full_sib_pairs += n_same_father
        n_half_sib_pairs += n_diff_father

        # Check if any offspring has a half-sib (different father exists)
        unique_fathers = np.unique(fathers)
        if len(unique_fathers) > 1:
            n_offspring_with_half_sib += n

        # Collect sample pairs for correlation
        if (
            len(half_sib_pairs_sample) < max_pairs
            or len(full_sib_pairs_sample) < max_pairs
        ):
            for i in range(min(n, 5)):
                for j in range(i + 1, min(n, 5)):
                    if fathers[i] == fathers[j]:
                        if len(full_sib_pairs_sample) < max_pairs:
                            full_sib_pairs_sample.append((ids[i], ids[j]))
                    else:
                        if len(half_sib_pairs_sample) < max_pairs:
                            half_sib_pairs_sample.append((ids[i], ids[j]))

    total_pairs = n_full_sib_pairs + n_half_sib_pairs

    # Half-sib pair proportion
    if total_pairs > 0:
        observed_half_sib_prop = n_half_sib_pairs / total_pairs
        half_sib_ok = abs(observed_half_sib_prop - expected_half_sib_prop) < 0.05
        results["half_sib_pair_proportion"] = {
            "passed": half_sib_ok,
            "expected": float(expected_half_sib_prop),
            "observed": float(observed_half_sib_prop),
            "n_full_sib_pairs": int(n_full_sib_pairs),
            "n_half_sib_pairs": int(n_half_sib_pairs),
            "details": f"Half-sib pair proportion: {observed_half_sib_prop:.4f} (expected: {expected_half_sib_prop:.4f})",
        }
    else:
        results["half_sib_pair_proportion"] = {
            "passed": True,
            "details": "No maternal sibling pairs to check",
        }

    # Offspring with half-sib
    if n_offspring_with_sibs > 0:
        observed_frac = n_offspring_with_half_sib / n_offspring_with_sibs
        frac_ok = abs(observed_frac - expected_frac_with_half_sib) < 0.05
        results["offspring_with_half_sib"] = {
            "passed": frac_ok,
            "expected": float(expected_frac_with_half_sib),
            "observed": float(observed_frac),
            "n_offspring_with_half_sib": int(n_offspring_with_half_sib),
            "n_offspring_with_sibs": int(n_offspring_with_sibs),
            "details": f"Offspring with half-sib: {observed_frac:.4f} (expected: {expected_frac_with_half_sib:.4f})",
        }
    else:
        results["offspring_with_half_sib"] = {
            "passed": True,
            "details": "No non-twin offspring with siblings to check",
        }

    # Half-sib correlations - vectorized
    if len(half_sib_pairs_sample) >= 10:
        pairs = np.array(half_sib_pairs_sample)
        t1, t2 = pairs[:, 0], pairs[:, 1]

        # Get values using vectorized indexing
        A_vals = df_indexed["A"].values
        C_vals = df_indexed["C"].values
        E_vals = df_indexed["E"].values
        id_to_idx = pd.Series(np.arange(len(df_indexed)), index=df_indexed.index)

        idx1 = id_to_idx.reindex(t1).values.astype(int)
        idx2 = id_to_idx.reindex(t2).values.astype(int)

        hs_A1, hs_A2 = A_vals[idx1], A_vals[idx2]
        hs_C1, hs_C2 = C_vals[idx1], C_vals[idx2]
        hs_E1, hs_E2 = E_vals[idx1], E_vals[idx2]

        hs_A_corr = np.corrcoef(hs_A1, hs_A2)[0, 1]
        hs_A_ok = 0.1 <= hs_A_corr <= 0.4
        results["half_sib_A_correlation"] = {
            "passed": hs_A_ok,
            "expected": 0.25,
            "observed": float(hs_A_corr),
            "details": f"Half-sib A correlation: {hs_A_corr:.4f} (expected: ~0.25)",
            "n_pairs": len(half_sib_pairs_sample),
        }

        hs_P1 = hs_A1 + hs_C1 + hs_E1
        hs_P2 = hs_A2 + hs_C2 + hs_E2
        hs_pheno_corr = np.corrcoef(hs_P1, hs_P2)[0, 1]
        results["half_sib_phenotype_correlation"] = {
            "observed": float(hs_pheno_corr),
            "details": f"Half-sib phenotype correlation: {hs_pheno_corr:.4f}",
            "n_pairs": len(half_sib_pairs_sample),
        }

        c_shared_prop = np.mean(np.isclose(hs_C1, hs_C2))
        c_shared_ok = c_shared_prop > 0.99
        results["half_sib_shared_C"] = {
            "passed": c_shared_ok,
            "expected": 1.0,
            "observed": float(c_shared_prop),
            "details": f"Half-sibs sharing C: {c_shared_prop:.4f} (expected: 1.0)",
            "n_pairs": len(half_sib_pairs_sample),
        }
    else:
        results["half_sib_A_correlation"] = {
            "passed": True,
            "details": f"Not enough half-sib pairs ({len(half_sib_pairs_sample)}) to compute correlation",
        }
        results["half_sib_phenotype_correlation"] = {
            "details": f"Not enough half-sib pairs ({len(half_sib_pairs_sample)}) to compute correlation"
        }
        results["half_sib_shared_C"] = {
            "passed": True,
            "details": f"Not enough half-sib pairs ({len(half_sib_pairs_sample)}) to check shared C",
        }

    return results


def validate_statistical(df, params, df_indexed):
    """Validate statistical properties of variance components."""
    results = {}
    A_param = params["A"]
    C_param = params["C"]
    E_param = params["E"]

    # Use only founders (first generation) for initial variance estimates
    founders = df[df["mother"] == -1]

    # Variance of A
    var_A = founders["A"].var()
    var_A_ok = abs(var_A - A_param) < 0.1
    results["variance_A"] = {
        "passed": var_A_ok,
        "expected": A_param,
        "observed": float(var_A),
        "details": f"Var(A) in founders: {var_A:.4f} (expected: {A_param})",
    }

    # Variance of C
    var_C = founders["C"].var()
    var_C_ok = abs(var_C - C_param) < 0.1
    results["variance_C"] = {
        "passed": var_C_ok,
        "expected": C_param,
        "observed": float(var_C),
        "details": f"Var(C) in founders: {var_C:.4f} (expected: {C_param})",
    }

    # Variance of E
    var_E = founders["E"].var()
    var_E_ok = abs(var_E - E_param) < 0.1
    results["variance_E"] = {
        "passed": var_E_ok,
        "expected": E_param,
        "observed": float(var_E),
        "details": f"Var(E) in founders: {var_E:.4f} (expected: {E_param})",
    }

    # Total variance
    total_var = var_A + var_C + var_E
    total_ok = abs(total_var - 1.0) < 0.15
    results["total_variance"] = {
        "passed": total_ok,
        "expected": 1.0,
        "observed": float(total_var),
        "details": f"Total variance in founders: {total_var:.4f} (expected: 1.0)",
    }

    # C inheritance: siblings should share C (same mother)
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        c_by_mother = non_founders.groupby("mother")["C"].nunique()
        c_shared = (c_by_mother == 1).mean()
        c_ok = c_shared > 0.99
        results["c_inheritance"] = {
            "passed": c_ok,
            "details": f"Proportion of families with shared C: {c_shared:.4f}",
            "proportion_shared": float(c_shared),
        }
    else:
        results["c_inheritance"] = {
            "passed": True,
            "details": "No non-founders to check C inheritance",
        }

    # E independence - sample-based for efficiency
    if len(non_founders) > 0:
        # Get families with 2+ members
        fam_sizes = non_founders.groupby("mother").size()
        multi_child_mothers = fam_sizes[fam_sizes >= 2].index

        if len(multi_child_mothers) > 10:
            # Sample up to 1000 pairs
            e_pairs = []
            for mother in multi_child_mothers[:500]:
                group = non_founders[non_founders["mother"] == mother]
                e_vals = group["E"].values
                if len(e_vals) >= 2:
                    e_pairs.append((e_vals[0], e_vals[1]))
                if len(e_pairs) >= 1000:
                    break

            if len(e_pairs) > 10:
                e1, e2 = zip(*e_pairs)
                e_corr = np.corrcoef(e1, e2)[0, 1]
                e_indep = abs(e_corr) < 0.1
                results["e_independence"] = {
                    "passed": e_indep,
                    "details": f"E correlation between siblings: {e_corr:.4f} (expected: ~0)",
                    "observed_correlation": float(e_corr),
                }
            else:
                results["e_independence"] = {
                    "passed": True,
                    "details": "Not enough sibling pairs to check E independence",
                }
        else:
            results["e_independence"] = {
                "passed": True,
                "details": "Not enough sibling groups to check E independence",
            }
    else:
        results["e_independence"] = {
            "passed": True,
            "details": "No non-founders to check E independence",
        }

    return results


def validate_heritability(df, params, df_indexed):
    """Validate heritability estimates."""
    results = {}
    A_param = params["A"]

    # Get twin pairs - vectorized
    twins_df = df[df["twin"] != -1]
    twin_ids = twins_df["id"].values
    twin_partners = twins_df["twin"].values
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]

    # Precompute arrays for fast indexing
    A_vals = df_indexed["A"].values
    C_vals = df_indexed["C"].values
    E_vals = df_indexed["E"].values
    id_to_idx = pd.Series(np.arange(len(df_indexed)), index=df_indexed.index)

    # MZ twin correlations
    if len(t1_arr) >= 10:
        idx1 = id_to_idx.reindex(t1_arr).values.astype(int)
        idx2 = id_to_idx.reindex(t2_arr).values.astype(int)

        mz_A1, mz_A2 = A_vals[idx1], A_vals[idx2]
        mz_corr = np.corrcoef(mz_A1, mz_A2)[0, 1]
        mz_ok = mz_corr > 0.99
        results["mz_twin_A_correlation"] = {
            "passed": mz_ok,
            "expected": 1.0,
            "observed": float(mz_corr),
            "details": f"MZ twin A correlation: {mz_corr:.4f} (expected: 1.0)",
            "n_pairs": len(t1_arr),
        }

        mz_P1 = mz_A1 + C_vals[idx1] + E_vals[idx1]
        mz_P2 = mz_A2 + C_vals[idx2] + E_vals[idx2]
        mz_pheno_corr = np.corrcoef(mz_P1, mz_P2)[0, 1]
        results["mz_twin_phenotype_correlation"] = {
            "observed": float(mz_pheno_corr),
            "details": f"MZ twin phenotype correlation: {mz_pheno_corr:.4f}",
            "n_pairs": len(t1_arr),
        }
    else:
        results["mz_twin_A_correlation"] = {
            "passed": True,
            "details": f"Not enough MZ twin pairs ({len(t1_arr)}) to compute correlation",
        }
        mz_pheno_corr = None

    # DZ sibling correlations - find full sibling pairs (same parents, not twins)
    non_founders = df[df["mother"] != -1]
    non_twin_sibs = non_founders[non_founders["twin"] == -1]

    # Group by parent pair
    parent_keys = (
        non_twin_sibs["mother"].astype(str) + "_" + non_twin_sibs["father"].astype(str)
    )
    non_twin_sibs = non_twin_sibs.copy()
    non_twin_sibs["parent_key"] = parent_keys

    # Sample DZ pairs
    dz_pairs = []
    for key, group in non_twin_sibs.groupby("parent_key"):
        if len(group) >= 2:
            ids = group["id"].values
            for i in range(min(len(ids), 3)):
                for j in range(i + 1, min(len(ids), 3)):
                    dz_pairs.append((ids[i], ids[j]))
        if len(dz_pairs) >= 5000:
            break

    if len(dz_pairs) >= 10:
        pairs = np.array(dz_pairs)
        idx1 = id_to_idx.reindex(pairs[:, 0]).values.astype(int)
        idx2 = id_to_idx.reindex(pairs[:, 1]).values.astype(int)

        dz_A1, dz_A2 = A_vals[idx1], A_vals[idx2]
        dz_corr = np.corrcoef(dz_A1, dz_A2)[0, 1]
        dz_ok = 0.3 <= dz_corr <= 0.7
        results["dz_sibling_A_correlation"] = {
            "passed": dz_ok,
            "expected": 0.5,
            "observed": float(dz_corr),
            "details": f"DZ sibling A correlation: {dz_corr:.4f} (expected: ~0.5)",
            "n_pairs": len(dz_pairs),
        }

        dz_P1 = dz_A1 + C_vals[idx1] + E_vals[idx1]
        dz_P2 = dz_A2 + C_vals[idx2] + E_vals[idx2]
        dz_pheno_corr = np.corrcoef(dz_P1, dz_P2)[0, 1]
        results["dz_sibling_phenotype_correlation"] = {
            "observed": float(dz_pheno_corr),
            "details": f"DZ sibling phenotype correlation: {dz_pheno_corr:.4f}",
            "n_pairs": len(dz_pairs),
        }
    else:
        results["dz_sibling_A_correlation"] = {
            "passed": True,
            "details": f"Not enough DZ sibling pairs ({len(dz_pairs)}) to compute correlation",
        }
        dz_pheno_corr = None

    # Falconer's estimate
    if mz_pheno_corr is not None and dz_pheno_corr is not None:
        falconer = 2 * (mz_pheno_corr - dz_pheno_corr)
        falconer_ok = abs(falconer - A_param) < 0.2
        results["falconer_estimate"] = {
            "passed": falconer_ok,
            "expected": A_param,
            "observed": float(falconer),
            "details": f"Falconer h² = 2(r_MZ - r_DZ) = {falconer:.4f} (expected: ~{A_param})",
        }
    else:
        results["falconer_estimate"] = {
            "passed": True,
            "details": "Cannot compute Falconer estimate without both MZ and DZ correlations",
        }

    # Parent-offspring regression - vectorized
    if len(non_founders) > 100:
        # Filter to offspring with both parents in data
        valid_offspring = non_founders[
            non_founders["mother"].isin(df_indexed.index)
            & non_founders["father"].isin(df_indexed.index)
        ]

        if len(valid_offspring) > 100:
            mother_idx = id_to_idx.reindex(valid_offspring["mother"]).values.astype(int)
            father_idx = id_to_idx.reindex(valid_offspring["father"]).values.astype(int)
            offspring_idx = id_to_idx.reindex(valid_offspring["id"]).values.astype(int)

            midparent_A = (A_vals[mother_idx] + A_vals[father_idx]) / 2
            offspring_A = A_vals[offspring_idx]

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                midparent_A, offspring_A
            )
            results["parent_offspring_A_regression"] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "details": f"Midparent-offspring A regression: slope={slope:.4f}, R²={r_value**2:.4f}",
            }

            # Phenotype regression
            P_vals = A_vals + C_vals + E_vals
            midparent_P = (P_vals[mother_idx] + P_vals[father_idx]) / 2
            offspring_P = P_vals[offspring_idx]

            slope_P, intercept_P, r_value_P, p_value_P, std_err_P = stats.linregress(
                midparent_P, offspring_P
            )
            results["parent_offspring_phenotype_regression"] = {
                "slope": float(slope_P),
                "intercept": float(intercept_P),
                "r_squared": float(r_value_P**2),
                "details": f"Midparent-offspring phenotype regression: slope={slope_P:.4f}, R²={r_value_P**2:.4f}",
            }
        else:
            results["parent_offspring_A_regression"] = {
                "details": "Not enough offspring with both parents in data"
            }
            results["parent_offspring_phenotype_regression"] = {
                "details": "Not enough offspring with both parents in data"
            }
    else:
        results["parent_offspring_A_regression"] = {
            "details": "Not enough non-founders for regression"
        }
        results["parent_offspring_phenotype_regression"] = {
            "details": "Not enough non-founders for regression"
        }

    return results


def compute_per_generation_stats(df, params):
    """Compute per-generation statistics."""
    N = params["N"]
    ngen = params["ngen"]

    results = {}
    for gen in range(1, ngen + 1):
        start_id = (gen - 1) * N
        end_id = gen * N
        gen_df = df[(df["id"] >= start_id) & (df["id"] < end_id)]

        phenotype = gen_df["A"] + gen_df["C"] + gen_df["E"]

        results[f"generation_{gen}"] = {
            "n": len(gen_df),
            "phenotype_mean": float(phenotype.mean()),
            "phenotype_variance": float(phenotype.var()),
            "phenotype_sd": float(phenotype.std()),
            "A_mean": float(gen_df["A"].mean()),
            "A_var": float(gen_df["A"].var()),
            "C_mean": float(gen_df["C"].mean()),
            "C_var": float(gen_df["C"].var()),
            "E_mean": float(gen_df["E"].mean()),
            "E_var": float(gen_df["E"].var()),
        }

    return results


def validate_population(df, params):
    """Validate population-level properties."""
    results = {}
    N = params["N"]
    ngen = params["ngen"]
    fam_size = params["fam_size"]

    # Generation sizes - vectorized
    gen_assignments = df["id"] // N
    gen_sizes = [int((gen_assignments == i).sum()) for i in range(ngen)]

    all_correct = all(s == N for s in gen_sizes)
    results["generation_sizes"] = {
        "passed": all_correct,
        "expected": N,
        "observed": gen_sizes,
        "details": f"Generation sizes: {gen_sizes} (expected: {N} each)",
    }

    results["generation_count"] = {
        "passed": len(gen_sizes) == ngen,
        "expected": ngen,
        "observed": len(gen_sizes),
        "details": f"Number of generations: {len(gen_sizes)} (expected: {ngen})",
    }

    # Mean family size
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        family_sizes = non_founders.groupby("mother").size()
        mean_fam = family_sizes.mean()
        fam_ok = abs(mean_fam - fam_size) < fam_size * 0.5
        results["family_size"] = {
            "passed": fam_ok,
            "expected": fam_size,
            "observed": float(mean_fam),
            "details": f"Mean family size: {mean_fam:.2f} (expected: ~{fam_size})",
        }
    else:
        results["family_size"] = {
            "passed": True,
            "details": "No non-founders to check family size",
        }

    return results


def run_validation(pedigree_path, params_path):
    """Run all validation checks and return results."""
    # Load data
    df = pd.read_parquet(pedigree_path)
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Create indexed dataframe once for all functions
    df_indexed = df.set_index("id")

    # Run all validation categories
    results = {
        "structural": validate_structural(df, params),
        "twins": validate_twins(df, params, df_indexed),
        "half_sibs": validate_half_sibs(df, params, df_indexed),
        "statistical": validate_statistical(df, params, df_indexed),
        "heritability": validate_heritability(df, params, df_indexed),
        "population": validate_population(df, params),
        "per_generation": compute_per_generation_stats(df, params),
    }

    # Compute summary
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
