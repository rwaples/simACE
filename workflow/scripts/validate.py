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
    N = params['N']
    ngen = params['ngen']
    expected_total = N * ngen

    # ID integrity
    ids = df['id'].values
    expected_ids = np.arange(expected_total)
    ids_contiguous = np.array_equal(np.sort(ids), expected_ids)
    results['id_integrity'] = {
        'passed': ids_contiguous and len(df) == expected_total,
        'details': f"Expected {expected_total} contiguous IDs, found {len(df)} individuals",
        'expected_count': expected_total,
        'observed_count': len(df)
    }

    # Parent references: valid IDs or -1 for founders
    valid_ids = set(df['id'].values) | {-1}
    mothers_valid = df['mother'].isin(valid_ids).all()
    fathers_valid = df['father'].isin(valid_ids).all()
    no_self_parent = ((df['mother'] != df['id']) & (df['father'] != df['id'])).all()
    results['parent_references'] = {
        'passed': bool(mothers_valid and fathers_valid and no_self_parent),
        'details': f"Mothers valid: {mothers_valid}, Fathers valid: {fathers_valid}, No self-parenting: {no_self_parent}"
    }

    # Sex-parent consistency (only for non-founders)
    non_founders = df[df['mother'] != -1].copy()
    if len(non_founders) > 0:
        mother_sex = df.set_index('id')['sex'].reindex(non_founders['mother']).values
        father_sex = df.set_index('id')['sex'].reindex(non_founders['father']).values
        mothers_female = (mother_sex == 0).all()
        fathers_male = (father_sex == 1).all()
        results['sex_parent_consistency'] = {
            'passed': bool(mothers_female and fathers_male),
            'details': f"Mothers female: {mothers_female}, Fathers male: {fathers_male}"
        }
    else:
        results['sex_parent_consistency'] = {
            'passed': True,
            'details': "No non-founders to check"
        }

    # Sex distribution
    sex_ratio = df['sex'].mean()
    sex_balanced = 0.45 <= sex_ratio <= 0.55
    results['sex_distribution'] = {
        'passed': sex_balanced,
        'details': f"Male ratio: {sex_ratio:.3f} (expected ~0.5)",
        'observed_ratio': float(sex_ratio)
    }

    return results


def validate_twins(df, params):
    """Validate MZ twin properties."""
    results = {}
    p_mztwin = params['p_mztwin']

    # Find twins
    twins_df = df[df['twin'] != -1].copy()
    n_twins = len(twins_df)

    if n_twins == 0:
        results['twin_bidirectional'] = {'passed': True, 'details': "No twins found"}
        results['twin_same_parents'] = {'passed': True, 'details': "No twins found"}
        results['twin_same_A'] = {'passed': True, 'details': "No twins found"}
        results['twin_same_sex'] = {'passed': True, 'details': "No twins found"}
        results['twin_rate'] = {
            'passed': p_mztwin < 0.01,
            'details': f"No twins found, expected rate: {p_mztwin}",
            'expected_rate': p_mztwin,
            'observed_rate': 0.0
        }
        return results

    # Bidirectional check
    id_to_twin = df.set_index('id')['twin'].to_dict()
    bidirectional = all(
        id_to_twin.get(id_to_twin.get(i, -1), -1) == i
        for i in twins_df['id'] if id_to_twin.get(i, -1) != -1
    )
    results['twin_bidirectional'] = {
        'passed': bidirectional,
        'details': f"All {n_twins} twin references are bidirectional: {bidirectional}"
    }

    # Same parents
    df_indexed = df.set_index('id')
    twin_pairs = []
    seen = set()
    for idx, row in twins_df.iterrows():
        if row['id'] not in seen and row['twin'] != -1:
            twin_pairs.append((row['id'], row['twin']))
            seen.add(row['id'])
            seen.add(row['twin'])

    same_parents = True
    for t1, t2 in twin_pairs:
        if t2 in df_indexed.index:
            t1_data = df_indexed.loc[t1]
            t2_data = df_indexed.loc[t2]
            if t1_data['mother'] != t2_data['mother'] or t1_data['father'] != t2_data['father']:
                same_parents = False
                break
    results['twin_same_parents'] = {
        'passed': same_parents,
        'details': f"All {len(twin_pairs)} twin pairs share parents: {same_parents}"
    }

    # Same A value (MZ twins)
    same_A = True
    for t1, t2 in twin_pairs:
        if t2 in df_indexed.index:
            if not np.isclose(df_indexed.loc[t1, 'A'], df_indexed.loc[t2, 'A']):
                same_A = False
                break
    results['twin_same_A'] = {
        'passed': same_A,
        'details': f"All MZ twin pairs have identical A values: {same_A}"
    }

    # Same sex
    same_sex = True
    for t1, t2 in twin_pairs:
        if t2 in df_indexed.index:
            if df_indexed.loc[t1, 'sex'] != df_indexed.loc[t2, 'sex']:
                same_sex = False
                break
    results['twin_same_sex'] = {
        'passed': same_sex,
        'details': f"All MZ twin pairs have same sex: {same_sex}"
    }

    # Twin rate (approximate check - twins only appear in non-founder generations)
    non_founders = df[df['mother'] != -1]
    if len(non_founders) > 0:
        observed_rate = len(twin_pairs) * 2 / len(non_founders)
        # Allow significant tolerance due to stochastic nature
        rate_ok = observed_rate <= p_mztwin * 3 + 0.01
        results['twin_rate'] = {
            'passed': rate_ok,
            'details': f"Twin rate in non-founders: {observed_rate:.4f} (param: {p_mztwin})",
            'expected_rate': p_mztwin,
            'observed_rate': float(observed_rate),
            'twin_pairs': len(twin_pairs)
        }
    else:
        results['twin_rate'] = {
            'passed': True,
            'details': "No non-founders to check twin rate"
        }

    return results


def validate_statistical(df, params):
    """Validate statistical properties of variance components."""
    results = {}
    A_param = params['A']
    C_param = params['C']
    E_param = params['E']

    # Use only founders (first generation) for initial variance estimates
    founders = df[df['mother'] == -1]

    # Variance of A
    var_A = founders['A'].var()
    var_A_ok = abs(var_A - A_param) < 0.1
    results['variance_A'] = {
        'passed': var_A_ok,
        'expected': A_param,
        'observed': float(var_A),
        'details': f"Var(A) in founders: {var_A:.4f} (expected: {A_param})"
    }

    # Variance of C
    var_C = founders['C'].var()
    var_C_ok = abs(var_C - C_param) < 0.1
    results['variance_C'] = {
        'passed': var_C_ok,
        'expected': C_param,
        'observed': float(var_C),
        'details': f"Var(C) in founders: {var_C:.4f} (expected: {C_param})"
    }

    # Variance of E
    var_E = founders['E'].var()
    var_E_ok = abs(var_E - E_param) < 0.1
    results['variance_E'] = {
        'passed': var_E_ok,
        'expected': E_param,
        'observed': float(var_E),
        'details': f"Var(E) in founders: {var_E:.4f} (expected: {E_param})"
    }

    # Total variance
    total_var = var_A + var_C + var_E
    total_ok = abs(total_var - 1.0) < 0.15
    results['total_variance'] = {
        'passed': total_ok,
        'expected': 1.0,
        'observed': float(total_var),
        'details': f"Total variance in founders: {total_var:.4f} (expected: 1.0)"
    }

    # C inheritance: siblings should share C (same mother)
    non_founders = df[df['mother'] != -1]
    if len(non_founders) > 0:
        # Group by mother and check C values
        c_by_mother = non_founders.groupby('mother')['C'].nunique()
        # Allow for floating point issues
        c_shared = (c_by_mother == 1).mean()
        c_ok = c_shared > 0.99
        results['c_inheritance'] = {
            'passed': c_ok,
            'details': f"Proportion of families with shared C: {c_shared:.4f}",
            'proportion_shared': float(c_shared)
        }
    else:
        results['c_inheritance'] = {
            'passed': True,
            'details': "No non-founders to check C inheritance"
        }

    # E independence: correlation of E between siblings should be ~0
    if len(non_founders) > 0:
        sibling_groups = non_founders.groupby('mother').filter(lambda x: len(x) >= 2)
        if len(sibling_groups) > 10:
            # Sample sibling pairs and compute correlation
            e_corrs = []
            for mother, group in sibling_groups.groupby('mother'):
                if len(group) >= 2:
                    e_vals = group['E'].values
                    for i in range(len(e_vals)):
                        for j in range(i + 1, len(e_vals)):
                            e_corrs.append((e_vals[i], e_vals[j]))
                if len(e_corrs) > 1000:
                    break
            if len(e_corrs) > 10:
                e1, e2 = zip(*e_corrs)
                e_corr = np.corrcoef(e1, e2)[0, 1]
                e_indep = abs(e_corr) < 0.1
                results['e_independence'] = {
                    'passed': e_indep,
                    'details': f"E correlation between siblings: {e_corr:.4f} (expected: ~0)",
                    'observed_correlation': float(e_corr)
                }
            else:
                results['e_independence'] = {
                    'passed': True,
                    'details': "Not enough sibling pairs to check E independence"
                }
        else:
            results['e_independence'] = {
                'passed': True,
                'details': "Not enough sibling groups to check E independence"
            }
    else:
        results['e_independence'] = {
            'passed': True,
            'details': "No non-founders to check E independence"
        }

    return results


def validate_heritability(df, params):
    """Validate heritability estimates."""
    results = {}
    A_param = params['A']

    # Get twin pairs for MZ correlation
    twins_df = df[df['twin'] != -1]
    df_indexed = df.set_index('id')

    twin_pairs = []
    seen = set()
    for idx, row in twins_df.iterrows():
        if row['id'] not in seen and row['twin'] != -1:
            if row['twin'] in df_indexed.index:
                twin_pairs.append((row['id'], row['twin']))
                seen.add(row['id'])
                seen.add(row['twin'])

    # MZ twin A correlation
    if len(twin_pairs) >= 10:
        mz_A1 = [df_indexed.loc[t1, 'A'] for t1, t2 in twin_pairs]
        mz_A2 = [df_indexed.loc[t2, 'A'] for t1, t2 in twin_pairs]
        mz_corr = np.corrcoef(mz_A1, mz_A2)[0, 1]
        mz_ok = mz_corr > 0.99  # Should be 1.0 for MZ twins
        results['mz_twin_A_correlation'] = {
            'passed': mz_ok,
            'expected': 1.0,
            'observed': float(mz_corr),
            'details': f"MZ twin A correlation: {mz_corr:.4f} (expected: 1.0)",
            'n_pairs': len(twin_pairs)
        }

        # MZ phenotype correlation
        mz_P1 = [df_indexed.loc[t1, 'A'] + df_indexed.loc[t1, 'C'] + df_indexed.loc[t1, 'E']
                 for t1, t2 in twin_pairs]
        mz_P2 = [df_indexed.loc[t2, 'A'] + df_indexed.loc[t2, 'C'] + df_indexed.loc[t2, 'E']
                 for t1, t2 in twin_pairs]
        mz_pheno_corr = np.corrcoef(mz_P1, mz_P2)[0, 1]
        results['mz_twin_phenotype_correlation'] = {
            'observed': float(mz_pheno_corr),
            'details': f"MZ twin phenotype correlation: {mz_pheno_corr:.4f}",
            'n_pairs': len(twin_pairs)
        }
    else:
        results['mz_twin_A_correlation'] = {
            'passed': True,
            'details': f"Not enough MZ twin pairs ({len(twin_pairs)}) to compute correlation"
        }
        mz_pheno_corr = None

    # DZ sibling A correlation (siblings with same parents but not MZ twins)
    non_founders = df[df['mother'] != -1].copy()
    non_founders['parent_key'] = non_founders['mother'].astype(str) + '_' + non_founders['father'].astype(str)

    # Find sibling pairs (same parents, not twins)
    dz_pairs = []
    for key, group in non_founders.groupby('parent_key'):
        non_twin_sibs = group[group['twin'] == -1]
        if len(non_twin_sibs) >= 2:
            ids = non_twin_sibs['id'].values
            for i in range(min(len(ids), 5)):  # Limit pairs per family
                for j in range(i + 1, min(len(ids), 5)):
                    dz_pairs.append((ids[i], ids[j]))
        if len(dz_pairs) > 5000:
            break

    if len(dz_pairs) >= 10:
        dz_A1 = [df_indexed.loc[t1, 'A'] for t1, t2 in dz_pairs]
        dz_A2 = [df_indexed.loc[t2, 'A'] for t1, t2 in dz_pairs]
        dz_corr = np.corrcoef(dz_A1, dz_A2)[0, 1]
        # DZ siblings should have ~0.5 correlation in A
        dz_ok = 0.3 <= dz_corr <= 0.7
        results['dz_sibling_A_correlation'] = {
            'passed': dz_ok,
            'expected': 0.5,
            'observed': float(dz_corr),
            'details': f"DZ sibling A correlation: {dz_corr:.4f} (expected: ~0.5)",
            'n_pairs': len(dz_pairs)
        }

        # DZ phenotype correlation
        dz_P1 = [df_indexed.loc[t1, 'A'] + df_indexed.loc[t1, 'C'] + df_indexed.loc[t1, 'E']
                 for t1, t2 in dz_pairs]
        dz_P2 = [df_indexed.loc[t2, 'A'] + df_indexed.loc[t2, 'C'] + df_indexed.loc[t2, 'E']
                 for t1, t2 in dz_pairs]
        dz_pheno_corr = np.corrcoef(dz_P1, dz_P2)[0, 1]
        results['dz_sibling_phenotype_correlation'] = {
            'observed': float(dz_pheno_corr),
            'details': f"DZ sibling phenotype correlation: {dz_pheno_corr:.4f}",
            'n_pairs': len(dz_pairs)
        }
    else:
        results['dz_sibling_A_correlation'] = {
            'passed': True,
            'details': f"Not enough DZ sibling pairs ({len(dz_pairs)}) to compute correlation"
        }
        dz_pheno_corr = None

    # Half sibling proportion (maternal half-sibs: same mother, different father)
    # For two children from the same mother, each independently has:
    #   - P(social father) = (1-p)
    #   - P(random father) = p
    # P(full sibs) = P(both have social father) ≈ (1-p)²  (ignoring rare same-random-father)
    # P(half sibs) = 1 - (1-p)² = 2p - p²
    p_nonsocial = params.get('p_nonsocial_father', 0)
    expected_half_sib_prop = 1 - (1 - p_nonsocial) ** 2

    # Count maternal sibling pairs and classify as full vs half
    n_full_sib_pairs = 0
    n_half_sib_pairs = 0
    for mother, group in non_founders.groupby('mother'):
        non_twin_sibs = group[group['twin'] == -1]
        if len(non_twin_sibs) >= 2:
            ids = non_twin_sibs['id'].values
            fathers = non_twin_sibs['father'].values
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if fathers[i] == fathers[j]:
                        n_full_sib_pairs += 1
                    else:
                        n_half_sib_pairs += 1

    total_maternal_pairs = n_full_sib_pairs + n_half_sib_pairs
    if total_maternal_pairs > 0:
        observed_half_sib_prop = n_half_sib_pairs / total_maternal_pairs
        # Allow reasonable tolerance for stochastic variation
        half_sib_ok = abs(observed_half_sib_prop - expected_half_sib_prop) < 0.05
        results['half_sib_proportion'] = {
            'passed': half_sib_ok,
            'expected': float(expected_half_sib_prop),
            'observed': float(observed_half_sib_prop),
            'n_full_sib_pairs': n_full_sib_pairs,
            'n_half_sib_pairs': n_half_sib_pairs,
            'details': f"Half-sib proportion: {observed_half_sib_prop:.4f} (expected: {expected_half_sib_prop:.4f})"
        }
    else:
        results['half_sib_proportion'] = {
            'passed': True,
            'details': "No maternal sibling pairs to check"
        }

    # Falconer's estimate: h² = 2(r_MZ - r_DZ)
    if mz_pheno_corr is not None and dz_pheno_corr is not None:
        falconer = 2 * (mz_pheno_corr - dz_pheno_corr)
        falconer_ok = abs(falconer - A_param) < 0.2
        results['falconer_estimate'] = {
            'passed': falconer_ok,
            'expected': A_param,
            'observed': float(falconer),
            'details': f"Falconer h² = 2(r_MZ - r_DZ) = {falconer:.4f} (expected: ~{A_param})"
        }
    else:
        results['falconer_estimate'] = {
            'passed': True,
            'details': "Cannot compute Falconer estimate without both MZ and DZ correlations"
        }

    # Parent-offspring A regression
    if len(non_founders) > 100:
        # Get midparent A values
        parent_A = df_indexed['A']
        non_founders_with_parents = non_founders[
            non_founders['mother'].isin(df_indexed.index) &
            non_founders['father'].isin(df_indexed.index)
        ].copy()

        if len(non_founders_with_parents) > 100:
            midparent_A = (
                parent_A.reindex(non_founders_with_parents['mother']).values +
                parent_A.reindex(non_founders_with_parents['father']).values
            ) / 2
            offspring_A = non_founders_with_parents['A'].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(midparent_A, offspring_A)
            results['parent_offspring_A_regression'] = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'details': f"Midparent-offspring A regression: slope={slope:.4f}, R²={r_value**2:.4f}"
            }

            # Parent-offspring phenotype regression
            parent_pheno = df_indexed['A'] + df_indexed['C'] + df_indexed['E']
            midparent_P = (
                parent_pheno.reindex(non_founders_with_parents['mother']).values +
                parent_pheno.reindex(non_founders_with_parents['father']).values
            ) / 2
            offspring_P = (
                non_founders_with_parents['A'] +
                non_founders_with_parents['C'] +
                non_founders_with_parents['E']
            ).values

            slope_P, intercept_P, r_value_P, p_value_P, std_err_P = stats.linregress(midparent_P, offspring_P)
            results['parent_offspring_phenotype_regression'] = {
                'slope': float(slope_P),
                'intercept': float(intercept_P),
                'r_squared': float(r_value_P ** 2),
                'details': f"Midparent-offspring phenotype regression: slope={slope_P:.4f}, R²={r_value_P**2:.4f}"
            }
        else:
            results['parent_offspring_A_regression'] = {
                'details': "Not enough offspring with both parents in data"
            }
            results['parent_offspring_phenotype_regression'] = {
                'details': "Not enough offspring with both parents in data"
            }
    else:
        results['parent_offspring_A_regression'] = {
            'details': "Not enough non-founders for regression"
        }
        results['parent_offspring_phenotype_regression'] = {
            'details': "Not enough non-founders for regression"
        }

    return results


def compute_per_generation_stats(df, params):
    """Compute per-generation statistics."""
    N = params['N']
    ngen = params['ngen']

    # Determine generation based on ID ranges
    # First generation: IDs 0 to N-1 (founders with parent=-1)
    # Subsequent generations: IDs N to 2N-1, 2N to 3N-1, etc.

    results = {}
    for gen in range(1, ngen + 1):
        start_id = (gen - 1) * N
        end_id = gen * N
        gen_df = df[(df['id'] >= start_id) & (df['id'] < end_id)]

        phenotype = gen_df['A'] + gen_df['C'] + gen_df['E']

        results[f'generation_{gen}'] = {
            'n': len(gen_df),
            'phenotype_mean': float(phenotype.mean()),
            'phenotype_variance': float(phenotype.var()),
            'phenotype_sd': float(phenotype.std()),
            'A_mean': float(gen_df['A'].mean()),
            'A_var': float(gen_df['A'].var()),
            'C_mean': float(gen_df['C'].mean()),
            'C_var': float(gen_df['C'].var()),
            'E_mean': float(gen_df['E'].mean()),
            'E_var': float(gen_df['E'].var()),
        }

    return results


def validate_population(df, params):
    """Validate population-level properties."""
    results = {}
    N = params['N']
    ngen = params['ngen']
    fam_size = params['fam_size']

    # Generation sizes
    gen_sizes = []
    for gen in range(1, ngen + 1):
        start_id = (gen - 1) * N
        end_id = gen * N
        gen_df = df[(df['id'] >= start_id) & (df['id'] < end_id)]
        gen_sizes.append(len(gen_df))

    all_correct = all(s == N for s in gen_sizes)
    results['generation_sizes'] = {
        'passed': all_correct,
        'expected': N,
        'observed': gen_sizes,
        'details': f"Generation sizes: {gen_sizes} (expected: {N} each)"
    }

    # Generation count
    results['generation_count'] = {
        'passed': len(gen_sizes) == ngen,
        'expected': ngen,
        'observed': len(gen_sizes),
        'details': f"Number of generations: {len(gen_sizes)} (expected: {ngen})"
    }

    # Mean family size (in non-founder generations)
    non_founders = df[df['mother'] != -1]
    if len(non_founders) > 0:
        family_sizes = non_founders.groupby('mother').size()
        mean_fam = family_sizes.mean()
        # Allow tolerance for stochastic variation
        fam_ok = abs(mean_fam - fam_size) < fam_size * 0.5
        results['family_size'] = {
            'passed': fam_ok,
            'expected': fam_size,
            'observed': float(mean_fam),
            'details': f"Mean family size: {mean_fam:.2f} (expected: ~{fam_size})"
        }
    else:
        results['family_size'] = {
            'passed': True,
            'details': "No non-founders to check family size"
        }

    return results


def run_validation(pedigree_path, params_path):
    """Run all validation checks and return results."""
    # Load data
    df = pd.read_parquet(pedigree_path)
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Run all validation categories
    results = {
        'structural': validate_structural(df, params),
        'twins': validate_twins(df, params),
        'statistical': validate_statistical(df, params),
        'heritability': validate_heritability(df, params),
        'population': validate_population(df, params),
        'per_generation': compute_per_generation_stats(df, params),
    }

    # Compute summary
    checks_passed = 0
    checks_failed = 0

    for category, checks in results.items():
        if category == 'per_generation':
            continue  # Not pass/fail checks
        for check_name, check_result in checks.items():
            if 'passed' in check_result:
                if check_result['passed']:
                    checks_passed += 1
                else:
                    checks_failed += 1

    results['summary'] = {
        'passed': checks_failed == 0,
        'checks_passed': checks_passed,
        'checks_failed': checks_failed,
        'checks_total': checks_passed + checks_failed,
    }

    # Add params for reference
    results['parameters'] = params

    return results


if __name__ == "__main__":
    # Get paths from Snakemake
    pedigree_path = snakemake.input.pedigree
    params_path = snakemake.input.params
    output_path = snakemake.output.report

    # Run validation
    results = run_validation(pedigree_path, params_path)

    # Convert numpy types to native Python for clean YAML serialization
    results = to_native(results)

    # Write YAML report
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
