"""
Compute per-rep phenotype statistics for downstream plotting.

Reads a single phenotype.weibull.parquet and produces:
  - phenotype_stats.yaml: scalar and array statistics
  - phenotype_samples.parquet: 10K row random sample for scatter/histogram plots
"""

import argparse

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import norm, linregress
from scipy.optimize import minimize_scalar
from scipy.special import owens_t


def _bvn_pos(h, k, r, sq):
    """BVN CDF for h >= 0, k >= 0, |r| < 1 using Owen's T function."""
    if h < 1e-15 and k < 1e-15:
        return 0.25 + np.arcsin(r) / (2.0 * np.pi)
    if h < 1e-15:
        return 0.5 * norm.cdf(k) - owens_t(k, -r / sq)
    if k < 1e-15:
        return 0.5 * norm.cdf(h) - owens_t(h, -r / sq)
    return (0.5 * norm.cdf(h) + 0.5 * norm.cdf(k)
            - owens_t(h, (k - r * h) / (h * sq))
            - owens_t(k, (h - r * k) / (k * sq)))


def _bvn_cdf(h, k, r):
    """Fast bivariate normal CDF using Owen's T function."""
    if abs(r) < 1e-15:
        return norm.cdf(h) * norm.cdf(k)
    sq = np.sqrt(1.0 - r * r)
    if h < 0 and k < 0:
        return 1.0 - norm.cdf(-h) - norm.cdf(-k) + _bvn_pos(-h, -k, r, sq)
    if h < 0:
        return norm.cdf(k) - _bvn_pos(-h, k, -r, np.sqrt(1.0 - r * r))
    if k < 0:
        return norm.cdf(h) - _bvn_pos(h, -k, -r, np.sqrt(1.0 - r * r))
    return _bvn_pos(h, k, r, sq)


def tetrachoric_corr_se(a, b):
    """Estimate tetrachoric correlation and SE from two binary arrays.

    Uses MLE with Owen's T-based BVN CDF (57x faster than multivariate_normal.cdf).
    Returns (r, se) tuple.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    n11 = np.sum(a & b)
    n10 = np.sum(a & ~b)
    n01 = np.sum(~a & b)
    n00 = np.sum(~a & ~b)

    p_a = a.mean()
    p_b = b.mean()

    if p_a == 0 or p_a == 1 or p_b == 0 or p_b == 1:
        return np.nan, np.nan

    t_a = norm.ppf(1 - p_a)
    t_b = norm.ppf(1 - p_b)

    # Precompute constants that don't depend on r
    phi_ta = norm.cdf(t_a)
    phi_tb = norm.cdf(t_b)

    # For typical disease prevalence (<50%), both thresholds are positive
    # and we can use the fast precomputed path
    both_positive = t_a > 1e-15 and t_b > 1e-15

    def neg_log_lik(r):
        if both_positive:
            sq = np.sqrt(1.0 - r * r)
            p00 = (0.5 * phi_ta + 0.5 * phi_tb
                   - owens_t(t_a, (t_b - r * t_a) / (t_a * sq))
                   - owens_t(t_b, (t_a - r * t_b) / (t_b * sq)))
        else:
            p00 = _bvn_cdf(t_a, t_b, r)
        p01 = phi_ta - p00
        p10 = phi_tb - p00
        p11 = 1 - p00 - p01 - p10
        eps = 1e-15
        return -(n11 * np.log(max(p11, eps)) + n10 * np.log(max(p10, eps)) +
                 n01 * np.log(max(p01, eps)) + n00 * np.log(max(p00, eps)))

    result = minimize_scalar(neg_log_lik, bounds=(-0.999, 0.999), method="bounded")
    r = result.x

    if np.isnan(r):
        return np.nan, np.nan

    # SE via observed Fisher information (numerical second derivative)
    try:
        dx = 1e-4
        r_clamped = np.clip(r, -0.999 + dx, 0.999 - dx)
        fisher_info = (neg_log_lik(r_clamped + dx) - 2 * neg_log_lik(r_clamped) +
                       neg_log_lik(r_clamped - dx)) / dx ** 2
        se = 1.0 / np.sqrt(fisher_info) if fisher_info > 0 else np.nan
    except Exception:
        se = np.nan

    return r, se


# Backward-compatible wrappers used by compute_threshold_stats.py
def tetrachoric_corr(a, b):
    """Estimate tetrachoric correlation from two binary arrays via MLE."""
    r, _ = tetrachoric_corr_se(a, b)
    return r


def tetrachoric_se(r, a, b):
    """Approximate SE of tetrachoric correlation."""
    _, se = tetrachoric_corr_se(a, b)
    return se


def extract_relationship_pairs(df, seed=42):
    """Extract relationship pairs as aligned row-index arrays.

    Performance-optimized version using dict-based lookups and pandas merge
    for sibling extraction instead of groupby loops.
    """
    # Map (id) -> row position via numpy array for vectorized lookup
    ids_arr = df["id"].values.astype(np.int64)
    id_to_row = np.full(ids_arr.max() + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)

    def resolve_rows(ids):
        """Map array of ids to row positions; -1 if missing."""
        ids = np.asarray(ids, dtype=np.int64)
        mask = (ids >= 0) & (ids < len(id_to_row))
        result = np.full(len(ids), -1, dtype=np.int32)
        result[mask] = id_to_row[ids[mask]]
        return result

    pairs = {}

    # MZ twins: twin != -1, deduplicate by keeping id < twin
    twins = df[df["twin"] != -1]
    ta = twins["id"].values.astype(int)
    tb = twins["twin"].values.astype(int)
    mask = ta < tb
    pairs["MZ twin"] = (resolve_rows(ta[mask]), resolve_rows(tb[mask]))

    # Full sibs and half sibs via self-merge on mother and father
    non_twin_nf = df[(df["mother"] != -1) & (df["twin"] == -1)].copy()
    non_twin_nf["_row"] = non_twin_nf.index.values

    full_rows_1, full_rows_2 = [], []
    mat_half_rows_1, mat_half_rows_2 = [], []

    # Maternal sibs: same mother
    sib_counts = non_twin_nf.groupby("mother").size()
    multi_mothers = sib_counts[sib_counts >= 2].index
    mat_sib = non_twin_nf[non_twin_nf["mother"].isin(multi_mothers)]

    if len(mat_sib) > 0:
        mat_pairs = mat_sib[["mother", "father", "_row"]].merge(
            mat_sib[["mother", "father", "_row"]],
            on="mother", suffixes=("_1", "_2"),
        )
        mat_pairs = mat_pairs[mat_pairs["_row_1"] < mat_pairs["_row_2"]]
        same_father = mat_pairs["father_1"] == mat_pairs["father_2"]

        full_rows_1.append(mat_pairs.loc[same_father, "_row_1"].values)
        full_rows_2.append(mat_pairs.loc[same_father, "_row_2"].values)
        mat_half_rows_1.append(mat_pairs.loc[~same_father, "_row_1"].values)
        mat_half_rows_2.append(mat_pairs.loc[~same_father, "_row_2"].values)

    # Paternal half-sibs: same father, different mother
    pat_half_rows_1, pat_half_rows_2 = [], []
    pat_counts = non_twin_nf.groupby("father").size()
    multi_fathers = pat_counts[pat_counts >= 2].index
    pat_sib = non_twin_nf[non_twin_nf["father"].isin(multi_fathers)]

    if len(pat_sib) > 0:
        pat_pairs = pat_sib[["mother", "father", "_row"]].merge(
            pat_sib[["mother", "father", "_row"]],
            on="father", suffixes=("_1", "_2"),
        )
        pat_pairs = pat_pairs[pat_pairs["_row_1"] < pat_pairs["_row_2"]]
        diff_mother = pat_pairs["mother_1"] != pat_pairs["mother_2"]
        pat_half_rows_1.append(pat_pairs.loc[diff_mother, "_row_1"].values)
        pat_half_rows_2.append(pat_pairs.loc[diff_mother, "_row_2"].values)

    pairs["Full sib"] = (
        np.concatenate(full_rows_1) if full_rows_1 else np.array([], dtype=int),
        np.concatenate(full_rows_2) if full_rows_1 else np.array([], dtype=int),
    )
    pairs["Maternal half sib"] = (
        np.concatenate(mat_half_rows_1) if mat_half_rows_1 else np.array([], dtype=int),
        np.concatenate(mat_half_rows_2) if mat_half_rows_1 else np.array([], dtype=int),
    )
    pairs["Paternal half sib"] = (
        np.concatenate(pat_half_rows_1) if pat_half_rows_1 else np.array([], dtype=int),
        np.concatenate(pat_half_rows_2) if pat_half_rows_1 else np.array([], dtype=int),
    )

    # Mother-offspring and Father-offspring
    all_nf = df[df["mother"] != -1]
    child_rows = all_nf.index.values
    mother_rows = resolve_rows(all_nf["mother"].values.astype(int))
    father_rows = resolve_rows(all_nf["father"].values.astype(int))

    m_valid = mother_rows >= 0
    f_valid = father_rows >= 0
    pairs["Mother-offspring"] = (child_rows[m_valid], mother_rows[m_valid])
    pairs["Father-offspring"] = (child_rows[f_valid], father_rows[f_valid])

    # 1st cousins via grandparents (numpy-native, no pandas merges)
    child_ids = all_nf["id"].values.astype(np.int64)
    mother_ids = all_nf["mother"].values.astype(np.int64)
    father_ids = all_nf["father"].values.astype(np.int64)
    n_children = len(child_ids)

    # Look up each parent's parents (grandparents) via id_to_row
    df_mothers_col = df["mother"].values.astype(np.int64)
    df_fathers_col = df["father"].values.astype(np.int64)
    mother_row = resolve_rows(mother_ids)
    father_row = resolve_rows(father_ids)

    gp_ids = np.full((n_children, 4), -1, dtype=np.int64)
    m_ok = mother_row >= 0
    gp_ids[m_ok, 0] = df_mothers_col[mother_row[m_ok]]
    gp_ids[m_ok, 1] = df_fathers_col[mother_row[m_ok]]
    f_ok = father_row >= 0
    gp_ids[f_ok, 2] = df_mothers_col[father_row[f_ok]]
    gp_ids[f_ok, 3] = df_fathers_col[father_row[f_ok]]

    # Flat (child, parent, grandparent) arrays -- 4 entries per child
    gp_child = np.tile(child_ids, 4)
    gp_parent = np.concatenate([mother_ids, mother_ids, father_ids, father_ids])
    gp_gp = np.concatenate([gp_ids[:, 0], gp_ids[:, 1], gp_ids[:, 2], gp_ids[:, 3]])

    # Filter invalid grandparents
    valid_gp = gp_gp >= 0
    gp_child = gp_child[valid_gp]
    gp_parent = gp_parent[valid_gp]
    gp_gp = gp_gp[valid_gp]

    # Cap grandparents to limit pair explosion
    unique_gp_arr = np.unique(gp_gp)
    if len(unique_gp_arr) > 100000:
        rng = np.random.default_rng(seed)
        selected_gp = rng.choice(unique_gp_arr, 100000, replace=False)
        gp_mask = np.isin(gp_gp, selected_gp)
        gp_child = gp_child[gp_mask]
        gp_parent = gp_parent[gp_mask]
        gp_gp = gp_gp[gp_mask]

    # Sort by grandparent for grouping
    sort_idx = np.argsort(gp_gp, kind="mergesort")
    gp_child = gp_child[sort_idx]
    gp_parent = gp_parent[sort_idx]
    gp_gp = gp_gp[sort_idx]

    _, group_starts, group_counts = np.unique(
        gp_gp, return_index=True, return_counts=True
    )

    # Only groups with >= 2 members can produce cousin pairs
    multi = group_counts >= 2
    group_starts = group_starts[multi]
    group_counts = group_counts[multi]

    # Generate pair indices by batching groups of the same size
    pair_i_parts = []
    pair_j_parts = []
    for size in np.unique(group_counts):
        gs = group_starts[group_counts == size]
        ii, jj = np.triu_indices(size, k=1)
        all_i = (gs[:, np.newaxis] + ii[np.newaxis, :]).ravel()
        all_j = (gs[:, np.newaxis] + jj[np.newaxis, :]).ravel()
        pair_i_parts.append(all_i)
        pair_j_parts.append(all_j)

    pair_i = np.concatenate(pair_i_parts)
    pair_j = np.concatenate(pair_j_parts)

    # Filter: different parents only (cousins, not siblings)
    diff_parent = gp_parent[pair_i] != gp_parent[pair_j]
    c1_raw = gp_child[pair_i[diff_parent]]
    c2_raw = gp_child[pair_j[diff_parent]]

    # Canonical ordering and dedup (children may share multiple grandparents)
    c1 = np.minimum(c1_raw, c2_raw)
    c2 = np.maximum(c1_raw, c2_raw)

    # Pack pair into single int64 for fast 1D dedup
    max_id = int(c2.max()) + 1
    pair_keys = c1.astype(np.int64) * max_id + c2.astype(np.int64)
    unique_keys = np.unique(pair_keys)
    c1_final = unique_keys // max_id
    c2_final = unique_keys % max_id

    c_idx1 = resolve_rows(c1_final)
    c_idx2 = resolve_rows(c2_final)
    c_valid = (c_idx1 >= 0) & (c_idx2 >= 0)
    pairs["1st cousin"] = (c_idx1[c_valid], c_idx2[c_valid])

    return pairs


def compute_mortality(df, censor_age):
    """Compute mortality rates and cumulative mortality by decade."""
    decade_edges = np.arange(0, censor_age + 10, 10)
    mortality_rates = []
    decade_labels = []

    for i in range(len(decade_edges) - 1):
        lo, hi = decade_edges[i], decade_edges[i + 1]
        if lo >= censor_age:
            break
        alive_at_start = (df["death_age"].values >= lo).sum()
        died_in_decade = (
            (df["death_age"].values >= lo)
            & (df["death_age"].values < hi)
            & (df["death_age"].values < censor_age)
        ).sum()
        rate = float(died_in_decade / alive_at_start) if alive_at_start > 0 else 0.0
        mortality_rates.append(rate)
        decade_labels.append(f"{int(lo)}-{int(hi - 1)}")

    return {"decade_labels": decade_labels, "rates": mortality_rates}


def compute_cumulative_incidence(df, censor_age, n_points=200):
    """Compute cumulative incidence curves using searchsorted (O(N log N)).

    Returns both true (from uncensored t) and observed (from affected + t_observed)
    curves per trait.
    """
    ages = np.linspace(0, censor_age, n_points)
    n = len(df)
    result = {}

    for trait_num in [1, 2]:
        aff_vals = df[f"affected{trait_num}"].values.astype(bool)
        t_obs_vals = df[f"t_observed{trait_num}"].values
        t_raw_vals = df[f"t{trait_num}"].values

        # Observed incidence: fraction affected with t_observed <= age
        t_affected = t_obs_vals[aff_vals]
        sorted_obs = np.sort(t_affected)
        obs_inc = np.searchsorted(sorted_obs, ages, side="right") / n

        # True incidence: fraction with t_raw <= age (all events, ignoring censoring)
        sorted_raw = np.sort(t_raw_vals)
        true_inc = np.searchsorted(sorted_raw, ages, side="right") / n

        # Find age when 50% of lifetime cases diagnosed (observed curve)
        lifetime_prev = obs_inc[-1]
        half_target = lifetime_prev / 2
        idx_50 = np.searchsorted(obs_inc, half_target)
        half_target_age = float(ages[min(idx_50, len(ages) - 1)])

        result[f"trait{trait_num}"] = {
            "ages": ages.tolist(),
            "observed_values": obs_inc.tolist(),
            "true_values": true_inc.tolist(),
            "half_target_age": half_target_age,
        }

    return result


def compute_censoring_windows(df, censor_age, young_gen_censoring,
                               middle_gen_censoring, old_gen_censoring, n_points=300):
    """Compute per-generation censoring window curves using searchsorted."""
    if "generation" not in df.columns:
        return None

    max_gen = df["generation"].max()
    gen_order = [max_gen - 2, max_gen - 1, max_gen]
    gen_names = ["old", "middle", "young"]
    gen_windows = [old_gen_censoring, middle_gen_censoring, young_gen_censoring]

    ages = np.linspace(0, censor_age, n_points)
    result = {}

    for gen, name, (win_lo, win_hi) in zip(gen_order, gen_names, gen_windows):
        gen_df = df[df["generation"] == gen]
        n_gen = len(gen_df)
        if n_gen == 0:
            result[name] = {"n": 0}
            continue

        gen_result = {"n": int(n_gen)}
        for trait_num in [1, 2]:
            t_raw = gen_df[f"t{trait_num}"].values
            t_obs = gen_df[f"t_observed{trait_num}"].values
            affected = gen_df[f"affected{trait_num}"].values.astype(bool)
            death_censored = gen_df[f"death_censored{trait_num}"].values.astype(bool)

            # True incidence: fraction with t_raw <= age (O(N log N))
            sorted_raw = np.sort(t_raw)
            true_inc = np.searchsorted(sorted_raw, ages, side="right") / n_gen

            # Observed incidence: fraction affected with t_obs <= age
            t_obs_affected = t_obs[affected]
            sorted_obs = np.sort(t_obs_affected)
            obs_inc = np.searchsorted(sorted_obs, ages, side="right") / n_gen

            pct_affected = float(affected.mean())
            left_censored = float((t_raw < win_lo).sum() / n_gen)
            right_censored = float((t_raw > win_hi).sum() / n_gen)
            pct_death_cens = float(
                (death_censored & ~(t_raw < win_lo) & ~(t_raw > win_hi)).mean()
            )

            gen_result[f"trait{trait_num}"] = {
                "true_incidence": true_inc.tolist(),
                "observed_incidence": obs_inc.tolist(),
                "pct_affected": pct_affected,
                "left_censored": left_censored,
                "right_censored": right_censored,
                "death_censored": pct_death_cens,
            }

        result[name] = gen_result

    return {"generations": result, "censoring_ages": ages.tolist(), "censor_age": int(censor_age)}


def compute_tetrachoric(df, seed=42, pairs=None):
    """Compute tetrachoric correlations for all relationship types."""
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    pair_types = ["MZ twin", "Full sib", "Mother-offspring", "Father-offspring", "Maternal half sib", "Paternal half sib", "1st cousin"]
    result = {}

    for trait_num in [1, 2]:
        affected = df[f"affected{trait_num}"].values.astype(bool)
        trait_result = {}
        for ptype in pair_types:
            idx1, idx2 = pairs[ptype]
            n_p = len(idx1)
            if n_p < 10:
                trait_result[ptype] = {"r": None, "se": None, "n_pairs": int(n_p)}
                continue
            a_vals = affected[idx1]
            b_vals = affected[idx2]
            r_tet, se = tetrachoric_corr_se(a_vals, b_vals)
            trait_result[ptype] = {
                "r": float(r_tet) if not np.isnan(r_tet) else None,
                "se": float(se) if not np.isnan(se) else None,
                "n_pairs": int(n_p),
            }
        result[f"trait{trait_num}"] = trait_result

    return result


def compute_liability_correlations(df, seed=42, pairs=None):
    """Compute Pearson correlation on liability values for each relationship pair type."""
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    pair_types = ["MZ twin", "Full sib", "Mother-offspring", "Father-offspring", "Maternal half sib", "Paternal half sib", "1st cousin"]
    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].values
        trait_result = {}
        for ptype in pair_types:
            idx1, idx2 = pairs[ptype]
            if len(idx1) < 10:
                trait_result[ptype] = None
                continue
            trait_result[ptype] = float(np.corrcoef(liability[idx1], liability[idx2])[0, 1])
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_regression(df):
    """Compute regression stats (liability vs age at onset) for affected individuals."""
    result = {}
    for trait_num in [1, 2]:
        affected_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"
        liability_col = f"liability{trait_num}"

        if liability_col not in df.columns:
            result[f"trait{trait_num}"] = None
            continue

        affected = df[df[affected_col] == True].dropna(subset=[liability_col, t_col])
        x = affected[liability_col].values
        y = affected[t_col].values

        if len(x) < 2:
            result[f"trait{trait_num}"] = None
            continue

        reg = linregress(x, y)
        result[f"trait{trait_num}"] = {
            "slope": float(reg.slope),
            "intercept": float(reg.intercept),
            "r2": float(reg.rvalue ** 2),
            "n": int(len(x)),
        }

    return result


def compute_prevalence(df):
    """Compute trait prevalence."""
    return {
        "trait1": float(df["affected1"].mean()),
        "trait2": float(df["affected2"].mean()),
    }


def create_sample(df, seed=42, n_samples=100000):
    """Create a downsampled DataFrame for scatter/histogram/violin plots."""
    if len(df) <= n_samples:
        return df.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), n_samples, replace=False)
    return df.iloc[idx].copy()


def main(phenotype_path, censor_age, stats_output, samples_output,
         seed=42, young_gen_censoring=None, middle_gen_censoring=None,
         old_gen_censoring=None):
    """Compute all stats for a single rep and write outputs."""
    df = pd.read_parquet(phenotype_path)

    print(f"Computing stats for {phenotype_path} ({len(df)} rows)")

    stats = {}
    stats["n_individuals"] = int(len(df))

    # Prevalence
    stats["prevalence"] = compute_prevalence(df)

    # Mortality
    stats["mortality"] = compute_mortality(df, censor_age)

    # Regression
    stats["regression"] = compute_regression(df)

    # Cumulative incidence
    stats["cumulative_incidence"] = compute_cumulative_incidence(df, censor_age)

    # Censoring windows
    if young_gen_censoring is not None:
        stats["censoring"] = compute_censoring_windows(
            df, censor_age, young_gen_censoring,
            middle_gen_censoring, old_gen_censoring,
        )

    # Extract relationship pairs once (used by both liability and tetrachoric)
    print("Extracting relationship pairs...")
    pairs = extract_relationship_pairs(df, seed=seed)

    # Liability correlations
    print("Computing liability correlations...")
    stats["liability_correlations"] = compute_liability_correlations(df, seed=seed, pairs=pairs)

    # Tetrachoric correlations
    print("Computing tetrachoric correlations...")
    stats["tetrachoric"] = compute_tetrachoric(df, seed=seed, pairs=pairs)

    # Write stats YAML
    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    print(f"Stats written to {stats_path}")

    # Write downsampled parquet
    sample_df = create_sample(df, seed=seed)
    samples_path = Path(samples_output)
    sample_df.to_parquet(samples_path, index=False)
    print(f"Sample ({len(sample_df)} rows) written to {samples_path}")


def cli():
    """Command-line interface for computing phenotype statistics."""
    parser = argparse.ArgumentParser(description="Compute phenotype statistics")
    parser.add_argument("phenotype", help="Input phenotype parquet")
    parser.add_argument("censor_age", type=float, help="Censoring age")
    parser.add_argument("stats_output", help="Output stats YAML")
    parser.add_argument("samples_output", help="Output samples parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--young-gen-censoring", type=float, nargs=2, default=None)
    parser.add_argument("--middle-gen-censoring", type=float, nargs=2, default=None)
    parser.add_argument("--old-gen-censoring", type=float, nargs=2, default=None)
    args = parser.parse_args()

    main(args.phenotype, args.censor_age, args.stats_output, args.samples_output,
         seed=args.seed, young_gen_censoring=args.young_gen_censoring,
         middle_gen_censoring=args.middle_gen_censoring,
         old_gen_censoring=args.old_gen_censoring)
