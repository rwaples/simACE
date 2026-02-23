"""
Compute per-rep phenotype statistics for downstream plotting.

Reads a single phenotype.weibull.parquet and produces:
  - phenotype_stats.yaml: scalar and array statistics
  - phenotype_samples.parquet: 10K row random sample for scatter/histogram plots
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import norm, linregress
from scipy.optimize import minimize_scalar
from scipy.special import owens_t

import logging
import time
logger = logging.getLogger(__name__)


def _bvn_pos(h: float, k: float, r: float, sq: float) -> float:
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


def _bvn_cdf(h: float, k: float, r: float) -> float:
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


def tetrachoric_corr_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Estimate tetrachoric correlation and SE from two binary arrays.

    Uses MLE with Owen's T-based BVN CDF (57x faster than multivariate_normal.cdf).
    Returns (r, se) tuple.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    n_pairs = len(a)
    if n_pairs < 50:
        logger.warning("tetrachoric_corr_se: n_pairs=%d < 50, SE may be unreliable", n_pairs)

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

    def neg_log_lik(r: float) -> float:
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

    # SE via analytic Fisher information using bivariate normal PDF
    try:
        one_minus_r2 = 1.0 - r * r
        if one_minus_r2 <= 0:
            se = np.nan
        else:
            bvn_pdf = (1.0 / (2.0 * np.pi * np.sqrt(one_minus_r2))) * np.exp(
                -(t_a**2 - 2.0 * r * t_a * t_b + t_b**2) / (2.0 * one_minus_r2)
            )
            # Cell probabilities at MLE
            if both_positive:
                sq = np.sqrt(one_minus_r2)
                p00 = (0.5 * phi_ta + 0.5 * phi_tb
                       - owens_t(t_a, (t_b - r * t_a) / (t_a * sq))
                       - owens_t(t_b, (t_a - r * t_b) / (t_b * sq)))
            else:
                p00 = _bvn_cdf(t_a, t_b, r)
            p01 = phi_ta - p00
            p10 = phi_tb - p00
            p11 = 1.0 - p00 - p01 - p10
            denom = p00 * p01 * p10 * p11
            if denom <= 0:
                se = np.nan
            else:
                fisher_info = n_pairs * bvn_pdf**2 / denom
                se = 1.0 / np.sqrt(fisher_info)
    except Exception:
        se = np.nan

    return r, se


# Backward-compatible wrappers used by compute_threshold_stats.py
def tetrachoric_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Estimate tetrachoric correlation from two binary arrays via MLE."""
    r, _ = tetrachoric_corr_se(a, b)
    return r


def tetrachoric_se(r: float, a: np.ndarray, b: np.ndarray) -> float:
    """Approximate SE of tetrachoric correlation."""
    _, se = tetrachoric_corr_se(a, b)
    return se


def extract_relationship_pairs(df: pd.DataFrame, seed: int = 42) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract relationship pairs as aligned row-index arrays.

    Performance-optimized version using dict-based lookups and pandas merge
    for sibling extraction instead of groupby loops.
    """
    # Map (id) -> row position via numpy array for vectorized lookup
    ids_arr = df["id"].values.astype(np.int64)
    id_to_row = np.full(ids_arr.max() + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)

    def resolve_rows(ids: np.ndarray) -> np.ndarray:
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
        logger.info(
            "extract_relationship_pairs: %d grandparents exceed 100K cap, sampling subset",
            len(unique_gp_arr),
        )
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


def compute_mortality(df: pd.DataFrame, censor_age: float) -> dict[str, Any]:
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


def compute_cumulative_incidence(df: pd.DataFrame, censor_age: float, n_points: int = 200) -> dict[str, Any]:
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


def compute_censoring_windows(
    df: pd.DataFrame,
    censor_age: float,
    gen_censoring: dict[int, list[float]],
    n_points: int = 300,
) -> dict[str, Any] | None:
    """Compute per-generation censoring window curves using searchsorted."""
    if "generation" not in df.columns:
        return None

    ages = np.linspace(0, censor_age, n_points)
    result = {}

    for gen in sorted(gen_censoring.keys()):
        win_lo, win_hi = gen_censoring[gen]
        name = f"gen{gen}"
        gen_df = df[df["generation"] == gen]
        n_gen = len(gen_df)
        if n_gen == 0:
            result[name] = {"n": 0}
            continue

        gen_result: dict[str, Any] = {"n": int(n_gen)}
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


def compute_tetrachoric(df: pd.DataFrame, seed: int = 42, pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> dict[str, Any]:
    """Compute tetrachoric correlations for all relationship types."""
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    pair_types = ["MZ twin", "Full sib", "Mother-offspring", "Father-offspring", "Maternal half sib", "Paternal half sib", "1st cousin"]
    result = {}

    for trait_num in [1, 2]:
        affected = df[f"affected{trait_num}"].values.astype(bool)
        trait_result: dict[str, dict[str, int | float | None]] = {}
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


def compute_tetrachoric_by_generation(
    df: pd.DataFrame, seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compute tetrachoric correlations broken out by the last 3 pedigree generations.

    For each generation, filters each pair type to pairs where the first index
    (child for parent-offspring, either member for same-gen pairs) belongs to
    that generation, then computes tetrachoric r + SE and liability Pearson r.

    Args:
        df: phenotype DataFrame with 'generation', 'affected{N}', 'liability{N}' columns
        seed: random seed (passed through for pair extraction)
        pairs: pre-extracted relationship pair indices from extract_relationship_pairs()

    Returns:
        dict keyed by "gen{N}" -> "trait{N}" -> pair_type -> {r, se, n_pairs, liability_r}
    """
    if "generation" not in df.columns:
        return {}
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)

    gen_arr = df["generation"].values
    max_gen = int(gen_arr.max())
    plot_gens = list(range(max(1, max_gen - 2), max_gen + 1))

    pair_types = [
        "MZ twin", "Full sib", "Mother-offspring", "Father-offspring",
        "Maternal half sib", "Paternal half sib", "1st cousin",
    ]

    result = {}
    for gen in plot_gens:
        gen_result = {}
        for trait_num in [1, 2]:
            affected = df[f"affected{trait_num}"].values.astype(bool)
            liability = df[f"liability{trait_num}"].values
            trait_result: dict[str, dict[str, int | float | None]] = {}

            for ptype in pair_types:
                idx1, idx2 = pairs[ptype]
                # Filter pairs where idx1 belongs to this generation
                mask = gen_arr[idx1] == gen
                g_idx1 = idx1[mask]
                g_idx2 = idx2[mask]
                n_p = len(g_idx1)

                if n_p < 10:
                    trait_result[ptype] = {
                        "r": None, "se": None, "n_pairs": int(n_p),
                        "liability_r": None,
                    }
                    continue

                a_vals = affected[g_idx1]
                b_vals = affected[g_idx2]
                r_tet, se = tetrachoric_corr_se(a_vals, b_vals)

                # Liability Pearson r for reference lines
                liab_r = float(np.corrcoef(liability[g_idx1], liability[g_idx2])[0, 1])

                trait_result[ptype] = {
                    "r": float(r_tet) if not np.isnan(r_tet) else None,
                    "se": float(se) if not np.isnan(se) else None,
                    "n_pairs": int(n_p),
                    "liability_r": liab_r if not np.isnan(liab_r) else None,
                }

            gen_result[f"trait{trait_num}"] = trait_result
        result[f"gen{gen}"] = gen_result

    return result


def compute_liability_correlations(df: pd.DataFrame, seed: int = 42, pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> dict[str, Any]:
    """Compute Pearson correlation on liability values for each relationship pair type."""
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    pair_types = ["MZ twin", "Full sib", "Mother-offspring", "Father-offspring", "Maternal half sib", "Paternal half sib", "1st cousin"]
    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].values
        trait_result: dict[str, float | None] = {}
        for ptype in pair_types:
            idx1, idx2 = pairs[ptype]
            if len(idx1) < 10:
                trait_result[ptype] = None
                continue
            trait_result[ptype] = float(np.corrcoef(liability[idx1], liability[idx2])[0, 1])
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_regression(df: pd.DataFrame) -> dict[str, Any]:
    """Compute regression stats (liability vs age at onset) for affected individuals."""
    result: dict[str, dict[str, float | int] | None] = {}
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


def compute_prevalence(df: pd.DataFrame) -> dict[str, float]:
    """Compute trait prevalence."""
    return {
        "trait1": float(df["affected1"].mean()),
        "trait2": float(df["affected2"].mean()),
    }


def create_sample(df: pd.DataFrame, seed: int = 42, n_samples: int = 100000) -> pd.DataFrame:
    """Create a downsampled DataFrame for scatter/histogram/violin plots."""
    if len(df) <= n_samples:
        return df.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), n_samples, replace=False)
    return df.iloc[idx].copy()


def compute_parent_offspring_corr(df: pd.DataFrame) -> dict[str, Any]:
    """Compute midparent-offspring liability correlations by generation.

    For each non-founder generation, finds offspring with both parents present,
    computes midparent liability = (mother + father) / 2, and returns
    Pearson r, regression slope/intercept, and n_pairs.
    """
    if "generation" not in df.columns:
        return {}

    max_gen = int(df["generation"].max())

    # Build id -> row lookup
    ids_arr = df["id"].values.astype(np.int64)
    id_to_row = np.full(ids_arr.max() + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)

    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].values
        trait_result: dict[str, dict[str, int | float | None]] = {}

        for gen in range(1, max_gen + 1):
            gen_idx = np.where(df["generation"].values == gen)[0]
            mother_ids = df["mother"].values[gen_idx].astype(np.int64)
            father_ids = df["father"].values[gen_idx].astype(np.int64)

            has_m = (mother_ids >= 0) & (mother_ids < len(id_to_row))
            has_f = (father_ids >= 0) & (father_ids < len(id_to_row))

            m_rows = np.full(len(gen_idx), -1, dtype=np.int32)
            f_rows = np.full(len(gen_idx), -1, dtype=np.int32)
            m_rows[has_m] = id_to_row[mother_ids[has_m]]
            f_rows[has_f] = id_to_row[father_ids[has_f]]

            valid = (m_rows >= 0) & (f_rows >= 0)
            n_pairs = int(valid.sum())

            if n_pairs < 10:
                trait_result[f"gen{gen}"] = {
                    "r": None, "slope": None, "intercept": None,
                    "n_pairs": n_pairs,
                }
                continue

            offspring_liab = liability[gen_idx[valid]]
            midparent_liab = (liability[m_rows[valid]] + liability[f_rows[valid]]) / 2.0

            r = float(np.corrcoef(midparent_liab, offspring_liab)[0, 1])
            reg = linregress(midparent_liab, offspring_liab)

            trait_result[f"gen{gen}"] = {
                "r": r,
                "slope": float(reg.slope),
                "intercept": float(reg.intercept),
                "n_pairs": n_pairs,
            }

        result[f"trait{trait_num}"] = trait_result

    return result


def compute_weibull_correlations(
    df: pd.DataFrame,
    weibull_params: dict[str, dict[str, float]],
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    n_quad: int = 20,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute pairwise Weibull liability correlations for all relationship types.

    Args:
        df: phenotype DataFrame
        weibull_params: {"trait1": {"scale": float, "rho": float, "beta": float},
                         "trait2": {...}}
        pairs: pre-extracted relationship pair indices
        n_quad: number of quadrature nodes per dimension
        seed: random seed for pair subsampling
    """
    from sim_ace.survival_corr import compute_weibull_pair_corr

    result: dict[str, Any] = {}
    result_uncensored: dict[str, Any] = {}
    for trait_num in [1, 2]:
        key = f"trait{trait_num}"
        params = weibull_params.get(key, {})
        if not params:
            result[key] = {}
            result_uncensored[key] = {}
            continue
        result[key] = compute_weibull_pair_corr(
            df, trait_num,
            scale=params["scale"], rho=params["rho"], beta=params["beta"],
            pairs=pairs, n_quad=n_quad, seed=seed,
        )
        result_uncensored[key] = compute_weibull_pair_corr(
            df, trait_num,
            scale=params["scale"], rho=params["rho"], beta=params["beta"],
            pairs=pairs, n_quad=n_quad, seed=seed, use_raw=True,
        )
    return result, result_uncensored


def main(
    phenotype_path: str,
    censor_age: float,
    stats_output: str,
    samples_output: str,
    seed: int = 42,
    gen_censoring: dict[int, list[float]] | None = None,
    weibull_params: dict[str, dict[str, float]] | None = None,
) -> None:
    """Compute all stats for a single rep and write outputs."""
    df = pd.read_parquet(phenotype_path)

    logger.info("Computing stats for %s (%d rows)", phenotype_path, len(df))

    stats: dict[str, Any] = {}
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
    if gen_censoring is not None:
        stats["censoring"] = compute_censoring_windows(
            df, censor_age, gen_censoring,
        )

    # Extract relationship pairs once (used by both liability and tetrachoric)
    logger.info("Extracting relationship pairs...")
    t_pairs = time.perf_counter()
    pairs = extract_relationship_pairs(df, seed=seed)
    logger.info(
        "Relationship pairs extracted in %.1fs: %s",
        time.perf_counter() - t_pairs,
        ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
    )

    # Liability correlations
    logger.info("Computing liability correlations...")
    stats["liability_correlations"] = compute_liability_correlations(df, seed=seed, pairs=pairs)

    # Parent-offspring correlations
    logger.info("Computing parent-offspring correlations...")
    stats["parent_offspring_corr"] = compute_parent_offspring_corr(df)

    # Tetrachoric correlations
    logger.info("Computing tetrachoric correlations...")
    t_tet = time.perf_counter()
    stats["tetrachoric"] = compute_tetrachoric(df, seed=seed, pairs=pairs)
    logger.info("Tetrachoric correlations computed in %.1fs", time.perf_counter() - t_tet)

    # Tetrachoric correlations by generation
    logger.info("Computing tetrachoric correlations by generation...")
    t_tet_gen = time.perf_counter()
    stats["tetrachoric_by_generation"] = compute_tetrachoric_by_generation(df, seed=seed, pairs=pairs)
    logger.info("Tetrachoric by generation computed in %.1fs", time.perf_counter() - t_tet_gen)

    # Pairwise Weibull correlations (if params supplied)
    if weibull_params is not None:
        logger.info("Computing pairwise Weibull correlations...")
        t_weib = time.perf_counter()
        censored, uncensored = compute_weibull_correlations(
            df, weibull_params, pairs=pairs, seed=seed,
        )
        stats["weibull_corr"] = censored
        stats["weibull_corr_uncensored"] = uncensored
        logger.info("Weibull correlations computed in %.1fs", time.perf_counter() - t_weib)

    # Write stats YAML
    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    logger.info("Stats written to %s", stats_path)

    # Write downsampled parquet
    sample_df = create_sample(df, seed=seed)
    samples_path = Path(samples_output)
    sample_df.to_parquet(samples_path, index=False)
    logger.info("Sample (%d rows) written to %s", len(sample_df), samples_path)


def cli() -> None:
    """Command-line interface for computing phenotype statistics."""
    from sim_ace.cli_base import add_logging_args, init_logging
    parser = argparse.ArgumentParser(description="Compute phenotype statistics")
    add_logging_args(parser)
    parser.add_argument("phenotype", help="Input phenotype parquet")
    parser.add_argument("censor_age", type=float, help="Censoring age")
    parser.add_argument("stats_output", help="Output stats YAML")
    parser.add_argument("samples_output", help="Output samples parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--gen-censoring", type=str, default=None, help="Per-generation censoring windows as JSON dict")
    parser.add_argument("--beta1", type=float, default=None, help="Weibull frailty coefficient for trait 1")
    parser.add_argument("--scale1", type=float, default=None, help="Weibull scale parameter for trait 1")
    parser.add_argument("--rho1", type=float, default=None, help="Weibull shape parameter for trait 1")
    parser.add_argument("--beta2", type=float, default=None, help="Weibull frailty coefficient for trait 2")
    parser.add_argument("--scale2", type=float, default=None, help="Weibull scale parameter for trait 2")
    parser.add_argument("--rho2", type=float, default=None, help="Weibull shape parameter for trait 2")
    args = parser.parse_args()

    init_logging(args)

    weibull_params = None
    if args.beta1 is not None and args.scale1 is not None and args.rho1 is not None:
        weibull_params = {
            "trait1": {"beta": args.beta1, "scale": args.scale1, "rho": args.rho1},
            "trait2": {"beta": args.beta2, "scale": args.scale2, "rho": args.rho2},
        }

    import json
    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    main(args.phenotype, args.censor_age, args.stats_output, args.samples_output,
         seed=args.seed, gen_censoring=gen_censoring,
         weibull_params=weibull_params)
