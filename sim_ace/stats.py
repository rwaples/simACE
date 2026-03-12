"""
Compute per-rep phenotype statistics for downstream plotting.

Reads a single phenotype.parquet and produces:
  - phenotype_stats.yaml: scalar and array statistics
  - phenotype_samples.parquet: downsampled rows for scatter/histogram plots
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import owens_t
from scipy.stats import linregress, norm

from sim_ace.pedigree_graph import extract_relationship_pairs
from sim_ace.utils import save_parquet, PAIR_TYPES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tetrachoric correlation
# ---------------------------------------------------------------------------

def _bvn_pos(h: float, k: float, r: float, sq: float) -> float:
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
    if abs(r) < 1e-15:
        return norm.cdf(h) * norm.cdf(k)
    sq = np.sqrt(1.0 - r * r)
    if h < 0 and k < 0:
        return 1.0 - norm.cdf(-h) - norm.cdf(-k) + _bvn_pos(-h, -k, r, sq)
    if h < 0:
        return norm.cdf(k) - _bvn_pos(-h, k, -r, sq)
    if k < 0:
        return norm.cdf(h) - _bvn_pos(h, -k, -r, sq)
    return _bvn_pos(h, k, r, sq)


def tetrachoric_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Convenience wrapper: returns only the MLE tetrachoric correlation.
    Kept for backward compatibility with older code that expects a scalar r.
    """
    r, _ = tetrachoric_corr_se(a, b)
    return r


def tetrachoric_corr_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Estimate tetrachoric correlation and SE from two binary arrays via MLE."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n_pairs = len(a)
    if n_pairs < 50:
        logger.warning("tetrachoric_corr_se: n_pairs=%d < 50, SE may be unreliable", n_pairs)

    n11 = np.sum(a & b)
    n10 = np.sum(a & ~b)
    n01 = np.sum(~a & b)
    n00 = np.sum(~a & ~b)

    p_a, p_b = a.mean(), b.mean()
    if p_a in (0, 1) or p_b in (0, 1):
        return np.nan, np.nan

    t_a = norm.ppf(1 - p_a)
    t_b = norm.ppf(1 - p_b)
    phi_ta = norm.cdf(t_a)
    phi_tb = norm.cdf(t_b)
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

    try:
        one_minus_r2 = 1.0 - r * r
        if one_minus_r2 <= 0:
            return r, np.nan
        bvn_pdf = (1.0 / (2.0 * np.pi * np.sqrt(one_minus_r2))) * np.exp(
            -(t_a**2 - 2.0 * r * t_a * t_b + t_b**2) / (2.0 * one_minus_r2)
        )
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
        se = np.nan if denom <= 0 else 1.0 / np.sqrt(n_pairs * bvn_pdf**2 / denom)
    except Exception:
        se = np.nan

    return r, se


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def compute_mortality(df: pd.DataFrame, censor_age: float) -> dict[str, Any]:
    decade_edges = np.arange(0, censor_age + 10, 10)
    mortality_rates, decade_labels = [], []
    death_ages = df["death_age"].values
    for i in range(len(decade_edges) - 1):
        lo, hi = decade_edges[i], decade_edges[i + 1]
        if lo >= censor_age:
            break
        alive = (death_ages >= lo).sum()
        died  = ((death_ages >= lo) & (death_ages < hi) & (death_ages < censor_age)).sum()
        mortality_rates.append(float(died / alive) if alive > 0 else 0.0)
        decade_labels.append(f"{int(lo)}-{int(hi - 1)}")
    return {"decade_labels": decade_labels, "rates": mortality_rates}


def compute_cumulative_incidence(
    df: pd.DataFrame, censor_age: float, n_points: int = 200,
) -> dict[str, Any]:
    ages = np.linspace(0, censor_age, n_points)
    n    = len(df)
    result = {}
    for trait_num in [1, 2]:
        aff      = df[f"affected{trait_num}"].values.astype(bool)
        t_obs    = df[f"t_observed{trait_num}"].values
        t_raw    = df[f"t{trait_num}"].values
        sorted_obs = np.sort(t_obs[aff])
        sorted_raw = np.sort(t_raw)
        obs_inc  = np.searchsorted(sorted_obs, ages, side="right") / n
        true_inc = np.searchsorted(sorted_raw, ages, side="right") / n
        half_idx = np.searchsorted(obs_inc, obs_inc[-1] / 2)
        result[f"trait{trait_num}"] = {
            "ages":            ages.tolist(),
            "observed_values": obs_inc.tolist(),
            "true_values":     true_inc.tolist(),
            "half_target_age": float(ages[min(half_idx, len(ages) - 1)]),
        }
    return result


def compute_censoring_windows(
    df: pd.DataFrame,
    censor_age: float,
    gen_censoring: dict[int, list[float]],
    n_points: int = 300,
) -> dict[str, Any] | None:
    if "generation" not in df.columns:
        return None
    ages   = np.linspace(0, censor_age, n_points)
    result = {}
    for gen in sorted(gen_censoring.keys()):
        win_lo, win_hi = gen_censoring[gen]
        gen_df = df[df["generation"] == gen]
        n_gen  = len(gen_df)
        if n_gen == 0:
            result[f"gen{gen}"] = {"n": 0}
            continue
        gen_result: dict[str, Any] = {"n": int(n_gen)}
        for trait_num in [1, 2]:
            t_raw    = gen_df[f"t{trait_num}"].values
            t_obs    = gen_df[f"t_observed{trait_num}"].values
            affected = gen_df[f"affected{trait_num}"].values.astype(bool)
            death_c  = gen_df[f"death_censored{trait_num}"].values.astype(bool)
            obs_inc  = np.searchsorted(np.sort(t_obs[affected]), ages, side="right") / n_gen
            true_inc = np.searchsorted(np.sort(t_raw),           ages, side="right") / n_gen
            gen_result[f"trait{trait_num}"] = {
                "true_incidence":     true_inc.tolist(),
                "observed_incidence": obs_inc.tolist(),
                "pct_affected":       float(affected.mean()),
                "left_censored":      float((t_raw < win_lo).sum() / n_gen),
                "right_censored":     float((t_raw > win_hi).sum() / n_gen),
                "death_censored":     float(
                    (death_c & ~(t_raw < win_lo) & ~(t_raw > win_hi)).mean()
                ),
            }
        result[f"gen{gen}"] = gen_result
    return {"generations": result, "censoring_ages": ages.tolist(), "censor_age": int(censor_age)}


def compute_regression(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        aff_col  = f"affected{trait_num}"
        t_col    = f"t_observed{trait_num}"
        liab_col = f"liability{trait_num}"
        if liab_col not in df.columns:
            result[f"trait{trait_num}"] = None
            continue
        sub = df[df[aff_col]].dropna(subset=[liab_col, t_col])
        if len(sub) < 2:
            result[f"trait{trait_num}"] = None
            continue
        reg = linregress(sub[liab_col].values, sub[t_col].values)
        result[f"trait{trait_num}"] = {
            "slope": float(reg.slope), "intercept": float(reg.intercept),
            "r2": float(reg.rvalue ** 2), "n": int(len(sub)),
        }
    return result


def compute_prevalence(df: pd.DataFrame) -> dict[str, float]:
    return {"trait1": float(df["affected1"].mean()), "trait2": float(df["affected2"].mean())}


# ---------------------------------------------------------------------------
# Correlation statistics
# ---------------------------------------------------------------------------

def compute_liability_correlations(
    df: pd.DataFrame, seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].values
        trait_result: dict[str, float | None] = {}
        for ptype in PAIR_TYPES:
            idx1, idx2 = pairs[ptype]
            trait_result[ptype] = (
                float(np.corrcoef(liability[idx1], liability[idx2])[0, 1])
                if len(idx1) >= 10 else None
            )
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_tetrachoric(
    df: pd.DataFrame, seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    result = {}
    for trait_num in [1, 2]:
        affected     = df[f"affected{trait_num}"].values.astype(bool)
        trait_result = {}
        for ptype in PAIR_TYPES:
            idx1, idx2 = pairs[ptype]
            n_p = len(idx1)
            if n_p < 10:
                trait_result[ptype] = {"r": None, "se": None, "n_pairs": int(n_p)}
                continue
            r, se = tetrachoric_corr_se(affected[idx1], affected[idx2])
            trait_result[ptype] = {
                "r":       float(r)  if not np.isnan(r)  else None,
                "se":      float(se) if not np.isnan(se) else None,
                "n_pairs": int(n_p),
            }
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_tetrachoric_by_generation(
    df: pd.DataFrame, seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if "generation" not in df.columns:
        return {}
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    gen_arr  = df["generation"].values
    max_gen  = int(gen_arr.max())
    plot_gens = list(range(max(1, max_gen - 2), max_gen + 1))
    result   = {}
    for gen in plot_gens:
        gen_result = {}
        for trait_num in [1, 2]:
            affected  = df[f"affected{trait_num}"].values.astype(bool)
            liability = df[f"liability{trait_num}"].values
            trait_result = {}
            for ptype in PAIR_TYPES:
                idx1, idx2 = pairs[ptype]
                mask   = gen_arr[idx1] == gen
                g_idx1 = idx1[mask]
                g_idx2 = idx2[mask]
                n_p    = len(g_idx1)
                if n_p < 10:
                    trait_result[ptype] = {
                        "r": None, "se": None, "n_pairs": int(n_p), "liability_r": None,
                    }
                    continue
                r, se      = tetrachoric_corr_se(affected[g_idx1], affected[g_idx2])
                liab_r     = float(np.corrcoef(liability[g_idx1], liability[g_idx2])[0, 1])
                trait_result[ptype] = {
                    "r":          float(r)      if not np.isnan(r)      else None,
                    "se":         float(se)     if not np.isnan(se)     else None,
                    "n_pairs":    int(n_p),
                    "liability_r": liab_r       if not np.isnan(liab_r) else None,
                }
            gen_result[f"trait{trait_num}"] = trait_result
        result[f"gen{gen}"] = gen_result
    return result


def compute_cross_trait_tetrachoric(
    df: pd.DataFrame, seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    a1 = df["affected1"].values.astype(bool)
    a2 = df["affected2"].values.astype(bool)
    r_sp, se_sp = tetrachoric_corr_se(a1, a2)
    result: dict[str, Any] = {
        "same_person": {
            "r":  float(r_sp)  if not np.isnan(r_sp)  else None,
            "se": float(se_sp) if not np.isnan(se_sp) else None,
            "n":  int(len(df)),
        }
    }
    by_gen: dict[str, Any] = {}
    if "generation" in df.columns:
        gen_arr   = df["generation"].values
        max_gen   = int(gen_arr.max())
        plot_gens = list(range(max(1, max_gen - 2), max_gen + 1))
        for gen in plot_gens:
            mask = gen_arr == gen
            n_g  = int(mask.sum())
            if n_g < 50:
                by_gen[f"gen{gen}"] = {"r": None, "se": None, "n": n_g}
                continue
            r_g, se_g = tetrachoric_corr_se(a1[mask], a2[mask])
            by_gen[f"gen{gen}"] = {
                "r":  float(r_g)  if not np.isnan(r_g)  else None,
                "se": float(se_g) if not np.isnan(se_g) else None,
                "n":  n_g,
            }
    result["same_person_by_generation"] = by_gen
    cross: dict[str, Any] = {}
    for ptype in PAIR_TYPES:
        idx1, idx2 = pairs[ptype]
        n_p = len(idx1)
        if n_p < 10:
            cross[ptype] = {"r": None, "se": None, "n_pairs": int(n_p)}
            continue
        r_cp, se_cp = tetrachoric_corr_se(a1[idx1], a2[idx2])
        cross[ptype] = {
            "r":       float(r_cp)  if not np.isnan(r_cp)  else None,
            "se":      float(se_cp) if not np.isnan(se_cp) else None,
            "n_pairs": int(n_p),
        }
    result["cross_person"] = cross
    return result


def compute_parent_offspring_corr(df: pd.DataFrame) -> dict[str, Any]:
    if "generation" not in df.columns:
        return {}
    max_gen    = int(df["generation"].max())
    ids_arr    = df["id"].values.astype(np.int64)
    id_to_row  = np.full(ids_arr.max() + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)
    result     = {}
    for trait_num in [1, 2]:
        liability    = df[f"liability{trait_num}"].values
        trait_result = {}
        for gen in range(1, max_gen + 1):
            gen_idx    = np.where(df["generation"].values == gen)[0]
            mother_ids = df["mother"].values[gen_idx].astype(np.int64)
            father_ids = df["father"].values[gen_idx].astype(np.int64)
            has_m = (mother_ids >= 0) & (mother_ids < len(id_to_row))
            has_f = (father_ids >= 0) & (father_ids < len(id_to_row))
            m_rows = np.full(len(gen_idx), -1, dtype=np.int32)
            f_rows = np.full(len(gen_idx), -1, dtype=np.int32)
            m_rows[has_m] = id_to_row[mother_ids[has_m]]
            f_rows[has_f] = id_to_row[father_ids[has_f]]
            valid   = (m_rows >= 0) & (f_rows >= 0)
            n_pairs = int(valid.sum())
            if n_pairs < 10:
                trait_result[f"gen{gen}"] = {
                    "r": None, "slope": None, "intercept": None, "n_pairs": n_pairs,
                }
                continue
            offspring  = liability[gen_idx[valid]]
            midparent  = (liability[m_rows[valid]] + liability[f_rows[valid]]) / 2.0
            r          = float(np.corrcoef(midparent, offspring)[0, 1])
            reg        = linregress(midparent, offspring)
            trait_result[f"gen{gen}"] = {
                "r": r, "slope": float(reg.slope),
                "intercept": float(reg.intercept), "n_pairs": n_pairs,
            }
        result[f"trait{trait_num}"] = trait_result
    return result


# ---------------------------------------------------------------------------
# Frailty (survival) correlations — pluggable baseline hazard
# ---------------------------------------------------------------------------

def compute_frailty_correlations(
    df: pd.DataFrame,
    frailty_params: dict[str, dict[str, Any]],
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    n_quad: int = 20,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute pairwise frailty liability correlations for all relationship types.

    Args:
        df: phenotype DataFrame
        frailty_params: {
            "trait1": {"beta": float, "hazard_model": str, "hazard_params": dict},
            "trait2": {...}
          }
        pairs: pre-extracted relationship pair indices
        n_quad: quadrature nodes per dimension
        seed: random seed for pair subsampling

    Returns:
        (censored, uncensored) dicts keyed by "trait1"/"trait2"
    """
    from sim_ace.survival_corr import compute_pair_corr

    result: dict[str, Any] = {}
    result_uncensored: dict[str, Any] = {}
    for trait_num in [1, 2]:
        key    = f"trait{trait_num}"
        params = frailty_params.get(key, {})
        if not params:
            result[key] = {}
            result_uncensored[key] = {}
            continue
        common = dict(
            df           = df,
            trait_num    = trait_num,
            beta         = params["beta"],
            pairs        = pairs,
            n_quad       = n_quad,
            seed         = seed,
            hazard_model = params["hazard_model"],
            hazard_params = params["hazard_params"],
        )
        result[key]            = compute_pair_corr(**common)
        result_uncensored[key] = compute_pair_corr(**common, use_raw=True)
    return result, result_uncensored


def compute_frailty_cross_trait_corr(
    df: pd.DataFrame,
    frailty_params: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute cross-trait frailty liability correlation.

    Each individual is self-paired across traits. Returns
    (censored, uncensored, stratified) dicts with keys {r, se, n}.
    The stratified estimate uses inverse-variance weighting over generations.
    """
    from sim_ace.survival_corr import cross_trait_corr_se

    p1 = frailty_params.get("trait1", {})
    p2 = frailty_params.get("trait2", {})
    empty = {"r": None, "se": None, "n": 0}
    if not p1 or not p2:
        return empty, empty, empty

    def _run(t1, d1, t2, d2):
        return cross_trait_corr_se(
            t1, d1, t2, d2,
            beta1          = p1["beta"],
            beta2          = p2["beta"],
            hazard_model_1 = p1["hazard_model"],
            hazard_params_1 = p1["hazard_params"],
            hazard_model_2 = p2["hazard_model"],
            hazard_params_2 = p2["hazard_params"],
        )

    n = len(df)
    ones = np.ones(n, dtype=np.float64)

    r_cens,   se_cens   = _run(
        df["t_observed1"].values.astype(np.float64),
        df["affected1"].values.astype(np.float64),
        df["t_observed2"].values.astype(np.float64),
        df["affected2"].values.astype(np.float64),
    )
    r_uncens, se_uncens = _run(
        df["t1"].values.astype(np.float64), ones,
        df["t2"].values.astype(np.float64), ones,
    )

    # Generation-stratified with inverse-variance weighting
    r_strat = se_strat = np.nan
    gen_details: dict[str, Any] = {}
    if "generation" in df.columns:
        gen_rs, gen_ses = [], []
        for gen in sorted(df["generation"].unique()):
            g    = df[df["generation"] == gen]
            r_g, se_g = _run(
                g["t_observed1"].values.astype(np.float64),
                g["affected1"].values.astype(np.float64),
                g["t_observed2"].values.astype(np.float64),
                g["affected2"].values.astype(np.float64),
            )
            gen_details[f"gen{gen}"] = {
                "r":  float(r_g)  if not np.isnan(r_g)  else None,
                "se": float(se_g) if not np.isnan(se_g) else None,
                "n":  int(len(g)),
            }
            if not np.isnan(r_g) and not np.isnan(se_g) and se_g > 0:
                gen_rs.append(r_g)
                gen_ses.append(se_g)
        if gen_rs:
            rs  = np.array(gen_rs)
            w   = 1.0 / np.array(gen_ses) ** 2
            r_strat  = float(np.sum(w * rs) / np.sum(w))
            se_strat = float(1.0 / np.sqrt(np.sum(w)))

    def _fmt(r, se):
        return {
            "r":  float(r)  if not np.isnan(r)  else None,
            "se": float(se) if not np.isnan(se) else None,
            "n":  n,
        }

    return (
        _fmt(r_cens,   se_cens),
        _fmt(r_uncens, se_uncens),
        {**_fmt(r_strat, se_strat), "per_generation": gen_details},
    )


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def create_sample(
    df: pd.DataFrame, seed: int = 42, n_per_gen: int = 50_000,
) -> pd.DataFrame:
    """Downsample for scatter/histogram plots, preserving parent rows."""
    rng        = np.random.default_rng(seed)
    generations = df["generation"].values
    unique_gens = sorted(np.unique(generations))
    if all(int((generations == g).sum()) <= n_per_gen for g in unique_gens):
        return df.copy()
    ids       = df["id"].values.astype(np.int64)
    max_id    = int(ids.max()) + 1
    id_to_row = np.full(max_id, -1, dtype=np.int32)
    id_to_row[ids] = np.arange(len(df), dtype=np.int32)
    selected  = set()
    for gen in unique_gens:
        gen_idx = np.where(generations == gen)[0]
        chosen  = rng.choice(gen_idx, min(len(gen_idx), n_per_gen), replace=False)
        selected.update(chosen.tolist())
    tmp = np.array(list(selected), dtype=np.int64)
    for pid_arr in [df["mother"].values[tmp].astype(np.int64),
                    df["father"].values[tmp].astype(np.int64)]:
        valid = (pid_arr >= 0) & (pid_arr < max_id)
        rows  = id_to_row[pid_arr[valid]]
        selected.update(rows[rows >= 0].tolist())
    return df.iloc[np.sort(np.array(list(selected), dtype=np.int64))].copy()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    phenotype_path: str,
    censor_age: float,
    stats_output: str,
    samples_output: str,
    seed: int = 42,
    gen_censoring: dict[int, list[float]] | None = None,
    frailty_params: dict[str, dict[str, Any]] | None = None,
    extra_tetrachoric: bool = True,
    pedigree_path: str | None = None,
) -> None:
    """Compute all stats for a single rep and write outputs."""
    df = pd.read_parquet(phenotype_path)
    logger.info("Computing stats for %s (%d rows)", phenotype_path, len(df))

    stats: dict[str, Any] = {
        "n_individuals": int(len(df)),
        "n_generations": int(df["generation"].nunique()) if "generation" in df.columns else 1,
    }

    stats["prevalence"]          = compute_prevalence(df)
    stats["mortality"]           = compute_mortality(df, censor_age)
    stats["regression"]          = compute_regression(df)
    stats["cumulative_incidence"] = compute_cumulative_incidence(df, censor_age)

    if gen_censoring is not None:
        stats["censoring"] = compute_censoring_windows(df, censor_age, gen_censoring)

    logger.info("Extracting relationship pairs...")
    t0    = time.perf_counter()
    pairs = extract_relationship_pairs(df, seed=seed)
    logger.info(
        "Relationship pairs extracted in %.1fs: %s",
        time.perf_counter() - t0,
        ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
    )

    stats["pair_counts"] = {k: int(len(v[0])) for k, v in pairs.items()}

    if pedigree_path is not None:
        logger.info("Extracting relationship pairs from full pedigree...")
        t1      = time.perf_counter()
        df_ped  = pd.read_parquet(pedigree_path)
        pairs_p = extract_relationship_pairs(df_ped, seed=seed)
        stats["pair_counts_ped"]    = {k: int(len(v[0])) for k, v in pairs_p.items()}
        stats["n_individuals_ped"]  = int(len(df_ped))
        stats["n_generations_ped"]  = (
            int(df_ped["generation"].nunique()) if "generation" in df_ped.columns else 1
        )
        del df_ped, pairs_p
        logger.info("Pedigree pairs extracted in %.1fs", time.perf_counter() - t1)

    logger.info("Computing liability correlations...")
    stats["liability_correlations"] = compute_liability_correlations(df, seed=seed, pairs=pairs)

    logger.info("Computing parent-offspring correlations...")
    stats["parent_offspring_corr"] = compute_parent_offspring_corr(df)

    logger.info("Computing tetrachoric correlations...")
    t2 = time.perf_counter()
    stats["tetrachoric"] = compute_tetrachoric(df, seed=seed, pairs=pairs)
    logger.info("Tetrachoric computed in %.1fs", time.perf_counter() - t2)

    logger.info("Computing tetrachoric correlations by generation...")
    t3 = time.perf_counter()
    stats["tetrachoric_by_generation"] = compute_tetrachoric_by_generation(df, seed=seed, pairs=pairs)
    logger.info("Tetrachoric by generation computed in %.1fs", time.perf_counter() - t3)

    logger.info("Computing cross-trait tetrachoric correlations...")
    t4 = time.perf_counter()
    stats["cross_trait_tetrachoric"] = compute_cross_trait_tetrachoric(df, seed=seed, pairs=pairs)
    logger.info("Cross-trait tetrachoric computed in %.1fs", time.perf_counter() - t4)

    if frailty_params is not None and extra_tetrachoric:
        logger.info("Computing pairwise frailty correlations...")
        t5 = time.perf_counter()
        censored, uncensored = compute_frailty_correlations(
            df, frailty_params, pairs=pairs, seed=seed,
        )
        stats["frailty_corr"]            = censored
        stats["frailty_corr_uncensored"] = uncensored
        logger.info("Frailty correlations computed in %.1fs", time.perf_counter() - t5)

        logger.info("Computing cross-trait frailty correlation...")
        t6 = time.perf_counter()
        ct_cens, ct_uncens, ct_strat = compute_frailty_cross_trait_corr(df, frailty_params)
        stats["frailty_cross_trait"]            = ct_cens
        stats["frailty_cross_trait_uncensored"] = ct_uncens
        stats["frailty_cross_trait_stratified"] = ct_strat
        logger.info("Cross-trait frailty correlation computed in %.1fs", time.perf_counter() - t6)
    elif frailty_params is not None:
        logger.info("Skipping frailty pairwise correlations (extra_tetrachoric=False)")

    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        yaml.dump(stats, fh, default_flow_style=False, sort_keys=False)
    logger.info("Stats written to %s", stats_path)

    sample_df = create_sample(df, seed=seed)
    save_parquet(sample_df, Path(samples_output))
    logger.info("Sample (%d rows) written to %s", len(sample_df), samples_output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    from sim_ace.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Compute phenotype statistics")
    add_logging_args(parser)
    parser.add_argument("phenotype",      help="Input phenotype parquet")
    parser.add_argument("censor_age",     type=float)
    parser.add_argument("stats_output",   help="Output stats YAML")
    parser.add_argument("samples_output", help="Output samples parquet")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--gen-censoring", type=str, default=None,
                        help="Per-generation censoring windows as JSON dict")
    parser.add_argument("--pedigree",     default=None,
                        help="Full pedigree parquet for G_ped pair counts")
    parser.add_argument("--no-extra-tetrachoric", dest="extra_tetrachoric",
                        action="store_false", default=True)

    # Trait 1 frailty params
    parser.add_argument("--beta1",             type=float, default=None)
    parser.add_argument("--phenotype-model1",  default=None)
    parser.add_argument("--phenotype-params1", type=str,   default=None,
                        help="JSON dict, e.g. '{\"scale\": 2160, \"rho\": 0.8}'")

    # Trait 2 frailty params
    parser.add_argument("--beta2",             type=float, default=None)
    parser.add_argument("--phenotype-model2",  default=None)
    parser.add_argument("--phenotype-params2", type=str,   default=None)

    args = parser.parse_args()
    init_logging(args)

    frailty_params = None
    if args.beta1 is not None and args.phenotype_model1 and args.phenotype_params1:
        frailty_params = {
            "trait1": {
                "beta":          args.beta1,
                "hazard_model":  args.phenotype_model1,
                "hazard_params": json.loads(args.phenotype_params1),
            },
            "trait2": {
                "beta":          args.beta2,
                "hazard_model":  args.phenotype_model2,
                "hazard_params": json.loads(args.phenotype_params2),
            },
        }

    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    main(
        args.phenotype, args.censor_age, args.stats_output, args.samples_output,
        seed=args.seed, gen_censoring=gen_censoring,
        frailty_params=frailty_params,
        extra_tetrachoric=args.extra_tetrachoric,
        pedigree_path=args.pedigree,
    )