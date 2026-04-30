"""Compute per-rep phenotype statistics for downstream plotting.

Reads a single phenotype.parquet and produces:
  - phenotype_stats.yaml: scalar and array statistics
  - phenotype_samples.parquet: downsampled rows for scatter/histogram plots
"""

from __future__ import annotations

__all__ = [
    "compute_affected_correlations",
    "compute_censoring_cascade",
    "compute_censoring_confusion",
    "compute_censoring_windows",
    "compute_cross_trait_tetrachoric",
    "compute_cumulative_incidence",
    "compute_cumulative_incidence_by_sex",
    "compute_cumulative_incidence_by_sex_generation",
    "compute_joint_affection",
    "compute_liability_correlations",
    "compute_mate_correlation",
    "compute_mean_family_size",
    "compute_mortality",
    "compute_observed_h2_estimators",
    "compute_parent_offspring_affected_corr",
    "compute_parent_offspring_corr",
    "compute_parent_offspring_corr_by_sex",
    "compute_parent_status",
    "compute_person_years",
    "compute_prevalence",
    "compute_regression",
    "compute_tetrachoric",
    "compute_tetrachoric_by_generation",
    "compute_tetrachoric_by_sex",
    "create_sample",
    "tetrachoric_corr",
    "tetrachoric_corr_se",
]

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from simace.core._numba_utils import _ndtri_approx, _norm_cdf, _pearsonr_core, _tetrachoric_core
from simace.core.numerics import fast_linregress
from simace.core.parquet import save_parquet
from simace.core.pedigree_graph import PedigreeGraph
from simace.core.relationships import PAIR_TYPES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tetrachoric correlation
# ---------------------------------------------------------------------------


def tetrachoric_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return the MLE tetrachoric correlation between two binary arrays.

    Args:
        a: First binary array.
        b: Second binary array, same length as *a*.

    Returns:
        Tetrachoric correlation coefficient.
    """
    r, _ = tetrachoric_corr_se(a, b)
    return r


def tetrachoric_corr_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Estimate tetrachoric correlation and SE from two binary arrays via MLE.

    Delegates the numerical work (Brent optimization + bivariate normal CDF)
    to the numba-jitted ``_tetrachoric_core`` for speed.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n_pairs = len(a)
    if n_pairs < 50:
        logger.warning("tetrachoric_corr_se: n_pairs=%d < 50, SE may be unreliable", n_pairs)

    n11 = float(np.sum(a & b))
    n10 = float(np.sum(a & ~b))
    n01 = float(np.sum(~a & b))
    n00 = float(np.sum(~a & ~b))

    p_a, p_b = a.mean(), b.mean()
    if p_a in (0, 1) or p_b in (0, 1):
        return np.nan, np.nan

    t_a = float(_ndtri_approx(1.0 - p_a))
    t_b = float(_ndtri_approx(1.0 - p_b))
    phi_ta = float(_norm_cdf(t_a))
    phi_tb = float(_norm_cdf(t_b))

    return _tetrachoric_core(n11, n10, n01, n00, t_a, t_b, phi_ta, phi_tb)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def compute_mortality(df: pd.DataFrame, censor_age: float) -> dict[str, Any]:
    """Compute decade-binned mortality rates from death ages.

    Args:
        df: Phenotype DataFrame with ``death_age`` column.
        censor_age: Maximum observation age.

    Returns:
        Dict with ``decade_labels`` and ``rates`` lists.
    """
    decade_edges = np.arange(0, censor_age + 10, 10)
    mortality_rates, decade_labels = [], []
    death_ages = df["death_age"].values
    for i in range(len(decade_edges) - 1):
        lo, hi = decade_edges[i], decade_edges[i + 1]
        if lo >= censor_age:
            break
        alive = (death_ages >= lo).sum()
        died = ((death_ages >= lo) & (death_ages < hi) & (death_ages < censor_age)).sum()
        mortality_rates.append(float(died / alive) if alive > 0 else 0.0)
        decade_labels.append(f"{int(lo)}-{int(hi - 1)}")
    return {"decade_labels": decade_labels, "rates": mortality_rates}


def compute_cumulative_incidence(
    df: pd.DataFrame,
    censor_age: float,
    n_points: int = 200,
) -> dict[str, Any]:
    """Compute observed and true cumulative incidence curves per trait.

    Args:
        df: Phenotype DataFrame with event time and affection columns.
        censor_age: Maximum observation age for the x-axis grid.
        n_points: Number of age grid points.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each with ``ages``,
        ``observed_values``, ``true_values``, and ``half_target_age``.
    """
    ages = np.linspace(0, censor_age, n_points)
    n = len(df)
    result = {}
    for trait_num in [1, 2]:
        aff = df[f"affected{trait_num}"].values.astype(bool)
        t_obs = df[f"t_observed{trait_num}"].values
        t_raw = df[f"t{trait_num}"].values
        sorted_obs = np.sort(t_obs[aff])
        sorted_raw = np.sort(t_raw)
        obs_inc = np.searchsorted(sorted_obs, ages, side="right") / n
        true_inc = np.searchsorted(sorted_raw, ages, side="right") / n
        half_idx = np.searchsorted(obs_inc, obs_inc[-1] / 2)
        result[f"trait{trait_num}"] = {
            "ages": ages.tolist(),
            "observed_values": obs_inc.tolist(),
            "true_values": true_inc.tolist(),
            "half_target_age": float(ages[min(half_idx, len(ages) - 1)]),
        }
    return result


def compute_censoring_windows(
    df: pd.DataFrame,
    censor_age: float,
    gen_censoring: dict[int, list[float]],
    n_points: int = 300,
) -> dict[str, Any] | None:
    """Compute per-generation censoring breakdown and incidence curves.

    Args:
        df: Phenotype DataFrame with generation, event time, and censoring columns.
        censor_age: Maximum observation age.
        gen_censoring: Dict mapping generation to ``[left, right]`` observation window.
        n_points: Number of age grid points for incidence curves.

    Returns:
        Dict with per-generation censoring stats, or None if no generation column.
    """
    if "generation" not in df.columns:
        return None
    ages = np.linspace(0, censor_age, n_points)
    result = {}
    for gen in sorted(gen_censoring.keys()):
        win_lo, win_hi = gen_censoring[gen]
        gen_df = df[df["generation"] == gen]
        n_gen = len(gen_df)
        if n_gen == 0:
            result[f"gen{gen}"] = {"n": 0}
            continue
        gen_result: dict[str, Any] = {"n": int(n_gen)}
        for trait_num in [1, 2]:
            t_raw = gen_df[f"t{trait_num}"].values
            t_obs = gen_df[f"t_observed{trait_num}"].values
            affected = gen_df[f"affected{trait_num}"].values.astype(bool)
            death_c = gen_df[f"death_censored{trait_num}"].values.astype(bool)
            obs_inc = np.searchsorted(np.sort(t_obs[affected]), ages, side="right") / n_gen
            true_inc = np.searchsorted(np.sort(t_raw), ages, side="right") / n_gen
            gen_result[f"trait{trait_num}"] = {
                "true_incidence": true_inc.tolist(),
                "observed_incidence": obs_inc.tolist(),
                "pct_affected": float(affected.mean()),
                "left_censored": float((t_raw < win_lo).sum() / n_gen),
                "right_censored": float((t_raw > win_hi).sum() / n_gen),
                "death_censored": float((death_c & ~(t_raw < win_lo) & ~(t_raw > win_hi)).mean()),
            }
        result[f"gen{gen}"] = gen_result
    return {"generations": result, "censoring_ages": ages.tolist(), "censor_age": int(censor_age)}


def compute_regression(df: pd.DataFrame) -> dict[str, Any]:
    """Regress observed event time on liability for affected individuals.

    Args:
        df: Phenotype DataFrame with liability and observed-time columns.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each with regression stats
        (slope, intercept, r, r2, stderr, pvalue, n) or None.
    """
    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        aff_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"
        liab_col = f"liability{trait_num}"
        if liab_col not in df.columns:
            result[f"trait{trait_num}"] = None
            continue
        sub = df[df[aff_col]].dropna(subset=[liab_col, t_col])
        if len(sub) < 2:
            result[f"trait{trait_num}"] = None
            continue
        slope, intercept, r, stderr, pvalue = fast_linregress(sub[liab_col].values, sub[t_col].values)
        result[f"trait{trait_num}"] = {
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "r2": r**2,
            "stderr": stderr,
            "pvalue": pvalue,
            "n": len(sub),
        }
    return result


def compute_prevalence(df: pd.DataFrame) -> dict[str, Any]:
    """Compute observed prevalence for each trait.

    Args:
        df: Phenotype DataFrame with ``affected1`` and ``affected2`` columns.
            If a ``generation`` column is present, per-generation prevalence
            is also reported under ``by_generation``.

    Returns:
        Dict with ``trait1`` and ``trait2`` marginal prevalence fractions, and
        (when ``generation`` is present) a ``by_generation`` subkey mapping
        ``int(generation) -> {"trait1": float, "trait2": float}``.
    """
    result: dict[str, Any] = {
        "trait1": float(df["affected1"].mean()),
        "trait2": float(df["affected2"].mean()),
    }
    if "generation" in df.columns:
        by_gen: dict[int, dict[str, float]] = {}
        for gen, grp in df.groupby("generation"):
            by_gen[int(gen)] = {
                "trait1": float(grp["affected1"].mean()),
                "trait2": float(grp["affected2"].mean()),
            }
        result["by_generation"] = by_gen
    return result


def compute_joint_affection(df: pd.DataFrame) -> dict[str, Any]:
    """Compute 2x2 contingency table for trait1 x trait2 affection status."""
    a1 = df["affected1"].values.astype(bool)
    a2 = df["affected2"].values.astype(bool)
    n = len(df)

    counts = {
        "both": int(np.sum(a1 & a2)),
        "trait1_only": int(np.sum(a1 & ~a2)),
        "trait2_only": int(np.sum(~a1 & a2)),
        "neither": int(np.sum(~a1 & ~a2)),
    }
    proportions = {k: v / n for k, v in counts.items()}

    # Sex-specific co-affection proportions
    by_sex: dict[str, float] = {}
    if "sex" in df.columns:
        for sex_val, sex_label in [(0, "female"), (1, "male")]:
            mask = df["sex"].values == sex_val
            n_sex = int(mask.sum())
            if n_sex > 0:
                by_sex[sex_label] = round(float(np.sum(a1[mask] & a2[mask])) / n_sex, 4)

    return {"counts": counts, "proportions": proportions, "n": n, "by_sex": by_sex}


def compute_censoring_confusion(
    df: pd.DataFrame,
    censor_age: float,
    gen_censoring: dict[int, list[float]],
) -> dict[str, Any]:
    """Compute per-trait 2x2 confusion matrix: true affected vs. observed affected.

    Only includes individuals from phenotyped generations (observation window > 0).
    """
    if "generation" in df.columns:
        active_gens = {int(g) for g, (lo, hi) in gen_censoring.items() if hi > lo}
        if active_gens:
            df = df[df["generation"].isin(active_gens)]

    result = {}
    for trait in [1, 2]:
        t_col = f"t{trait}"
        a_col = f"affected{trait}"
        if t_col not in df.columns or a_col not in df.columns:
            continue
        true_aff = df[t_col].values < censor_age
        obs_aff = df[a_col].values.astype(bool)
        n = len(df)
        result[f"trait{trait}"] = {
            "tp": int(np.sum(true_aff & obs_aff)),
            "fn": int(np.sum(true_aff & ~obs_aff)),
            "fp": int(np.sum(~true_aff & obs_aff)),
            "tn": int(np.sum(~true_aff & ~obs_aff)),
            "n": n,
        }
    return result


def compute_censoring_cascade(
    df: pd.DataFrame,
    censor_age: float,
    gen_censoring: dict[int, list[float]],
) -> dict[str, Any]:
    """Per-trait, per-generation decomposition of true cases by censoring fate."""
    if "generation" not in df.columns:
        return {}

    windows: dict[int, tuple[float, float]] = {}
    for g, (lo, hi) in gen_censoring.items():
        if hi > lo:
            windows[int(g)] = (lo, hi)

    if not windows:
        return {}

    has_death = "death_age" in df.columns
    result: dict[str, Any] = {}
    for trait in [1, 2]:
        t_col = f"t{trait}"
        a_col = f"affected{trait}"
        if t_col not in df.columns or a_col not in df.columns:
            continue
        trait_result: dict[str, Any] = {}
        for g in sorted(windows.keys()):
            lo, hi = windows[g]
            gen_mask = df["generation"] == g
            df_g = df.loc[gen_mask]
            t = df_g[t_col].values
            true_affected = t < censor_age
            n_true = int(true_affected.sum())
            n_gen = len(df_g)

            if n_true == 0:
                trait_result[f"gen{g}"] = {
                    "observed": 0,
                    "death_censored": 0,
                    "right_censored": 0,
                    "left_truncated": 0,
                    "true_affected": 0,
                    "n_gen": n_gen,
                    "sensitivity": float("nan"),
                    "window": [lo, hi],
                }
                continue

            left_trunc = true_affected & (t < lo)
            right_cens = true_affected & (t > hi)
            in_window = true_affected & (t >= lo) & (t <= hi)

            if has_death:
                death_age = df_g["death_age"].values
                death_cens = in_window & (death_age < t)
                observed = in_window & (death_age >= t)
            else:
                death_cens = np.zeros_like(in_window)
                observed = in_window

            n_obs = int(observed.sum())
            trait_result[f"gen{g}"] = {
                "observed": n_obs,
                "death_censored": int(death_cens.sum()),
                "right_censored": int(right_cens.sum()),
                "left_truncated": int(left_trunc.sum()),
                "true_affected": n_true,
                "n_gen": n_gen,
                "sensitivity": n_obs / n_true if n_true > 0 else float("nan"),
                "window": [lo, hi],
            }
        result[f"trait{trait}"] = trait_result
    return result


def compute_cumulative_incidence_by_sex(
    df: pd.DataFrame,
    censor_age: float,
    n_points: int = 200,
) -> dict[str, Any]:
    """Compute cumulative incidence curves split by sex (0=female, 1=male)."""
    if "sex" not in df.columns:
        return {}

    ages = np.linspace(0, censor_age, n_points)
    result = {}
    for trait_num in [1, 2]:
        aff = df[f"affected{trait_num}"].values.astype(bool)
        t_obs = df[f"t_observed{trait_num}"].values
        sex = df["sex"].values

        trait_result = {}
        for sex_val, sex_label in [(0, "female"), (1, "male")]:
            mask = sex == sex_val
            n_sex = int(mask.sum())
            if n_sex == 0:
                continue
            aff_sex = aff[mask]
            sorted_t = np.sort(t_obs[mask & aff])
            inc = np.searchsorted(sorted_t, ages, side="right") / n_sex
            prev = float(aff_sex.sum() / n_sex)
            trait_result[sex_label] = {
                "ages": ages.tolist(),
                "values": inc.tolist(),
                "n": n_sex,
                "prevalence": prev,
            }
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_cumulative_incidence_by_sex_generation(
    df: pd.DataFrame,
    censor_age: float,
    n_points: int = 200,
) -> dict[str, Any]:
    """Compute cumulative incidence curves split by sex and generation."""
    if "sex" not in df.columns or "generation" not in df.columns:
        return {}

    ages = np.linspace(0, censor_age, n_points)
    generations = sorted(df["generation"].unique())
    result = {}
    for trait_num in [1, 2]:
        aff = df[f"affected{trait_num}"].values.astype(bool)
        t_obs = df[f"t_observed{trait_num}"].values
        sex = df["sex"].values
        gen_arr = df["generation"].values

        trait_result: dict[str, Any] = {}
        for gen in generations:
            gen_result: dict[str, Any] = {}
            gen_mask = gen_arr == gen
            for sex_val, sex_label in [(0, "female"), (1, "male")]:
                mask = gen_mask & (sex == sex_val)
                n_sex = int(mask.sum())
                if n_sex == 0:
                    continue
                aff_sex = aff[mask]
                sorted_t = np.sort(t_obs[mask & aff])
                inc = np.searchsorted(sorted_t, ages, side="right") / n_sex
                prev = float(aff_sex.sum() / n_sex)
                gen_result[sex_label] = {
                    "ages": ages.tolist(),
                    "values": inc.tolist(),
                    "n": n_sex,
                    "prevalence": prev,
                }
            trait_result[f"gen{int(gen)}"] = gen_result
        result[f"trait{trait_num}"] = trait_result
    return result


# ---------------------------------------------------------------------------
# Correlation statistics
# ---------------------------------------------------------------------------


def compute_liability_correlations(
    df: pd.DataFrame,
    seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compute Pearson liability correlations per pair type and trait.

    Args:
        df: Phenotype DataFrame with liability columns.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs; extracted if None.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to correlation.
    """
    if pairs is None:
        pairs = PedigreeGraph(df).extract_pairs()
    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].values
        trait_result: dict[str, float | None] = {}
        for ptype in PAIR_TYPES:
            idx1, idx2 = pairs[ptype]
            trait_result[ptype] = float(_pearsonr_core(liability[idx1], liability[idx2])) if len(idx1) >= 10 else None
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_affected_correlations(
    df: pd.DataFrame,
    seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compute Pearson correlations on binary affected status per pair type and trait.

    This is the phi coefficient — Pearson r on {0, 1} data — and is the input
    to observed-scale Falconer-style h² estimators (e.g. ``2·(r_MZ − r_FS)``).

    Args:
        df: Phenotype DataFrame with ``affected{1,2}`` columns.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs; extracted if None.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to phi r or
        None (if fewer than 10 pairs, or either side is constant).
    """
    if pairs is None:
        pairs = PedigreeGraph(df).extract_pairs()
    result = {}
    for trait_num in [1, 2]:
        affected = df[f"affected{trait_num}"].values.astype(np.float64)
        trait_result: dict[str, float | None] = {}
        for ptype in PAIR_TYPES:
            idx1, idx2 = pairs[ptype]
            if len(idx1) < 10:
                trait_result[ptype] = None
                continue
            a1 = affected[idx1]
            a2 = affected[idx2]
            # Phi r is undefined when either indicator is constant.
            if a1.std() < 1e-12 or a2.std() < 1e-12:
                trait_result[ptype] = None
                continue
            trait_result[ptype] = float(_pearsonr_core(a1, a2))
        result[f"trait{trait_num}"] = trait_result
    return result


def _tetrachoric_for_pairs(
    idx1: np.ndarray,
    idx2: np.ndarray,
    affected: np.ndarray,
    liability: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute tetrachoric r, SE, and optionally liability r for one pair subset."""
    n_p = len(idx1)
    if n_p < 10:
        entry: dict[str, Any] = {"r": None, "se": None, "n_pairs": int(n_p)}
        if liability is not None:
            entry["liability_r"] = None
        return entry
    r, se = tetrachoric_corr_se(affected[idx1], affected[idx2])
    entry = {
        "r": float(r) if not np.isnan(r) else None,
        "se": float(se) if not np.isnan(se) else None,
        "n_pairs": int(n_p),
    }
    if liability is not None:
        liab_r = float(_pearsonr_core(liability[idx1], liability[idx2]))
        entry["liability_r"] = liab_r if not np.isnan(liab_r) else None
    return entry


def compute_tetrachoric(
    df: pd.DataFrame,
    seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compute tetrachoric correlations per pair type and trait.

    Args:
        df: Phenotype DataFrame with binary affection columns.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs; extracted if None.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to
        ``{r, se, n_pairs}``.
    """
    if pairs is None:
        pairs = PedigreeGraph(df).extract_pairs()
    result = {}
    for trait_num in [1, 2]:
        affected = df[f"affected{trait_num}"].values.astype(bool)
        trait_result = {}
        for ptype in PAIR_TYPES:
            idx1, idx2 = pairs[ptype]
            trait_result[ptype] = _tetrachoric_for_pairs(idx1, idx2, affected)
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_tetrachoric_by_generation(
    df: pd.DataFrame,
    seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compute tetrachoric correlations stratified by generation.

    Args:
        df: Phenotype DataFrame with generation and affection columns.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs; extracted if None.

    Returns:
        Dict keyed by ``gen{N}``, each containing per-trait per-pair-type
        ``{r, se, n_pairs, liability_r}``.
    """
    if "generation" not in df.columns:
        return {}
    if pairs is None:
        pairs = PedigreeGraph(df).extract_pairs()
    gen_arr = df["generation"].values
    max_gen = int(gen_arr.max())
    plot_gens = list(range(max(1, max_gen - 2), max_gen + 1))
    result = {}
    for gen in plot_gens:
        gen_result = {}
        for trait_num in [1, 2]:
            affected = df[f"affected{trait_num}"].values.astype(bool)
            liability = df[f"liability{trait_num}"].values
            trait_result = {}
            for ptype in PAIR_TYPES:
                idx1, idx2 = pairs[ptype]
                mask = gen_arr[idx1] == gen
                trait_result[ptype] = _tetrachoric_for_pairs(idx1[mask], idx2[mask], affected, liability)
            gen_result[f"trait{trait_num}"] = trait_result
        result[f"gen{gen}"] = gen_result
    return result


def compute_cross_trait_tetrachoric(
    df: pd.DataFrame,
    seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compute cross-trait tetrachoric correlations (trait 1 vs trait 2).

    Includes same-person, same-person-by-generation, and cross-person
    (across relationship pair types) correlations.

    Args:
        df: Phenotype DataFrame with binary affection columns for both traits.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs; extracted if None.

    Returns:
        Dict with keys ``same_person``, ``same_person_by_generation``,
        and ``cross_person``.
    """
    if pairs is None:
        pairs = PedigreeGraph(df).extract_pairs()
    a1 = df["affected1"].values.astype(bool)
    a2 = df["affected2"].values.astype(bool)
    r_sp, se_sp = tetrachoric_corr_se(a1, a2)
    result: dict[str, Any] = {
        "same_person": {
            "r": float(r_sp) if not np.isnan(r_sp) else None,
            "se": float(se_sp) if not np.isnan(se_sp) else None,
            "n": len(df),
        }
    }
    by_gen: dict[str, Any] = {}
    if "generation" in df.columns:
        gen_arr = df["generation"].values
        max_gen = int(gen_arr.max())
        plot_gens = list(range(max(1, max_gen - 2), max_gen + 1))
        for gen in plot_gens:
            mask = gen_arr == gen
            n_g = int(mask.sum())
            if n_g < 50:
                by_gen[f"gen{gen}"] = {"r": None, "se": None, "n": n_g}
                continue
            r_g, se_g = tetrachoric_corr_se(a1[mask], a2[mask])
            by_gen[f"gen{gen}"] = {
                "r": float(r_g) if not np.isnan(r_g) else None,
                "se": float(se_g) if not np.isnan(se_g) else None,
                "n": n_g,
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
            "r": float(r_cp) if not np.isnan(r_cp) else None,
            "se": float(se_cp) if not np.isnan(se_cp) else None,
            "n_pairs": int(n_p),
        }
    result["cross_person"] = cross
    return result


def _po_regression(gen_idx: np.ndarray, liability: np.ndarray, id_to_row: np.ndarray, df: pd.DataFrame) -> dict:
    """Midparent-offspring regression for a given set of offspring indices."""
    mother_ids = df["mother"].values[gen_idx]
    father_ids = df["father"].values[gen_idx]
    has_m = (mother_ids >= 0) & (mother_ids < len(id_to_row))
    has_f = (father_ids >= 0) & (father_ids < len(id_to_row))
    m_rows = np.full(len(gen_idx), -1, dtype=np.int32)
    f_rows = np.full(len(gen_idx), -1, dtype=np.int32)
    m_rows[has_m] = id_to_row[mother_ids[has_m]]
    f_rows[has_f] = id_to_row[father_ids[has_f]]
    valid = (m_rows >= 0) & (f_rows >= 0)
    n_pairs = int(valid.sum())
    null = {"r": None, "r2": None, "slope": None, "intercept": None, "stderr": None, "pvalue": None, "n_pairs": n_pairs}
    if n_pairs < 10:
        return null
    offspring = liability[gen_idx[valid]]
    midparent = (liability[m_rows[valid]] + liability[f_rows[valid]]) / 2.0
    slope, intercept, r, stderr, pvalue = fast_linregress(midparent, offspring)
    return {
        "r": r,
        "r2": r**2,
        "slope": slope,
        "intercept": intercept,
        "stderr": stderr,
        "pvalue": pvalue,
        "n_pairs": n_pairs,
    }


def compute_parent_offspring_corr(df: pd.DataFrame) -> dict[str, Any]:
    """Compute midparent-offspring liability regression per generation and trait.

    Args:
        df: Phenotype DataFrame with liability, generation, and parent columns.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each containing per-generation
        regression stats (slope, r, r2, intercept, stderr, pvalue, n_pairs).
    """
    if "generation" not in df.columns:
        return {}
    max_gen = int(df["generation"].max())
    ids_arr = df["id"].values
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)
    gen_arr = df["generation"].values
    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].values
        trait_result = {}
        for gen in range(1, max_gen + 1):
            gen_idx = np.where(gen_arr == gen)[0]
            trait_result[f"gen{gen}"] = _po_regression(gen_idx, liability, id_to_row, df)
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_parent_offspring_affected_corr(df: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled midparent-offspring regression on binary affected status.

    Regresses ``offspring.affected`` (0/1) on midparent affected status
    ``(mother.affected + father.affected) / 2`` (values in {0, 0.5, 1}),
    pooled across every non-founder individual whose parents are both in the
    DataFrame.  The regression slope is the observed-scale PO heritability
    estimator; under LTM it can be back-transformed to liability via
    Dempster-Lerner.

    Args:
        df: Phenotype DataFrame with ``id``, ``mother``, ``father``, and
            ``affected{1,2}`` columns.

    Returns:
        Dict keyed ``trait1``/``trait2``, each with
        ``{slope, r, r2, intercept, stderr, pvalue, n_pairs}``.  Values are
        None when fewer than 10 valid trios or midparent has zero variance.
    """
    if "id" not in df.columns or "mother" not in df.columns or "father" not in df.columns:
        return {}

    ids_arr = df["id"].values
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)

    non_founder_idx = np.where(df["mother"].values >= 0)[0]

    null = {
        "r": None,
        "r2": None,
        "slope": None,
        "intercept": None,
        "stderr": None,
        "pvalue": None,
        "n_pairs": 0,
    }

    # Precompute valid trios once (parents present, looked up via id_to_row).
    mother_ids_arr = df["mother"].values[non_founder_idx]
    father_ids_arr = df["father"].values[non_founder_idx]
    has_m = (mother_ids_arr >= 0) & (mother_ids_arr < len(id_to_row))
    has_f = (father_ids_arr >= 0) & (father_ids_arr < len(id_to_row))
    m_rows = np.full(len(non_founder_idx), -1, dtype=np.int32)
    f_rows = np.full(len(non_founder_idx), -1, dtype=np.int32)
    m_rows[has_m] = id_to_row[mother_ids_arr[has_m]]
    f_rows[has_f] = id_to_row[father_ids_arr[has_f]]
    valid = (m_rows >= 0) & (f_rows >= 0)
    n_pairs = int(valid.sum())

    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        aff_col = f"affected{trait_num}"
        if aff_col not in df.columns:
            result[f"trait{trait_num}"] = {**null}
            continue
        affected = df[aff_col].values.astype(np.float64)
        if n_pairs < 10:
            result[f"trait{trait_num}"] = {**null, "n_pairs": n_pairs}
            continue
        midparent = (affected[m_rows[valid]] + affected[f_rows[valid]]) / 2.0
        # Zero-variance midparent (all parents concordant) gives an undefined
        # regression; surface as None rather than 0/0.
        if float(np.var(midparent)) < 1e-12:
            result[f"trait{trait_num}"] = {**null, "n_pairs": n_pairs}
            continue
        entry = _po_regression(non_founder_idx, affected, id_to_row, df)
        slope = entry.get("slope")
        if slope is not None and not np.isfinite(slope):
            entry = {**null, "n_pairs": entry.get("n_pairs", 0)}
        result[f"trait{trait_num}"] = entry
    return result


def compute_observed_h2_estimators(stats: dict[str, Any]) -> dict[str, Any]:
    """Derive five naive observed-scale h² estimators from precomputed correlations.

    Reads from ``stats["affected_correlations"]`` (phi r per pair type) and
    ``stats["parent_offspring_affected_corr"]`` (PO regression slope on binary).
    Each estimator is a closed-form combination that, under a liability-threshold
    model, is an unbiased estimator of ``h²_liab · z(K)²/(K(1−K))`` — i.e. the
    observed-scale h² — where K is the affected-status prevalence.

    Args:
        stats: The in-progress stats dict with ``affected_correlations`` and
            ``parent_offspring_affected_corr`` already populated.

    Returns:
        Dict keyed ``trait1``/``trait2``, each mapping estimator name to a
        float or None: ``{falconer, sibs, po, hs, cousins}``.
    """
    aff = stats.get("affected_correlations", {}) or {}
    po_all = stats.get("parent_offspring_affected_corr", {}) or {}

    def _two_diff(r_a: Any, r_b: Any) -> float | None:
        if r_a is None or r_b is None:
            return None
        return 2.0 * (float(r_a) - float(r_b))

    def _scale(r: Any, factor: float) -> float | None:
        if r is None:
            return None
        return factor * float(r)

    def _mean_hs(r_mhs: Any, r_phs: Any) -> float | None:
        vals = [float(v) for v in (r_mhs, r_phs) if v is not None]
        if not vals:
            return None
        return 4.0 * (sum(vals) / len(vals))

    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        key = f"trait{trait_num}"
        rs = aff.get(key, {}) or {}
        po_entry = po_all.get(key, {}) or {}
        po_slope = po_entry.get("slope")
        result[key] = {
            "falconer": _two_diff(rs.get("MZ"), rs.get("FS")),
            "sibs": _scale(rs.get("FS"), 2.0),
            "po": float(po_slope) if po_slope is not None else None,
            "hs": _mean_hs(rs.get("MHS"), rs.get("PHS")),
            "cousins": _scale(rs.get("1C"), 8.0),
        }
    return result


def compute_tetrachoric_by_sex(
    df: pd.DataFrame,
    seed: int = 42,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compute tetrachoric correlations for same-sex pairs only (FF and MM).

    Returns dict keyed by "female"/"male", each containing per-trait
    per-pair-type {r, se, n_pairs, liability_r}.
    """
    if pairs is None:
        pairs = PedigreeGraph(df).extract_pairs()
    sex_arr = df["sex"].values
    result: dict[str, Any] = {}
    for sex_val, sex_label in [(0, "female"), (1, "male")]:
        sex_result: dict[str, Any] = {}
        for trait_num in [1, 2]:
            affected = df[f"affected{trait_num}"].values.astype(bool)
            liability = df[f"liability{trait_num}"].values
            trait_result: dict[str, Any] = {}
            for ptype in PAIR_TYPES:
                idx1, idx2 = pairs[ptype]
                sex_mask = (sex_arr[idx1] == sex_val) & (sex_arr[idx2] == sex_val)
                trait_result[ptype] = _tetrachoric_for_pairs(idx1[sex_mask], idx2[sex_mask], affected, liability)
            sex_result[f"trait{trait_num}"] = trait_result
        result[sex_label] = sex_result
    return result


def compute_parent_offspring_corr_by_sex(df: pd.DataFrame) -> dict[str, Any]:
    """Compute midparent-offspring regression partitioned by offspring sex.

    Returns dict keyed by "female"/"male", each containing per-trait
    per-generation {slope, r, r2, intercept, stderr, pvalue, n_pairs}.
    """
    if "generation" not in df.columns:
        return {}
    max_gen = int(df["generation"].max())
    ids_arr = df["id"].values
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)
    gen_arr = df["generation"].values
    sex_arr = df["sex"].values
    result: dict[str, Any] = {}
    for sex_val, sex_label in [(0, "female"), (1, "male")]:
        sex_result: dict[str, Any] = {}
        for trait_num in [1, 2]:
            liability = df[f"liability{trait_num}"].values
            trait_result: dict[str, Any] = {}
            for gen in range(1, max_gen + 1):
                gen_idx = np.where((gen_arr == gen) & (sex_arr == sex_val))[0]
                trait_result[f"gen{gen}"] = _po_regression(gen_idx, liability, id_to_row, df)
            sex_result[f"trait{trait_num}"] = trait_result
        result[sex_label] = sex_result
    return result


# ---------------------------------------------------------------------------
# Mate correlation
# ---------------------------------------------------------------------------


def compute_mate_correlation(df: pd.DataFrame) -> dict:
    """Compute 2x2 Pearson correlation matrix between mated pairs' liabilities.

    Each unique (mother, father) pair is counted once (not weighted by offspring).
    Only non-founders are considered.
    """
    from simace.core.numerics import safe_corrcoef

    nf = df[df["mother"] != -1][["mother", "father"]].drop_duplicates()
    if len(nf) < 2:
        return {"matrix": [[float("nan")] * 2] * 2, "n_pairs": 0}

    lookup = df.set_index("id")[["liability1", "liability2"]]
    f_liab = lookup.loc[nf["mother"].values].values  # (N, 2)
    m_liab = lookup.loc[nf["father"].values].values  # (N, 2)

    matrix = [[float(safe_corrcoef(f_liab[:, i], m_liab[:, j])) for j in range(2)] for i in range(2)]
    return {"matrix": matrix, "n_pairs": len(nf)}


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def create_sample(
    df: pd.DataFrame,
    seed: int = 42,
    n_per_gen: int = 50_000,
) -> pd.DataFrame:
    """Downsample for scatter/histogram plots, preserving parent rows."""
    rng = np.random.default_rng(seed)
    generations = df["generation"].values
    unique_gens = sorted(np.unique(generations))
    if all(int((generations == g).sum()) <= n_per_gen for g in unique_gens):
        return df.copy()
    ids = df["id"].values
    max_id = int(ids.max()) + 1
    id_to_row = np.full(max_id, -1, dtype=np.int32)
    id_to_row[ids] = np.arange(len(df), dtype=np.int32)
    selected = set()
    for gen in unique_gens:
        gen_idx = np.where(generations == gen)[0]
        chosen = rng.choice(gen_idx, min(len(gen_idx), n_per_gen), replace=False)
        selected.update(chosen.tolist())
    tmp = np.array(list(selected), dtype=np.intp)
    for pid_arr in [df["mother"].values[tmp], df["father"].values[tmp]]:
        valid = (pid_arr >= 0) & (pid_arr < max_id)
        rows = id_to_row[pid_arr[valid]]
        selected.update(rows[rows >= 0].tolist())
    return df.iloc[np.sort(np.array(list(selected), dtype=np.intp))].copy()


# ---------------------------------------------------------------------------
# Person-years and family size
# ---------------------------------------------------------------------------


def compute_person_years(
    df: pd.DataFrame,
    censor_age: float,
    gen_censoring: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    """Compute person-years of follow-up, total and per-trait at-risk.

    For each individual in generation *g* with observation window [lo, hi]:
      - Total follow-up ends at min(death_age, hi).
      - Trait-specific at-risk time ends at min(t_observed, death_age, hi).

    Returns dict with total_person_years and per-trait person_years_at_risk.
    """
    if "generation" not in df.columns:
        return {}

    # Build window lookup: generation → (lo, hi)
    windows: dict[int, tuple[float, float]] = {}
    if gen_censoring is not None:
        for g, (lo, hi) in gen_censoring.items():
            windows[int(g)] = (float(lo), float(hi))

    has_death = "death_age" in df.columns
    total_py = 0.0
    total_deaths = 0
    trait_py: dict[str, float] = {}

    for trait in [1, 2]:
        trait_py[f"trait{trait}"] = 0.0

    for g, df_g in df.groupby("generation"):
        lo, hi = windows.get(int(g), (0.0, censor_age))
        if hi <= lo:
            continue
        n = len(df_g)

        # End of observation for each person (not trait-specific)
        if has_death:
            death_ages = df_g["death_age"].values
            end = np.minimum(death_ages, hi)
            # Deaths observed during follow-up window
            total_deaths += int(((death_ages >= lo) & (death_ages < hi)).sum())
        else:
            end = np.full(n, hi)
        gen_total = np.clip(end - lo, 0, None).sum()
        total_py += float(gen_total)

        # Trait-specific at-risk person-years
        for trait in [1, 2]:
            t_col = f"t_observed{trait}"
            if t_col not in df_g.columns:
                continue
            t_obs = df_g[t_col].values
            if has_death:
                trait_end = np.minimum(np.minimum(t_obs, df_g["death_age"].values), hi)
            else:
                trait_end = np.minimum(t_obs, hi)
            trait_py[f"trait{trait}"] += float(np.clip(trait_end - lo, 0, None).sum())

    return {
        "total": round(total_py, 1),
        "deaths": total_deaths,
        **{k: round(v, 1) for k, v in trait_py.items()},
    }


def compute_mean_family_size(df: pd.DataFrame) -> dict[str, Any]:
    """Compute mean realised family size (offspring per mating pair).

    Uses non-founder individuals (mother != -1) grouped by (mother, father).
    """
    if "mother" not in df.columns or "father" not in df.columns:
        return {}

    children = df.loc[(df["mother"] != -1) & (df["father"] != -1)]
    if len(children) == 0:
        return {}

    family_sizes = children.groupby(["mother", "father"]).size()

    # Fraction with at least one phenotyped full sibling
    families_with_sibs = family_sizes[family_sizes >= 2].index
    has_sib = children.set_index(["mother", "father"]).index.isin(families_with_sibs)
    frac_with_full_sib = round(float(has_sib.sum()) / len(children), 4)

    # Family size distribution per mating (1, 2, 3, 4+)
    n_fam = len(family_sizes)
    dist: dict[str, float] = {}
    for k in [1, 2, 3]:
        dist[str(k)] = round(int((family_sizes == k).sum()) / n_fam, 4)
    dist["4+"] = round(int((family_sizes >= 4).sum()) / n_fam, 4)

    # Offspring per person (including 0 for childless individuals)
    parent_ids = set(children["mother"].values) | set(children["father"].values)
    parent_ids.discard(-1)
    # Count offspring per person (as parent)
    n_as_mother = children.groupby("mother").size()
    n_as_father = children.groupby("father").size()
    offspring_counts = pd.Series(0, index=df["id"])
    offspring_counts.update(n_as_mother.rename_axis("id"))
    # Add father counts (some people may be both, though unlikely)
    father_counts = n_as_father.rename_axis("id")
    offspring_counts = offspring_counts.add(father_counts, fill_value=0).astype(int)
    n_total = len(df)
    person_dist: dict[str, float] = {}
    person_dist["0"] = round(int((offspring_counts == 0).sum()) / n_total, 4)
    for k in [1, 2, 3]:
        person_dist[str(k)] = round(int((offspring_counts == k).sum()) / n_total, 4)
    person_dist["4+"] = round(int((offspring_counts >= 4).sum()) / n_total, 4)

    # Offspring per person by sex
    person_dist_by_sex: dict[str, dict[str, float]] = {}
    if "sex" in df.columns:
        sex_by_id = df.set_index("id")["sex"]
        for sex_label, sex_val in [("female", 0), ("male", 1)]:
            sex_ids = sex_by_id[sex_by_id == sex_val].index
            sex_counts = offspring_counts.reindex(sex_ids, fill_value=0)
            n_sex = len(sex_counts)
            if n_sex > 0:
                sd: dict[str, float] = {}
                sd["0"] = round(int((sex_counts == 0).sum()) / n_sex, 4)
                for k in [1, 2, 3]:
                    sd[str(k)] = round(int((sex_counts == k).sum()) / n_sex, 4)
                sd["4+"] = round(int((sex_counts >= 4).sum()) / n_sex, 4)
                person_dist_by_sex[sex_label] = sd

    # Number of mates by sex
    # Females: unique fathers per mother
    mates_female = children.groupby("mother")["father"].nunique()
    # Males: unique mothers per father
    mates_male = children.groupby("father")["mother"].nunique()
    n_mothers = len(mates_female)
    n_fathers = len(mates_male)
    mates_by_sex: dict[str, Any] = {
        "female_mean": round(float(mates_female.mean()), 2) if n_mothers else 0,
        "male_mean": round(float(mates_male.mean()), 2) if n_fathers else 0,
        "female_1": round(int((mates_female == 1).sum()) / n_mothers, 4) if n_mothers else 0,
        "female_2+": round(int((mates_female >= 2).sum()) / n_mothers, 4) if n_mothers else 0,
        "male_1": round(int((mates_male == 1).sum()) / n_fathers, 4) if n_fathers else 0,
        "male_2+": round(int((mates_male >= 2).sum()) / n_fathers, 4) if n_fathers else 0,
    }

    return {
        "mean": round(float(family_sizes.mean()), 2),
        "median": round(float(family_sizes.median()), 1),
        "q1": round(float(family_sizes.quantile(0.25)), 1),
        "q3": round(float(family_sizes.quantile(0.75)), 1),
        "n_families": len(family_sizes),
        "frac_with_full_sib": frac_with_full_sib,
        "size_dist": dist,
        "person_offspring_dist": person_dist,
        "person_offspring_dist_by_sex": person_dist_by_sex,
        "mates_by_sex": mates_by_sex,
    }


def compute_parent_status(
    df: pd.DataFrame,
    df_ped: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Count individuals by number of parents phenotyped and in pedigree.

    Returns dict with 'phenotyped' and optionally 'in_pedigree', each mapping
    0/1/2 → count of individuals with that many parents present.
    """
    if "mother" not in df.columns or "father" not in df.columns:
        return {}

    pheno_ids = set(df["id"].values)
    mothers = df["mother"].values
    fathers = df["father"].values

    # Parents phenotyped
    m_pheno = np.isin(mothers, list(pheno_ids)) & (mothers != -1)
    f_pheno = np.isin(fathers, list(pheno_ids)) & (fathers != -1)
    n_parents_pheno = m_pheno.astype(int) + f_pheno.astype(int)
    result: dict[str, Any] = {
        "phenotyped": {str(k): int((n_parents_pheno == k).sum()) for k in [0, 1, 2]},
    }

    # Parents in pedigree
    if df_ped is not None:
        ped_ids = set(df_ped["id"].values)
        m_ped = np.isin(mothers, list(ped_ids)) & (mothers != -1)
        f_ped = np.isin(fathers, list(ped_ids)) & (fathers != -1)
        n_parents_ped = m_ped.astype(int) + f_ped.astype(int)
        result["in_pedigree"] = {str(k): int((n_parents_ped == k).sum()) for k in [0, 1, 2]}

    return result


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
    pedigree_path: str | None = None,
    max_degree: int = 2,
    case_ascertainment_ratio: float = 1.0,
) -> None:
    """Compute all stats for a single rep and write outputs."""
    df = pd.read_parquet(phenotype_path)
    logger.info("Computing stats for %s (%d rows)", phenotype_path, len(df))

    stats: dict[str, Any] = {
        "n_individuals": len(df),
        "n_generations": int(df["generation"].nunique()) if "generation" in df.columns else 1,
    }

    if case_ascertainment_ratio != 1.0:
        stats["case_ascertainment_ratio"] = case_ascertainment_ratio

    stats["prevalence"] = compute_prevalence(df)
    stats["mortality"] = compute_mortality(df, censor_age)
    stats["regression"] = compute_regression(df)
    stats["cumulative_incidence"] = compute_cumulative_incidence(df, censor_age)
    stats["joint_affection"] = compute_joint_affection(df)
    stats["cumulative_incidence_by_sex"] = compute_cumulative_incidence_by_sex(df, censor_age)
    stats["cumulative_incidence_by_sex_generation"] = compute_cumulative_incidence_by_sex_generation(df, censor_age)

    if gen_censoring is not None:
        stats["censoring"] = compute_censoring_windows(df, censor_age, gen_censoring)
        stats["censoring_confusion"] = compute_censoring_confusion(df, censor_age, gen_censoring)
        stats["censoring_cascade"] = compute_censoring_cascade(df, censor_age, gen_censoring)

    stats["person_years"] = compute_person_years(df, censor_age, gen_censoring)
    stats["family_size"] = compute_mean_family_size(df)

    # Read full pedigree once (used for both pair extraction and pair counts)
    df_ped = pd.read_parquet(pedigree_path) if pedigree_path is not None else None

    logger.info("Extracting relationship pairs...")
    t0 = time.perf_counter()
    if df_ped is not None:
        pg = PedigreeGraph.from_subsample(df_ped, df)
        pairs = pg.extract_pairs(max_degree=max_degree)
        full_counts = pg.count_pairs(max_degree=max_degree, scope="full")
    else:
        pairs = PedigreeGraph(df).extract_pairs(max_degree=max_degree)
        full_counts = None
    logger.info(
        "Relationship pairs extracted in %.1fs: %s",
        time.perf_counter() - t0,
        ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
    )

    stats["pair_counts"] = {k: len(v[0]) for k, v in pairs.items()}

    stats["parent_status"] = compute_parent_status(df, df_ped)

    if df_ped is not None and full_counts is not None:
        stats["pair_counts_ped"] = full_counts
        stats["n_individuals_ped"] = len(df_ped)
        stats["n_generations_ped"] = int(df_ped["generation"].nunique()) if "generation" in df_ped.columns else 1
        logger.info(
            "Pedigree pair counts (from same graph): %s",
            ", ".join(f"{k}: {v}" for k, v in full_counts.items()),
        )

    if df_ped is not None:
        logger.info("Computing mate liability correlations...")
        stats["mate_correlation"] = compute_mate_correlation(df_ped)
        del df_ped
    else:
        del df_ped

    # Fast sequential computations
    stats["liability_correlations"] = compute_liability_correlations(df, seed=seed, pairs=pairs)
    stats["affected_correlations"] = compute_affected_correlations(df, seed=seed, pairs=pairs)
    stats["parent_offspring_corr"] = compute_parent_offspring_corr(df)
    stats["parent_offspring_corr_by_sex"] = compute_parent_offspring_corr_by_sex(df)
    stats["parent_offspring_affected_corr"] = compute_parent_offspring_affected_corr(df)
    stats["observed_h2_estimators"] = compute_observed_h2_estimators(stats)

    # Expensive MLE computations — run in parallel (scipy.optimize releases the GIL)
    logger.info("Computing tetrachoric correlations in parallel...")
    t_mle = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:
        fut_tetra = pool.submit(compute_tetrachoric, df, seed=seed, pairs=pairs)
        fut_tetra_gen = pool.submit(compute_tetrachoric_by_generation, df, seed=seed, pairs=pairs)
        fut_cross = pool.submit(compute_cross_trait_tetrachoric, df, seed=seed, pairs=pairs)
        fut_tetra_sex = pool.submit(compute_tetrachoric_by_sex, df, seed=seed, pairs=pairs)

        stats["tetrachoric"] = fut_tetra.result()
        stats["tetrachoric_by_generation"] = fut_tetra_gen.result()
        stats["cross_trait_tetrachoric"] = fut_cross.result()
        stats["tetrachoric_by_sex"] = fut_tetra_sex.result()
    logger.info("All MLE correlations computed in %.1fs", time.perf_counter() - t_mle)

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
    """Command-line interface for phenotype statistics computation."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Compute phenotype statistics")
    add_logging_args(parser)
    parser.add_argument("phenotype", help="Input phenotype parquet")
    parser.add_argument("censor_age", type=float)
    parser.add_argument("stats_output", help="Output stats YAML")
    parser.add_argument("samples_output", help="Output samples parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-censoring", type=str, default=None, help="Per-generation censoring windows as JSON dict")
    parser.add_argument("--pedigree", default=None, help="Full pedigree parquet for G_ped pair counts")
    parser.add_argument(
        "--max-degree",
        dest="max_degree",
        type=int,
        default=2,
        help="Maximum kinship degree for pair extraction (1-5, default 2)",
    )

    # Trait 1 frailty params
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--phenotype-model1", default=None)
    parser.add_argument(
        "--phenotype-params1", type=str, default=None, help='JSON dict, e.g. \'{"scale": 2160, "rho": 0.8}\''
    )

    # Trait 2 frailty params
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--phenotype-model2", default=None)
    parser.add_argument("--phenotype-params2", type=str, default=None)

    args = parser.parse_args()
    init_logging(args)

    frailty_params = None
    if args.beta1 is not None and args.phenotype_model1 == "frailty" and args.phenotype_params1:
        pp1 = json.loads(args.phenotype_params1)
        pp2 = json.loads(args.phenotype_params2) if args.phenotype_params2 else {}
        frailty_params = {
            "trait1": {
                "beta": args.beta1,
                "hazard_model": pp1.get("distribution", ""),
                "hazard_params": {k: v for k, v in pp1.items() if k != "distribution"},
            },
            "trait2": {
                "beta": args.beta2,
                "hazard_model": pp2.get("distribution", ""),
                "hazard_params": {k: v for k, v in pp2.items() if k != "distribution"},
            }
            if args.phenotype_model2 == "frailty"
            else {},
        }

    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    main(
        args.phenotype,
        args.censor_age,
        args.stats_output,
        args.samples_output,
        seed=args.seed,
        gen_censoring=gen_censoring,
        frailty_params=frailty_params,
        pedigree_path=args.pedigree,
        max_degree=args.max_degree,
    )
