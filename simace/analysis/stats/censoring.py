"""Censoring window, confusion-matrix, cascade, and person-years statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


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
