"""Censoring window, confusion-matrix, cascade, and person-years statistics.

Library-agnostic by design (ADR 0015): columns come out through
``.to_numpy()`` and all slicing runs on NumPy masks, so any frame exposing
that interface yields identical results. polars is the canonical caller; the
NumPy boundary is a deliberate contract, not a migration leftover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    type _Frame = pd.DataFrame | pl.DataFrame


def compute_censoring_windows(
    df: _Frame,
    censor_age: float,
    gen_censoring: dict[int, list[float]],
    n_points: int = 300,
) -> dict[str, Any] | None:
    """Compute per-generation censoring breakdown and incidence curves.

    The shares ``pct_affected``, ``left_censored``, ``right_censored`` and
    ``death_censored`` partition the generation by the censor stage's flags:
    age-censored rows split by which side of the window midpoint the onset
    falls on, with a never-onset (null) row right-censored as ``run_censor``
    treats it, and death-censored rows are those not already age-censored.
    See :func:`compute_censoring_cascade` for why the flags, not the times.

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
    gens = df["generation"].to_numpy()
    result = {}
    for gen in sorted(gen_censoring.keys()):
        win_lo, win_hi = gen_censoring[gen]
        gen_mask = gens == gen
        n_gen = int(gen_mask.sum())
        if n_gen == 0:
            result[f"gen{gen}"] = {"n": 0}
            continue
        gen_result: dict[str, Any] = {"n": int(n_gen)}
        for trait_num in [1, 2]:
            t_raw = df[f"t{trait_num}"].to_numpy()[gen_mask]
            t_obs = df[f"t_observed{trait_num}"].to_numpy()[gen_mask]
            affected = df[f"affected{trait_num}"].to_numpy()[gen_mask].astype(bool)
            age_c = df[f"age_censored{trait_num}"].to_numpy()[gen_mask].astype(bool)
            death_c = df[f"death_censored{trait_num}"].to_numpy()[gen_mask].astype(bool)
            left_c = age_c & (t_raw < np.float64((win_lo + win_hi) / 2))
            obs_inc = np.searchsorted(np.sort(t_obs[affected]), ages, side="right") / n_gen
            true_inc = np.searchsorted(np.sort(t_raw), ages, side="right") / n_gen
            gen_result[f"trait{trait_num}"] = {
                "true_incidence": true_inc.tolist(),
                "observed_incidence": obs_inc.tolist(),
                "pct_affected": float(affected.mean()),
                "left_censored": float(left_c.mean()),
                "right_censored": float((age_c & ~left_c).mean()),
                "death_censored": float((death_c & ~age_c).mean()),
            }
        result[f"gen{gen}"] = gen_result
    return {"generations": result, "censoring_ages": ages.tolist(), "censor_age": int(censor_age)}


def compute_censoring_confusion(
    df: _Frame,
    censor_age: float,
    gen_censoring: dict[int, list[float]],
) -> dict[str, Any]:
    """Compute per-trait 2x2 confusion matrix: true affected vs. observed affected.

    Only includes individuals from phenotyped generations (observation window > 0).
    """
    keep = np.ones(len(df), dtype=bool)
    if "generation" in df.columns:
        active_gens = {int(g) for g, (lo, hi) in gen_censoring.items() if hi > lo}
        if active_gens:
            gens = df["generation"].to_numpy()
            keep = np.isin(gens, list(active_gens))

    result = {}
    for trait in [1, 2]:
        t_col = f"t{trait}"
        a_col = f"affected{trait}"
        if t_col not in df.columns or a_col not in df.columns:
            continue
        true_aff = df[t_col].to_numpy()[keep] < censor_age
        obs_aff = df[a_col].to_numpy()[keep].astype(bool)
        n = int(keep.sum())
        result[f"trait{trait}"] = {
            "tp": int(np.sum(true_aff & obs_aff)),
            "fn": int(np.sum(true_aff & ~obs_aff)),
            "fp": int(np.sum(~true_aff & obs_aff)),
            "tn": int(np.sum(~true_aff & ~obs_aff)),
            "n": n,
        }
    return result


def compute_censoring_cascade(
    df: _Frame,
    censor_age: float,
    gen_censoring: dict[int, list[float]],
) -> dict[str, Any]:
    """Per-trait, per-generation decomposition of true cases by censoring fate.

    A true case is an affected row or an onset before ``censor_age``. Its fate
    comes from the censor stage's flags, not from comparing ``t`` with the
    window and ``death_age`` again: the stats stage reads those columns
    narrowed to float32, where an onset can tie or swap order with the death
    age or a window edge that it did not tie on the float64 values the flags
    came from. Age-censored cases split into left-truncated and
    right-censored by which side of the window midpoint the onset falls on,
    which is exact for any window wider than the onset's float32 resolution.
    An unaffected onset that ties ``censor_age`` in float32 is not counted.
    """
    if "generation" not in df.columns:
        return {}

    windows: dict[int, tuple[float, float]] = {}
    for g, (lo, hi) in gen_censoring.items():
        if hi > lo:
            windows[int(g)] = (lo, hi)

    if not windows:
        return {}

    result: dict[str, Any] = {}
    gens = df["generation"].to_numpy()
    for trait in [1, 2]:
        columns = [f"{name}{trait}" for name in ("t", "affected", "age_censored", "death_censored")]
        if not all(c in df.columns for c in columns):
            continue
        t_all, affected_all, age_censored_all, death_censored_all = (df[c].to_numpy() for c in columns)
        trait_result: dict[str, Any] = {}
        for g in sorted(windows.keys()):
            lo, hi = windows[g]
            gen_mask = gens == g
            t = t_all[gen_mask]
            affected = np.asarray(affected_all[gen_mask], dtype=bool)
            # Compared in t's own dtype (NEP 50): monotone rounding keeps every float64 onset >= censor_age out.
            true_affected = (t < censor_age) | affected
            n_true = int(true_affected.sum())
            n_gen = int(gen_mask.sum())

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

            age_censored = true_affected & np.asarray(age_censored_all[gen_mask], dtype=bool)
            left_trunc = age_censored & (t < np.float64((lo + hi) / 2))
            right_cens = age_censored & ~left_trunc
            observed = affected
            death_cens = true_affected & ~age_censored & np.asarray(death_censored_all[gen_mask], dtype=bool)

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
    df: _Frame,
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

    gens = df["generation"].to_numpy()
    death_all = df["death_age"].to_numpy() if has_death else None
    for g in np.unique(gens):
        lo, hi = windows.get(int(g), (0.0, censor_age))
        if hi <= lo:
            continue
        gen_mask = gens == g
        n = int(gen_mask.sum())

        # End of observation for each person (not trait-specific)
        if death_all is not None:
            death_ages = death_all[gen_mask]
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
            if t_col not in df.columns:
                continue
            t_obs = df[t_col].to_numpy()[gen_mask]
            if has_death:
                trait_end = np.minimum(np.minimum(t_obs, death_ages), hi)
            else:
                trait_end = np.minimum(t_obs, hi)
            trait_py[f"trait{trait}"] += float(np.clip(trait_end - lo, 0, None).sum())

    return {
        "total": round(total_py, 1),
        "deaths": total_deaths,
        **{k: round(v, 1) for k, v in trait_py.items()},
    }


@dataclass(frozen=True)
class ConfusionRates:
    """Sensitivity, specificity, PPV and NPV of observed against true affection (NaN where a margin is 0)."""

    sensitivity: float
    specificity: float
    ppv: float
    npv: float


def confusion_rates(tp: float, fn: float, fp: float, tn: float) -> ConfusionRates:
    """The four rates of a 2x2 table of observed (rows: true affected, unaffected) against true affection."""
    return ConfusionRates(
        sensitivity=tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        specificity=tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        ppv=tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        npv=tn / (tn + fn) if (tn + fn) > 0 else float("nan"),
    )
