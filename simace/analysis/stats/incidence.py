"""Prevalence, mortality, cumulative incidence, and joint-affection statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from simace.core.numerics import fast_linregress
from simace.core.relationships import SEX_LEVELS

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    type _Frame = pd.DataFrame | pl.DataFrame


def _cumulative_curve(
    times: np.ndarray,
    ages: np.ndarray,
    n: int,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Empirical cumulative incidence ``F(a) = #{t_i <= a, valid_i} / n``.

    When ``mask`` is given, only individuals where ``mask`` is True
    contribute events to the numerator; ``n`` is the cohort size in the
    denominator (typically larger than ``mask.sum()`` because individuals
    who never experience the event still count toward the at-risk pool).
    """
    selected = times[mask] if mask is not None else times
    sorted_t = np.sort(selected)
    return np.searchsorted(sorted_t, ages, side="right") / n


def compute_mortality(df: _Frame, censor_age: float) -> dict[str, Any]:
    """Compute decade-binned mortality rates from death ages.

    Args:
        df: Phenotype DataFrame with ``death_age`` column.
        censor_age: Maximum observation age.

    Returns:
        Dict with ``decade_labels`` and ``rates`` lists.
    """
    decade_edges = np.arange(0, censor_age + 10, 10)
    mortality_rates, decade_labels = [], []
    death_ages = df["death_age"].to_numpy()
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
    df: _Frame,
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
        aff = df[f"affected{trait_num}"].to_numpy().astype(bool)
        t_obs = df[f"t_observed{trait_num}"].to_numpy()
        t_raw = df[f"t{trait_num}"].to_numpy()
        obs_inc = _cumulative_curve(t_obs, ages, n, mask=aff)
        true_inc = _cumulative_curve(t_raw, ages, n)
        half_idx = np.searchsorted(obs_inc, obs_inc[-1] / 2)
        result[f"trait{trait_num}"] = {
            "ages": ages.tolist(),
            "observed_values": obs_inc.tolist(),
            "true_values": true_inc.tolist(),
            "half_target_age": float(ages[min(half_idx, len(ages) - 1)]),
        }
    return result


def _build_entry_times(df: _Frame, gen_censoring: dict[int, list[float]] | None) -> np.ndarray:
    """Per-individual delayed-entry times from per-generation windows.

    Returns 0 for every individual when ``gen_censoring`` is None or no
    ``generation`` column exists. Individuals in generations not listed in
    ``gen_censoring`` also default to 0.
    """
    n = len(df)
    if gen_censoring is None or "generation" not in df.columns:
        return np.zeros(n)
    entry = np.zeros(n)
    gens = df["generation"].to_numpy()
    for gen, (lo, _hi) in gen_censoring.items():
        entry[gens == int(gen)] = float(lo)
    return entry


class _AJResult(TypedDict):
    """Aalen-Johansen CIF grids + event counts."""

    aj_disease: np.ndarray
    aj_death: np.ndarray
    aj_survival: np.ndarray
    n: int
    n_events_disease: int
    n_events_death: int


def _aalen_johansen(
    entry: np.ndarray,
    exit_time: np.ndarray,
    event_type: np.ndarray,
    ages: np.ndarray,
) -> _AJResult:
    """Aalen-Johansen CIF for disease (cause 1) with death (cause 2) as competing event.

    Args:
        entry: per-individual entry times (delayed-entry support).
        exit_time: per-individual exit times.
        event_type: 0=censored, 1=disease, 2=death.
        ages: monotone-increasing grid for step-evaluation.

    Returns dict with keys ``aj_disease``, ``aj_death``, ``aj_survival`` (all
    arrays on the ``ages`` grid), ``n``, ``n_events_disease``, and ``n_events_death``.
    """
    valid = entry <= exit_time
    entry = entry[valid]
    exit_time = exit_time[valid]
    event_type = event_type[valid]
    n = int(valid.sum())
    n_disease = int((event_type == 1).sum())
    n_death = int((event_type == 2).sum())

    n_ages = len(ages)
    if n == 0:
        return {
            "aj_disease": np.zeros(n_ages),
            "aj_death": np.zeros(n_ages),
            "aj_survival": np.ones(n_ages),
            "n": 0,
            "n_events_disease": 0,
            "n_events_death": 0,
        }

    is_event = event_type != 0
    if not is_event.any():
        return {
            "aj_disease": np.zeros(n_ages),
            "aj_death": np.zeros(n_ages),
            "aj_survival": np.ones(n_ages),
            "n": n,
            "n_events_disease": 0,
            "n_events_death": 0,
        }

    sorted_entry = np.sort(entry)
    sorted_exit = np.sort(exit_time)

    event_times = exit_time[is_event]
    event_kinds = event_type[is_event]
    unique_t, inv = np.unique(event_times, return_inverse=True)
    n_unique = len(unique_t)
    d_disease = np.bincount(inv, weights=(event_kinds == 1).astype(float), minlength=n_unique)
    d_death = np.bincount(inv, weights=(event_kinds == 2).astype(float), minlength=n_unique)
    d_total = d_disease + d_death

    entered = np.searchsorted(sorted_entry, unique_t, side="right")
    exited_before = np.searchsorted(sorted_exit, unique_t, side="left")
    Y = (entered - exited_before).astype(float)

    safe = Y > 0
    hazard_total = np.zeros(n_unique)
    hazard_total[safe] = d_total[safe] / Y[safe]
    s_after = np.cumprod(1.0 - hazard_total)
    s_before = np.empty(n_unique)
    s_before[0] = 1.0
    s_before[1:] = s_after[:-1]

    inc_disease = np.zeros(n_unique)
    inc_death = np.zeros(n_unique)
    inc_disease[safe] = s_before[safe] * d_disease[safe] / Y[safe]
    inc_death[safe] = s_before[safe] * d_death[safe] / Y[safe]
    F_disease = np.cumsum(inc_disease)
    F_death = np.cumsum(inc_death)

    idx = np.searchsorted(unique_t, ages, side="right") - 1
    valid_idx = idx >= 0
    F_disease_grid = np.where(valid_idx, F_disease[np.clip(idx, 0, n_unique - 1)], 0.0)
    F_death_grid = np.where(valid_idx, F_death[np.clip(idx, 0, n_unique - 1)], 0.0)
    S_grid = np.where(valid_idx, s_after[np.clip(idx, 0, n_unique - 1)], 1.0)

    return {
        "aj_disease": F_disease_grid,
        "aj_death": F_death_grid,
        "aj_survival": S_grid,
        "n": n,
        "n_events_disease": n_disease,
        "n_events_death": n_death,
    }


def _exit_event_arrays(df: _Frame, trait_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (exit_time, event_type) arrays from censoring columns for a trait.

    event_type: 1=disease, 2=death, 0=censored.
    """
    affected = df[f"affected{trait_num}"].to_numpy().astype(bool)
    death_censored = df[f"death_censored{trait_num}"].to_numpy().astype(bool)
    exit_time = df[f"t_observed{trait_num}"].to_numpy().astype(float)
    event_type = np.where(affected, 1, np.where(death_censored, 2, 0)).astype(np.int8)
    return exit_time, event_type


def compute_cumulative_incidence_aj(
    df: _Frame,
    censor_age: float,
    n_points: int = 200,
    *,
    gen_censoring: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    """Aalen-Johansen cumulative incidence with death as competing event.

    Supports delayed entry via ``gen_censoring`` (per-generation
    ``[left, right]`` windows); each individual enters the risk set at
    their generation's left bound. With ``gen_censoring=None`` or all-zero
    left bounds, the result equals the no-delayed-entry case.
    """
    ages = np.linspace(0, censor_age, n_points)
    entry = _build_entry_times(df, gen_censoring)
    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        exit_time, event_type = _exit_event_arrays(df, trait_num)
        aj = _aalen_johansen(entry, exit_time, event_type, ages)
        terminal = float(aj["aj_disease"][-1])
        if terminal > 0:
            half_idx = int(np.searchsorted(aj["aj_disease"], terminal / 2))
            half_age = float(ages[min(half_idx, len(ages) - 1)])
        else:
            half_age = float(ages[-1])
        entry_dict: dict[str, Any] = {
            "ages": ages.tolist(),
            "aj_values": aj["aj_disease"].tolist(),
            "aj_death_values": aj["aj_death"].tolist(),
            "aj_survival": aj["aj_survival"].tolist(),
            "n": int(aj["n"]),
            "n_events_disease": int(aj["n_events_disease"]),
            "n_events_death": int(aj["n_events_death"]),
            "half_target_age": half_age,
        }
        result[f"trait{trait_num}"] = entry_dict
    return result


def compute_cumulative_incidence_aj_by_sex(
    df: _Frame,
    censor_age: float,
    n_points: int = 200,
    *,
    gen_censoring: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    """Aalen-Johansen cumulative incidence stratified by sex."""
    if "sex" not in df.columns:
        return {}
    ages = np.linspace(0, censor_age, n_points)
    entry = _build_entry_times(df, gen_censoring)
    sex = df["sex"].to_numpy()
    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        exit_time, event_type = _exit_event_arrays(df, trait_num)
        trait_result: dict[str, Any] = {}
        for sex_val, sex_label in SEX_LEVELS:
            mask = sex == sex_val
            n_sex = int(mask.sum())
            if n_sex == 0:
                continue
            aj = _aalen_johansen(entry[mask], exit_time[mask], event_type[mask], ages)
            stratum: dict[str, Any] = {
                "ages": ages.tolist(),
                "aj_values": aj["aj_disease"].tolist(),
                "aj_death_values": aj["aj_death"].tolist(),
                "aj_survival": aj["aj_survival"].tolist(),
                "n": n_sex,
                "n_events_disease": int(aj["n_events_disease"]),
                "n_events_death": int(aj["n_events_death"]),
                "prevalence": float(aj["n_events_disease"] / n_sex) if n_sex else 0.0,
            }
            trait_result[sex_label] = stratum
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_cumulative_incidence_aj_by_sex_generation(
    df: _Frame,
    censor_age: float,
    n_points: int = 200,
    *,
    gen_censoring: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    """Aalen-Johansen cumulative incidence stratified by sex and generation."""
    if "sex" not in df.columns or "generation" not in df.columns:
        return {}
    ages = np.linspace(0, censor_age, n_points)
    entry = _build_entry_times(df, gen_censoring)
    sex = df["sex"].to_numpy()
    gen_arr = df["generation"].to_numpy()
    generations = sorted(df["generation"].unique())
    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        exit_time, event_type = _exit_event_arrays(df, trait_num)
        trait_result: dict[str, Any] = {}
        for gen in generations:
            gen_result: dict[str, Any] = {}
            in_gen = gen_arr == gen
            for sex_val, sex_label in SEX_LEVELS:
                mask = in_gen & (sex == sex_val)
                n_stratum = int(mask.sum())
                if n_stratum == 0:
                    continue
                aj = _aalen_johansen(entry[mask], exit_time[mask], event_type[mask], ages)
                stratum: dict[str, Any] = {
                    "ages": ages.tolist(),
                    "aj_values": aj["aj_disease"].tolist(),
                    "aj_death_values": aj["aj_death"].tolist(),
                    "aj_survival": aj["aj_survival"].tolist(),
                    "n": n_stratum,
                    "n_events_disease": int(aj["n_events_disease"]),
                    "n_events_death": int(aj["n_events_death"]),
                    "prevalence": float(aj["n_events_disease"] / n_stratum) if n_stratum else 0.0,
                }
                gen_result[sex_label] = stratum
            trait_result[f"gen{int(gen)}"] = gen_result
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_regression(df: _Frame) -> dict[str, Any]:
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
        aff = df[aff_col].to_numpy().astype(bool)
        liab = df[liab_col].to_numpy().astype(float)
        t_obs = df[t_col].to_numpy().astype(float)
        keep = aff & ~np.isnan(liab) & ~np.isnan(t_obs)
        n_kept = int(keep.sum())
        if n_kept < 2:
            result[f"trait{trait_num}"] = None
            continue
        slope, intercept, r, stderr, pvalue = fast_linregress(liab[keep], t_obs[keep])
        result[f"trait{trait_num}"] = {
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "r2": r**2,
            "stderr": stderr,
            "pvalue": pvalue,
            "n": n_kept,
        }
    return result


def compute_joint_affection(df: _Frame) -> dict[str, Any]:
    """Compute 2x2 contingency table for trait1 x trait2 affection status."""
    a1 = df["affected1"].to_numpy().astype(bool)
    a2 = df["affected2"].to_numpy().astype(bool)
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
        for sex_val, sex_label in SEX_LEVELS:
            mask = df["sex"].to_numpy() == sex_val
            n_sex = int(mask.sum())
            if n_sex > 0:
                by_sex[sex_label] = round(float(np.sum(a1[mask] & a2[mask])) / n_sex, 4)

    return {"counts": counts, "proportions": proportions, "n": n, "by_sex": by_sex}


def compute_cumulative_incidence_by_sex(
    df: _Frame,
    censor_age: float,
    n_points: int = 200,
) -> dict[str, Any]:
    """Compute cumulative incidence curves split by sex (0=female, 1=male)."""
    if "sex" not in df.columns:
        return {}

    ages = np.linspace(0, censor_age, n_points)
    sex = df["sex"].to_numpy()
    result = {}
    for trait_num in [1, 2]:
        aff = df[f"affected{trait_num}"].to_numpy().astype(bool)
        t_obs_aff = df[f"t_observed{trait_num}"].to_numpy()[aff]
        sex_aff = sex[aff]

        trait_result = {}
        for sex_val, sex_label in SEX_LEVELS:
            n_sex = int((sex == sex_val).sum())
            if n_sex == 0:
                continue
            in_stratum_aff = sex_aff == sex_val
            inc = _cumulative_curve(t_obs_aff, ages, n_sex, mask=in_stratum_aff)
            trait_result[sex_label] = {
                "ages": ages.tolist(),
                "values": inc.tolist(),
                "n": n_sex,
                "prevalence": float(in_stratum_aff.sum() / n_sex),
            }
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_cumulative_incidence_by_sex_generation(
    df: _Frame,
    censor_age: float,
    n_points: int = 200,
) -> dict[str, Any]:
    """Compute cumulative incidence curves split by sex and generation."""
    if "sex" not in df.columns or "generation" not in df.columns:
        return {}

    ages = np.linspace(0, censor_age, n_points)
    generations = sorted(df["generation"].unique())
    sex = df["sex"].to_numpy()
    gen_arr = df["generation"].to_numpy()
    result = {}
    for trait_num in [1, 2]:
        aff = df[f"affected{trait_num}"].to_numpy().astype(bool)
        t_obs_aff = df[f"t_observed{trait_num}"].to_numpy()[aff]
        sex_aff = sex[aff]
        gen_aff = gen_arr[aff]

        trait_result: dict[str, Any] = {}
        for gen in generations:
            gen_result: dict[str, Any] = {}
            for sex_val, sex_label in SEX_LEVELS:
                n_sex = int(((gen_arr == gen) & (sex == sex_val)).sum())
                if n_sex == 0:
                    continue
                in_stratum_aff = (gen_aff == gen) & (sex_aff == sex_val)
                inc = _cumulative_curve(t_obs_aff, ages, n_sex, mask=in_stratum_aff)
                gen_result[sex_label] = {
                    "ages": ages.tolist(),
                    "values": inc.tolist(),
                    "n": n_sex,
                    "prevalence": float(in_stratum_aff.sum() / n_sex),
                }
            trait_result[f"gen{int(gen)}"] = gen_result
        result[f"trait{trait_num}"] = trait_result
    return result
