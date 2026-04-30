"""Prevalence, mortality, cumulative incidence, and joint-affection statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core.numerics import fast_linregress
from simace.core.relationships import SEX_LEVELS

if TYPE_CHECKING:
    import pandas as pd


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
        means = df.groupby("generation")[["affected1", "affected2"]].mean()
        result["by_generation"] = {
            int(gen): {"trait1": float(row["affected1"]), "trait2": float(row["affected2"])}
            for gen, row in means.iterrows()
        }
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
        for sex_val, sex_label in SEX_LEVELS:
            mask = df["sex"].values == sex_val
            n_sex = int(mask.sum())
            if n_sex > 0:
                by_sex[sex_label] = round(float(np.sum(a1[mask] & a2[mask])) / n_sex, 4)

    return {"counts": counts, "proportions": proportions, "n": n, "by_sex": by_sex}


def compute_cumulative_incidence_by_sex(
    df: pd.DataFrame,
    censor_age: float,
    n_points: int = 200,
) -> dict[str, Any]:
    """Compute cumulative incidence curves split by sex (0=female, 1=male)."""
    if "sex" not in df.columns:
        return {}

    ages = np.linspace(0, censor_age, n_points)
    sex = df["sex"].values
    result = {}
    for trait_num in [1, 2]:
        aff = df[f"affected{trait_num}"].values.astype(bool)
        t_obs_aff = df[f"t_observed{trait_num}"].values[aff]
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
    df: pd.DataFrame,
    censor_age: float,
    n_points: int = 200,
) -> dict[str, Any]:
    """Compute cumulative incidence curves split by sex and generation."""
    if "sex" not in df.columns or "generation" not in df.columns:
        return {}

    ages = np.linspace(0, censor_age, n_points)
    generations = sorted(df["generation"].unique())
    sex = df["sex"].values
    gen_arr = df["generation"].values
    result = {}
    for trait_num in [1, 2]:
        aff = df[f"affected{trait_num}"].values.astype(bool)
        t_obs_aff = df[f"t_observed{trait_num}"].values[aff]
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
