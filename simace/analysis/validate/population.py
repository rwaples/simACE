"""Population-level checks and per-generation / family-size summaries."""

from typing import Any

import numpy as np
import pandas as pd

from ._common import _result


def compute_per_generation_stats(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Compute per-generation statistics for two traits.

    For each generation, computes liability mean/variance/sd and per-component
    (A, C, E) mean/variance for both traits.

    Args:
        df: Pedigree DataFrame with columns id, A1, C1, E1, A2, C2, E2.
        params: Scenario parameters; requires keys ``N`` and ``G_ped``.

    Returns:
        Dict keyed by ``"generation_{g}"`` where each value is a dict of
        summary statistics (n, liability mean/variance/sd, component mean/var).
    """
    N = params["N"]
    ngen = params["G_ped"]

    # Assign generation labels once via integer division
    gen_labels = df["id"].values // N

    results = {}
    for gen in range(1, ngen + 1):
        gen_mask = gen_labels == (gen - 1)
        gen_df = df[gen_mask]

        gen_stats: dict[str, int | float] = {"n": int(gen_mask.sum())}
        for t in [1, 2]:
            a_vals = gen_df[f"A{t}"].values
            c_vals = gen_df[f"C{t}"].values
            e_vals = gen_df[f"E{t}"].values
            liability = a_vals + c_vals + e_vals
            gen_stats[f"liability{t}_mean"] = float(liability.mean())
            gen_stats[f"liability{t}_variance"] = float(liability.var())
            gen_stats[f"liability{t}_sd"] = float(liability.std())
            for comp, vals in [("A", a_vals), ("C", c_vals), ("E", e_vals)]:
                col = f"{comp}{t}"
                gen_stats[f"{col}_mean"] = float(vals.mean())
                gen_stats[f"{col}_var"] = float(vals.var())

        results[f"generation_{gen}"] = gen_stats

    return results


def validate_population(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Validate population-level properties.

    Checks that each generation has exactly ``N`` individuals, the number of
    generations equals ``G_ped``, and the mean offspring per mother is
    approximately ``N / n_females`` (always ~2.0 for balanced sex ratios).

    Args:
        df: Pedigree DataFrame with columns id and mother.
        params: Scenario parameters; requires keys ``N``, ``G_ped``.

    Returns:
        Dict of check-name to result dicts.
    """
    results = {}
    N = params["N"]
    ngen = params["G_ped"]

    gen_assignments = df["id"].values // N
    gen_sizes = np.bincount(gen_assignments, minlength=ngen)[:ngen].tolist()

    all_correct = all(s == N for s in gen_sizes)
    results["generation_sizes"] = _result(
        all_correct,
        f"Generation sizes: {gen_sizes} (expected: {N} each)",
        expected=N,
        observed=gen_sizes,
    )

    results["generation_count"] = _result(
        len(gen_sizes) == ngen,
        f"Number of generations: {len(gen_sizes)} (expected: {ngen})",
        expected=ngen,
        observed=len(gen_sizes),
    )

    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        family_sizes = non_founders.groupby("mother").size()
        mean_fam = family_sizes.mean()
        # Mean offspring per mother is ~N / n_mothers ~= 2.0 for balanced sex
        expected_mean = 2.0
        fam_ok = abs(mean_fam - expected_mean) < expected_mean * 0.5
        results["family_size"] = _result(
            fam_ok,
            f"Mean offspring per mother: {mean_fam:.2f} (expected: ~{expected_mean:.1f})",
            expected=expected_mean,
            observed=float(mean_fam),
        )
    else:
        results["family_size"] = _result(True, "No non-founders to check family size")

    return results


def compute_family_size_distribution(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Compute offspring count distributions per parent sex.

    Args:
        df: Pedigree DataFrame with columns mother and father.
        params: Scenario parameters (unused but accepted for API consistency).

    Returns:
        Dict with keys ``"mother"`` and ``"father"``, each mapping to a dict
        of summary statistics (mean, median, std, n_parents). Empty dict if
        no non-founders exist.
    """
    non_founders = df[df["mother"] != -1]
    if len(non_founders) == 0:
        return {}

    mother_counts = non_founders.groupby("mother").size()
    father_counts = non_founders.groupby("father").size()

    result = {}
    for label, counts in [("mother", mother_counts), ("father", father_counts)]:
        result[label] = {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "std": float(counts.std()),
            "n_parents": len(counts),
        }

    return result
