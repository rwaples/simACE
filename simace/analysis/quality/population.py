"""Population-level checks and per-generation / family-size summaries.

Library-agnostic by design (ADR 0015): columns come out through
``.to_numpy()`` and all grouping/slicing runs in NumPy, so any frame exposing
that interface yields identical results. polars is the canonical caller; the
NumPy boundary is a deliberate contract, not a migration leftover.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._common import _result

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    type _Frame = pd.DataFrame | pl.DataFrame


def _population_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Return covariance with the population denominator (ddof=0)."""
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def compute_per_generation_stats(df: _Frame, params: dict[str, Any]) -> dict[str, Any]:
    """Compute per-generation statistics for two traits.

    For each generation, computes liability mean/variance/sd, per-component
    (A, C, E) mean/variance, and A-vs-(C+E) covariance primitives for both traits.

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
    gen_labels = df["id"].to_numpy() // N
    comp_all = {f"{c}{t}": df[f"{c}{t}"].to_numpy() for c in ("A", "C", "E") for t in (1, 2)}

    results = {}
    for gen in range(1, ngen + 1):
        gen_mask = gen_labels == (gen - 1)

        gen_stats: dict[str, int | float] = {"n": int(gen_mask.sum())}
        for t in [1, 2]:
            a_vals = comp_all[f"A{t}"][gen_mask]
            c_vals = comp_all[f"C{t}"][gen_mask]
            e_vals = comp_all[f"E{t}"][gen_mask]
            non_genetic = c_vals + e_vals
            liability = a_vals + non_genetic
            gen_stats[f"liability{t}_mean"] = float(liability.mean())
            gen_stats[f"liability{t}_variance"] = float(liability.var())
            gen_stats[f"liability{t}_sd"] = float(liability.std())
            for comp, vals in [("A", a_vals), ("C", c_vals), ("E", e_vals)]:
                col = f"{comp}{t}"
                gen_stats[f"{col}_mean"] = float(vals.mean())
                gen_stats[f"{col}_var"] = float(vals.var())

            # Match the population-variance convention above. np.cov defaults
            # to ddof=1, which would break exact per-generation identities.
            gen_stats[f"A{t}_cov_non_genetic"] = _population_covariance(a_vals, non_genetic)
            gen_stats[f"A{t}_cov_C"] = _population_covariance(a_vals, c_vals)
            gen_stats[f"A{t}_cov_E"] = _population_covariance(a_vals, e_vals)

        results[f"generation_{gen}"] = gen_stats

    return results


def validate_population(df: _Frame, params: dict[str, Any]) -> dict[str, Any]:
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

    gen_assignments = df["id"].to_numpy() // N
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

    mothers_all = df["mother"].to_numpy()
    mothers_nf = mothers_all[mothers_all != -1]
    if mothers_nf.size > 0:
        family_sizes = np.unique(mothers_nf, return_counts=True)[1]
        mean_fam = float(family_sizes.mean())
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


def compute_family_size_distribution(df: _Frame, params: dict[str, Any]) -> dict[str, Any]:
    """Compute offspring count distributions per parent sex.

    Args:
        df: Pedigree DataFrame with columns mother and father.
        params: Scenario parameters (unused but accepted for API consistency).

    Returns:
        Dict with keys ``"mother"`` and ``"father"``, each mapping to a dict
        of summary statistics (mean, median, std, n_parents). Empty dict if
        no non-founders exist.
    """
    mothers_all = df["mother"].to_numpy()
    nf_mask = mothers_all != -1
    if not nf_mask.any():
        return {}

    mother_counts = np.unique(mothers_all[nf_mask], return_counts=True)[1]
    father_counts = np.unique(df["father"].to_numpy()[nf_mask], return_counts=True)[1]

    result = {}
    for label, counts in [("mother", mother_counts), ("father", father_counts)]:
        result[label] = {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "std": float(np.std(counts, ddof=1)),
            "n_parents": len(counts),
        }

    return result
