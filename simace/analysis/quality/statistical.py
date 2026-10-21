"""Variance-component statistical checks (founder variances, correlations).

Library-agnostic by design (ADR 0015): columns come out through
``.to_numpy()`` and all grouping/slicing runs in NumPy, so any frame exposing
that interface yields identical results. polars is the canonical caller; the
NumPy boundary is a deliberate contract, not a migration leftover.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core.numerics import safe_corrcoef
from simace.core.relationships import transmission_coefficients

from ._common import _corr_tolerance, _info, _result, sorted_runs
from .am_relatedness import am_relatedness_mode

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import polars as pl


def _check_variance(founders: Mapping[str, np.ndarray], col: str, expected: float, tol: float = 0.1) -> dict[str, Any]:
    """Check that the variance of `col` in founders is close to `expected`.

    Accumulates in float64 (matching pandas ``Series.var`` on float32 columns).
    """
    var = float(np.var(founders[col], ddof=1, dtype=np.float64))
    return _result(
        abs(var - expected) < tol,
        f"Var({col}) in founders: {var:.4f} (expected: {expected})",
        expected=expected,
        observed=float(var),
    )


def validate_statistical(df: pd.DataFrame | pl.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Validate statistical properties of variance components for two traits.

    Checks founder variances for A, C, E against configured values,
    cross-trait correlations rA and rE, C sharing within households, and E
    independence between siblings. Under assortative mating on a trait, its
    ``Var(A)`` and the cross-trait ``rA`` are reported without being
    asserted. The total variance and the cross-trait ``rC`` are reported
    only: `simace validate`'s ``total_variance`` and ``cross_trait_rC`` test
    them (ADR 0030).

    Args:
        df: Pedigree DataFrame with variance-component columns A1, C1, E1,
            A2, C2, E2.
        params: Scenario parameters; requires keys ``A1``, ``C1``, ``E1``,
            ``A2``, ``C2``, ``E2``, ``rA``, ``rC``.

    Returns:
        Dict of check-name to result dicts.
    """
    results = {}

    rA_param = params.get("rA", 0)
    rC_param = params.get("rC", 0)

    mother_all = df["mother"].to_numpy()
    founder_mask = mother_all == -1
    comp_cols = [f"{c}{t}" for c in ("A", "C", "E") for t in (1, 2)]
    founders = {c: df[c].to_numpy()[founder_mask] for c in comp_cols}

    # For per-generation dict params, resolve to founder sim generation value
    G_sim = params.get("G_sim", 8)
    G_ped = params.get("G_ped", 6)
    founder_sim_gen = G_sim - G_ped

    def _resolve_founder_val(val):
        """Resolve a scalar or per-gen dict to the founder generation value."""
        if isinstance(val, dict):
            from simace.simulation.simulate import resolve_per_gen_param

            return resolve_per_gen_param(val, G_sim)[founder_sim_gen]
        return val

    # Variance checks for both traits. Under assortative mating the additive
    # variance inflates across generations (Bulmer), so the recorded "founders"
    # (first recorded generation) no longer carry the configured A; the
    # am_equilibrium check validates that inflation instead. Skip the
    # A-variance (and the dependent total) for any trait with active AM. C and
    # E are unaffected (drawn fresh each generation), so they remain asserted.
    am_active = {t: am_relatedness_mode(params, t) != "none" for t in (1, 2)}
    for t in [1, 2]:
        for comp in ["A", "C", "E"]:
            col = f"{comp}{t}"
            if comp == "A" and am_active[t]:
                # Reported, not asserted: the am_equilibrium check validates the
                # AM-inflated additive variance.
                am_var = float(np.var(founders[col], ddof=1, dtype=np.float64))
                results[f"variance_{col}"] = _info(
                    f"Var({col}): {am_var:.4f} (AM-inflated; asserted by am_equilibrium)",
                    observed=am_var,
                )
            else:
                results[f"variance_{col}"] = _check_variance(founders, col, _resolve_founder_val(params[col]))

    # Total variances, reported only: `simace validate`'s total_variance tests
    # them (ADR 0030). Var(A + C + E) directly, not the sum of component
    # variances, because AM with C transmission gives A and C a covariance (ADR 0025).
    for t in [1, 2]:
        total = float(np.var(founders[f"A{t}"] + founders[f"C{t}"] + founders[f"E{t}"], ddof=1, dtype=np.float64))
        results[f"total_variance_trait{t}"] = _info(f"Total variance trait {t}: {total:.4f}", observed=total)

    # Cross-trait correlations. Assortative mating on either trait builds
    # disequilibrium into the additive covariance as it does into Var(A), so
    # the recorded founders' rA drifts from the configured value whenever
    # rA != 0; there is no closed-form expectation to assert it against.
    rA_obs = safe_corrcoef(founders["A1"], founders["A2"])
    if am_active[1] or am_active[2]:
        results["cross_trait_rA"] = _info(
            f"Cross-trait A correlation: {rA_obs:.4f} (configured {rA_param}; AM-shifted, not asserted under AM)",
            observed=float(rA_obs),
        )
    else:
        rA_ok = abs(rA_obs - rA_param) < 0.15 if not np.isnan(rA_obs) else rA_param == 0
        results["cross_trait_rA"] = _result(
            rA_ok,
            f"Cross-trait A correlation: {rA_obs:.4f} (expected: {rA_param})",
            expected=rA_param,
            observed=float(rA_obs),
        )
    # Reported only: `simace validate`'s cross_trait_rC tests it (ADR 0030).
    rC_obs = safe_corrcoef(founders["C1"], founders["C2"])
    results["cross_trait_rC"] = _info(
        f"Cross-trait C correlation: {rC_obs:.4f} (configured {rC_param})",
        expected=rC_param,
        observed=float(rC_obs),
    )

    rE_param = params.get("rE", 0.0)
    rE_obs = safe_corrcoef(founders["E1"], founders["E2"])
    rE_ok = abs(rE_obs - rE_param) < 0.15 if not np.isnan(rE_obs) else rE_param == 0
    results["cross_trait_rE"] = _result(
        rE_ok,
        f"Cross-trait E correlation: {rE_obs:.4f} (expected: {rE_param})",
        expected=rE_param,
        observed=float(rE_obs),
    )

    # C inheritance: householdmates should share C. Without paternal C
    # transmission a household is one mother's children, so group every row,
    # recorded generation 0 included, by household_id with a stable sort +
    # reduceat. Under paternal transmission only full sibs share C, so the
    # trait's families are mating pairs instead (ADR 0025), which leaves out
    # generation 0: its fathers are unrecorded.
    nf_mask = mother_all != -1
    if nf_mask.any():
        mothers_nf = mother_all[nf_mask]
        order, starts = sorted_runs(mothers_nf)
        sizes = np.diff(np.r_[starts, len(mothers_nf)])
    by_household: tuple[np.ndarray, np.ndarray] | None = None
    for t in [1, 2]:
        col = f"C{t}"
        by_pair = transmission_coefficients(params, t)[1] != 0.0
        if by_pair and not nf_mask.any():
            results[f"c{t}_inheritance"] = _info("No non-founders to check C inheritance")
            continue
        if by_pair:
            father_nf = df["father"].to_numpy()[nf_mask].astype(np.int64)
            t_order, t_starts = sorted_runs(mothers_nf.astype(np.int64) * (int(father_nf.max()) + 1) + father_nf)
            vals = df[col].to_numpy()[nf_mask][t_order]
        else:
            if by_household is None:
                by_household = sorted_runs(df["household_id"].to_numpy())
            t_order, t_starts = by_household
            vals = df[col].to_numpy()[t_order]
        # nunique == 1 per family ⇔ group max equals group min (C is never NaN).
        c_shared = (np.maximum.reduceat(vals, t_starts) == np.minimum.reduceat(vals, t_starts)).mean()
        unit = "mating pairs" if by_pair else "households"
        results[f"c{t}_inheritance"] = _result(
            c_shared > 0.99,
            f"Proportion of {unit} with shared {col}: {c_shared:.4f}",
            proportion_shared=float(c_shared),
        )

    # E independence between siblings
    if nf_mask.any():
        multi_starts = starts[sizes >= 2]

        if len(multi_starts) > 10:
            # First two E1 values per multi-child mother (first 500 mothers,
            # ascending id — matching the groupby-index slice this replaced).
            multi_starts = multi_starts[:500]
            e1_sorted = df["E1"].to_numpy()[nf_mask][order]
            e1_pairs_arr = np.column_stack([e1_sorted[multi_starts], e1_sorted[multi_starts + 1]])
            e1_pairs_arr = e1_pairs_arr[:1000]

            if len(e1_pairs_arr) > 10:
                e1, e2 = e1_pairs_arr[:, 0], e1_pairs_arr[:, 1]
                e_corr = safe_corrcoef(e1, e2)
                # SE-based like the other correlation checks: a fixed 0.1 is
                # only ~2.2 SE at 500 pairs and failed ~2.5% of correct reps.
                tol = _corr_tolerance(0.0, len(e1_pairs_arr))
                results["e1_independence"] = _result(
                    abs(e_corr) < tol,
                    f"E1 correlation between siblings: {e_corr:.4f} (expected: ~0, tol: {tol:.4f})",
                    observed_correlation=float(e_corr),
                    tolerance=tol,
                    n_pairs=len(e1_pairs_arr),
                )
            else:
                results["e1_independence"] = _info("Not enough sibling pairs to check E independence")
        else:
            results["e1_independence"] = _info("Not enough sibling groups to check E independence")
    else:
        results["e1_independence"] = _info("No non-founders to check E independence")

    return results
