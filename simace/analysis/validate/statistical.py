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

from ._common import _info, _result
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

    Checks founder variances for A, C, E against configured values, total
    variance close to 1.0, cross-trait correlations (rA, rC, rE), C sharing
    within households, and E independence between siblings.

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

    # Total variances (reported, not asserted, under AM — total inflates with V_A).
    for t in [1, 2]:
        total = sum(results[f"variance_{c}{t}"]["observed"] for c in ["A", "C", "E"])
        if am_active[t]:
            results[f"total_variance_trait{t}"] = _info(
                f"Total variance trait {t}: {total:.4f} (AM-inflated; not asserted under AM)",
                observed=float(total),
            )
        else:
            results[f"total_variance_trait{t}"] = _result(
                abs(total - 1.0) < 0.15,
                f"Total variance trait {t}: {total:.4f} (expected: 1.0)",
                expected=1.0,
                observed=float(total),
            )

    # Cross-trait correlations
    for comp, expected, label in [("A", rA_param, "A"), ("C", rC_param, "C")]:
        obs = safe_corrcoef(founders[f"{comp}1"], founders[f"{comp}2"])
        ok = abs(obs - expected) < 0.15 if not np.isnan(obs) else expected == 0
        results[f"cross_trait_r{label}"] = _result(
            ok,
            f"Cross-trait {label} correlation: {obs:.4f} (expected: {expected})",
            expected=expected,
            observed=float(obs),
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

    # C inheritance: siblings should share C. Group children by mother via a
    # stable sort + reduceat (groups in ascending-mother order, rows in
    # original order within each group — matching pandas groupby semantics).
    nf_mask = mother_all != -1
    if nf_mask.any():
        mothers_nf = mother_all[nf_mask]
        order = np.argsort(mothers_nf, kind="stable")
        sorted_mothers = mothers_nf[order]
        starts = np.flatnonzero(np.r_[True, sorted_mothers[1:] != sorted_mothers[:-1]])
        sizes = np.diff(np.r_[starts, len(sorted_mothers)])
        for t in [1, 2]:
            col = f"C{t}"
            vals = df[col].to_numpy()[nf_mask][order]
            # nunique == 1 per family ⇔ group max equals group min (C is never NaN).
            c_shared = (np.maximum.reduceat(vals, starts) == np.minimum.reduceat(vals, starts)).mean()
            results[f"c{t}_inheritance"] = _result(
                c_shared > 0.99,
                f"Proportion of families with shared {col}: {c_shared:.4f}",
                proportion_shared=float(c_shared),
            )
    else:
        for t in [1, 2]:
            results[f"c{t}_inheritance"] = _result(True, "No non-founders to check C inheritance")

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
                results["e1_independence"] = _result(
                    abs(e_corr) < 0.1,
                    f"E1 correlation between siblings: {e_corr:.4f} (expected: ~0)",
                    observed_correlation=float(e_corr),
                )
            else:
                results["e1_independence"] = _result(True, "Not enough sibling pairs to check E independence")
        else:
            results["e1_independence"] = _result(True, "Not enough sibling groups to check E independence")
    else:
        results["e1_independence"] = _result(True, "No non-founders to check E independence")

    return results
