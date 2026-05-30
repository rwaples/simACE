"""Variance-component statistical checks (founder variances, correlations)."""

from typing import Any

import numpy as np
import pandas as pd

from simace.core.numerics import safe_corrcoef

from ._common import _result


def _check_variance(founders: pd.DataFrame, col: str, expected: float, tol: float = 0.1) -> dict[str, Any]:
    """Check that the variance of `col` in founders is close to `expected`."""
    var = founders[col].var()
    return _result(
        abs(var - expected) < tol,
        f"Var({col}) in founders: {var:.4f} (expected: {expected})",
        expected=expected,
        observed=float(var),
    )


def validate_statistical(df: pd.DataFrame, params: dict[str, Any], df_indexed: pd.DataFrame) -> dict[str, Any]:
    """Validate statistical properties of variance components for two traits.

    Checks founder variances for A, C, E against configured values, total
    variance close to 1.0, cross-trait correlations (rA, rC, rE), C sharing
    within households, and E independence between siblings.

    Args:
        df: Pedigree DataFrame with variance-component columns A1, C1, E1,
            A2, C2, E2.
        params: Scenario parameters; requires keys ``A1``, ``C1``, ``E1``,
            ``A2``, ``C2``, ``E2``, ``rA``, ``rC``.
        df_indexed: Pedigree DataFrame indexed by ``id``.

    Returns:
        Dict of check-name to result dicts.
    """
    results = {}

    rA_param = params.get("rA", 0)
    rC_param = params.get("rC", 0)

    founders = df[df["mother"] == -1]

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

    # Variance checks for both traits
    for t in [1, 2]:
        for comp in ["A", "C", "E"]:
            col = f"{comp}{t}"
            results[f"variance_{col}"] = _check_variance(founders, col, _resolve_founder_val(params[col]))

    # Total variances
    for t in [1, 2]:
        total = sum(results[f"variance_{c}{t}"]["observed"] for c in ["A", "C", "E"])
        results[f"total_variance_trait{t}"] = _result(
            abs(total - 1.0) < 0.15,
            f"Total variance trait {t}: {total:.4f} (expected: 1.0)",
            expected=1.0,
            observed=float(total),
        )

    # Cross-trait correlations
    for comp, expected, label in [("A", rA_param, "A"), ("C", rC_param, "C")]:
        obs = safe_corrcoef(founders[f"{comp}1"].values, founders[f"{comp}2"].values)
        ok = abs(obs - expected) < 0.15 if not np.isnan(obs) else expected == 0
        results[f"cross_trait_r{label}"] = _result(
            ok,
            f"Cross-trait {label} correlation: {obs:.4f} (expected: {expected})",
            expected=expected,
            observed=float(obs),
        )

    rE_param = params.get("rE", 0.0)
    rE_obs = safe_corrcoef(founders["E1"].values, founders["E2"].values)
    rE_ok = abs(rE_obs - rE_param) < 0.15 if not np.isnan(rE_obs) else rE_param == 0
    results["cross_trait_rE"] = _result(
        rE_ok,
        f"Cross-trait E correlation: {rE_obs:.4f} (expected: {rE_param})",
        expected=rE_param,
        observed=float(rE_obs),
    )

    # C inheritance: siblings should share C
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 0:
        for t in [1, 2]:
            col = f"C{t}"
            c_by_mother = non_founders.groupby("mother")[col].nunique()
            c_shared = (c_by_mother == 1).mean()
            results[f"c{t}_inheritance"] = _result(
                c_shared > 0.99,
                f"Proportion of families with shared {col}: {c_shared:.4f}",
                proportion_shared=float(c_shared),
            )
    else:
        for t in [1, 2]:
            results[f"c{t}_inheritance"] = _result(True, "No non-founders to check C inheritance")

    # E independence between siblings
    if len(non_founders) > 0:
        fam_sizes = non_founders.groupby("mother").size()
        multi_child_mothers = fam_sizes[fam_sizes >= 2].index

        if len(multi_child_mothers) > 10:
            # Vectorized: get first two E1 values per mother via groupby
            multi_child = non_founders[non_founders["mother"].isin(multi_child_mothers[:500])]
            grouped = multi_child.groupby("mother")["E1"]
            first = grouped.nth(0).values
            second = grouped.nth(1).values
            # nth returns NaN for groups with < 2 members; both arrays aligned by group
            valid = ~(np.isnan(first) | np.isnan(second))
            e1_pairs_arr = np.column_stack([first[valid], second[valid]])
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
