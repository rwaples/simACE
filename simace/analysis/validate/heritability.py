"""Heritability checks: MZ/DZ correlations, Falconer, parent-offspring."""

from typing import Any

import numpy as np
import pandas as pd

from simace.core.numerics import safe_corrcoef, safe_linregress

from ._common import (
    _DEFAULT_RNG_SEED,
    _MIN_PAIRS_FOR_CORR,
    _corr_se,
    _corr_tolerance,
    _extract_comp_vals,
    _result,
    _subsample_pairs,
)


def _midparent_regression(
    vals: np.ndarray, mother_idx: np.ndarray, father_idx: np.ndarray, offspring_idx: np.ndarray, label: str
) -> dict[str, Any]:
    """Run midparent-offspring regression and return result dict."""
    midparent = (vals[mother_idx] + vals[father_idx]) / 2
    offspring = vals[offspring_idx]
    reg = safe_linregress(midparent, offspring)
    if reg is not None:
        return {
            "slope": float(reg.slope),
            "intercept": float(reg.intercept),
            "r_squared": float(reg.rvalue**2),
            "details": f"Midparent-offspring {label} regression: slope={reg.slope:.4f}, R²={reg.rvalue**2:.4f}",
        }
    return {"details": f"Zero variance in midparent {label} values"}


def _validate_mz_correlations(
    df: pd.DataFrame,
    A_params: dict[int, float],
    comp_vals: dict[str, np.ndarray],
    id_to_idx: pd.Series,
    results: dict[str, Any],
) -> tuple[dict[int, float | None], int]:
    """Validate MZ twin correlations. Returns (mz_pheno_corr, n_mz_pairs)."""
    twins_df = df[df["twin"] != -1]
    twin_ids = twins_df["id"].values
    twin_partners = twins_df["twin"].values
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]

    mz_pheno_corr: dict[int, float | None] = {}
    if len(t1_arr) >= _MIN_PAIRS_FOR_CORR:
        idx1 = id_to_idx.reindex(t1_arr).values.astype(int)
        idx2 = id_to_idx.reindex(t2_arr).values.astype(int)

        for t in [1, 2]:
            col = f"A{t}"
            mz_v1, mz_v2 = comp_vals[col][idx1], comp_vals[col][idx2]
            mz_corr = safe_corrcoef(mz_v1, mz_v2)
            mz_ok = mz_corr > 0.99 if not np.isnan(mz_corr) else A_params[t] == 0
            results[f"mz_twin_{col}_correlation"] = _result(
                mz_ok,
                f"MZ twin {col} correlation: {mz_corr:.4f} (expected: 1.0)",
                expected=1.0,
                observed=float(mz_corr),
                n_pairs=len(t1_arr),
            )

            P1 = mz_v1 + comp_vals[f"C{t}"][idx1] + comp_vals[f"E{t}"][idx1]
            P2 = mz_v2 + comp_vals[f"C{t}"][idx2] + comp_vals[f"E{t}"][idx2]
            pheno_corr = safe_corrcoef(P1, P2)
            mz_pheno_corr[t] = pheno_corr
            results[f"mz_twin_liability{t}_correlation"] = {
                "observed": float(pheno_corr),
                "details": f"MZ twin liability{t} correlation: {pheno_corr:.4f}",
                "n_pairs": len(t1_arr),
            }
    else:
        for t in [1, 2]:
            results[f"mz_twin_A{t}_correlation"] = _result(
                True,
                f"Not enough MZ twin pairs ({len(t1_arr)}) to compute correlation",
            )
            mz_pheno_corr[t] = None

    return mz_pheno_corr, len(t1_arr)


def _validate_dz_correlations(
    params: dict[str, Any],
    A_params: dict[int, float],
    comp_vals: dict[str, np.ndarray],
    full_sib_pairs: tuple[np.ndarray, np.ndarray],
    results: dict[str, Any],
) -> tuple[dict[int, float | None], int]:
    """Validate DZ sibling correlations. Returns (dz_pheno_corr, n_dz_pairs)."""
    rng = np.random.default_rng(params.get("seed", _DEFAULT_RNG_SEED))
    idx1, idx2, n_dz_pairs = _subsample_pairs(full_sib_pairs[0], full_sib_pairs[1], rng)
    dz_pheno_corr: dict[int, float | None] = {}

    if n_dz_pairs >= _MIN_PAIRS_FOR_CORR:
        for t in [1, 2]:
            col = f"A{t}"
            dz_v1, dz_v2 = comp_vals[col][idx1], comp_vals[col][idx2]
            dz_corr = safe_corrcoef(dz_v1, dz_v2)
            expected_dz = 0.5
            dz_tol = _corr_tolerance(expected_dz, n_dz_pairs)
            if np.isnan(dz_corr):
                dz_ok = A_params[t] == 0
            else:
                dz_ok = abs(dz_corr - expected_dz) < dz_tol
            results[f"dz_sibling_{col}_correlation"] = _result(
                dz_ok,
                f"DZ sibling {col} correlation: {dz_corr:.4f} (expected: ~0.5, tol: {dz_tol:.4f})",
                expected=expected_dz,
                observed=float(dz_corr),
                n_pairs=n_dz_pairs,
            )

            P1 = dz_v1 + comp_vals[f"C{t}"][idx1] + comp_vals[f"E{t}"][idx1]
            P2 = dz_v2 + comp_vals[f"C{t}"][idx2] + comp_vals[f"E{t}"][idx2]
            pheno_corr = safe_corrcoef(P1, P2)
            dz_pheno_corr[t] = pheno_corr
            results[f"dz_sibling_liability{t}_correlation"] = {
                "observed": float(pheno_corr),
                "details": f"DZ sibling liability{t} correlation: {pheno_corr:.4f}",
                "n_pairs": n_dz_pairs,
            }

    if n_dz_pairs < _MIN_PAIRS_FOR_CORR:
        for t in [1, 2]:
            results[f"dz_sibling_A{t}_correlation"] = _result(
                True,
                f"Not enough DZ sibling pairs ({n_dz_pairs}) to compute correlation",
            )
            dz_pheno_corr[t] = None

    return dz_pheno_corr, n_dz_pairs


def _validate_falconer(
    A_params: dict[int, float],
    mz_pheno_corr: dict[int, float | None],
    dz_pheno_corr: dict[int, float | None],
    n_mz_pairs: int,
    n_dz_pairs: int,
    results: dict[str, Any],
) -> None:
    """Validate Falconer heritability estimates."""
    for t in [1, 2]:
        mz_c = mz_pheno_corr.get(t)
        dz_c = dz_pheno_corr.get(t)
        if mz_c is not None and dz_c is not None and not (np.isnan(mz_c) or np.isnan(dz_c)):
            falconer = 2 * (mz_c - dz_c)
            se_mz = _corr_se(mz_c, n_mz_pairs)
            se_dz = _corr_se(dz_c, n_dz_pairs)
            se_falconer = 2 * np.sqrt(se_mz**2 + se_dz**2)
            falconer_tol = max(4 * se_falconer, 0.05)
            results[f"falconer_estimate_trait{t}"] = _result(
                abs(falconer - A_params[t]) < falconer_tol,
                f"Falconer h²{chr(8320 + t)} = 2(r_MZ - r_DZ) = {falconer:.4f} "
                f"(expected: ~{A_params[t]}, tol: {falconer_tol:.4f})",
                expected=A_params[t],
                observed=float(falconer),
            )
        else:
            results[f"falconer_estimate_trait{t}"] = _result(
                True,
                "Cannot compute Falconer estimate without both MZ and DZ correlations",
            )


def _validate_parent_offspring(
    df: pd.DataFrame,
    comp_vals: dict[str, np.ndarray],
    id_to_idx: pd.Series,
    df_indexed: pd.DataFrame,
    results: dict[str, Any],
) -> None:
    """Validate parent-offspring regression."""
    non_founders = df[df["mother"] != -1]
    if len(non_founders) > 100:
        valid_offspring = non_founders[
            non_founders["mother"].isin(df_indexed.index) & non_founders["father"].isin(df_indexed.index)
        ]

        if len(valid_offspring) > 100:
            mother_idx = id_to_idx.reindex(valid_offspring["mother"]).values.astype(int)
            father_idx = id_to_idx.reindex(valid_offspring["father"]).values.astype(int)
            offspring_idx = id_to_idx.reindex(valid_offspring["id"]).values.astype(int)

            for t in [1, 2]:
                results[f"parent_offspring_A{t}_regression"] = _midparent_regression(
                    comp_vals[f"A{t}"],
                    mother_idx,
                    father_idx,
                    offspring_idx,
                    f"A{t}",
                )
                P_vals = comp_vals[f"A{t}"] + comp_vals[f"C{t}"] + comp_vals[f"E{t}"]
                results[f"parent_offspring_liability{t}_regression"] = _midparent_regression(
                    P_vals,
                    mother_idx,
                    father_idx,
                    offspring_idx,
                    f"liability{t}",
                )
        else:
            for t in [1, 2]:
                results[f"parent_offspring_A{t}_regression"] = {
                    "details": "Not enough offspring with both parents in data"
                }
                results[f"parent_offspring_liability{t}_regression"] = {
                    "details": "Not enough offspring with both parents in data"
                }
    else:
        for t in [1, 2]:
            results[f"parent_offspring_A{t}_regression"] = {"details": "Not enough non-founders for regression"}
            results[f"parent_offspring_liability{t}_regression"] = {"details": "Not enough non-founders for regression"}


def validate_heritability(
    df: pd.DataFrame,
    params: dict[str, Any],
    df_indexed: pd.DataFrame,
    sibling_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Validate heritability estimates for two-trait simulation.

    Computes MZ twin and DZ sibling liability correlations, Falconer
    heritability estimates ``h² = 2(r_MZ - r_DZ)``, and midparent-offspring
    regressions, comparing each to expected values derived from the
    configured A parameters.

    Args:
        df: Pedigree DataFrame.
        params: Scenario parameters; requires keys ``A1``, ``A2``, ``seed``.
        df_indexed: Pedigree DataFrame indexed by ``id``.
        sibling_pairs: Dict with keys ``FS``, ``MHS``, ``PHS`` mapping to
            ``(idx1, idx2)`` row-index arrays.

    Returns:
        Dict of check-name to result dicts, including MZ/DZ correlations,
        Falconer estimates, and parent-offspring regression slopes.
    """
    results: dict[str, Any] = {}
    A_params = {1: params["A1"], 2: params["A2"]}
    comp_vals = _extract_comp_vals(df_indexed)
    id_to_idx = pd.Series(np.arange(len(df_indexed)), index=df_indexed.index)

    mz_pheno_corr, n_mz_pairs = _validate_mz_correlations(
        df,
        A_params,
        comp_vals,
        id_to_idx,
        results,
    )
    dz_pheno_corr, n_dz_pairs = _validate_dz_correlations(
        params,
        A_params,
        comp_vals,
        sibling_pairs["FS"],
        results,
    )
    _validate_falconer(
        A_params,
        mz_pheno_corr,
        dz_pheno_corr,
        n_mz_pairs,
        n_dz_pairs,
        results,
    )
    _validate_parent_offspring(df, comp_vals, id_to_idx, df_indexed, results)

    return results
