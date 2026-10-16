"""Heritability checks: MZ/DZ correlations, Falconer, parent-offspring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pedigree_graph import RELATIONSHIPS
from pg_phenotype import Trait
from pg_phenotype.regression import parent_offspring

from simace.core.numerics import safe_corrcoef

from ._common import (
    _MIN_PAIRS_FOR_CORR,
    _corr_se,
    _corr_tolerance,
    _extract_comp_vals,
    _info,
    _result,
    category_cell,
    pair_correlation,
)
from .am_relatedness import am_relatedness_mode, observed_mate_correlations, resolve_expected_a_corr

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments
    from pg_phenotype import Pedigree

    from simace.core.pedigree_arrays import PedigreeArrays

    from ._common import MateCorrelations

# The midparent-offspring regressions need more than this many offspring.
_MIN_PO_OFFSPRING = 100


def _midparent_regression(pedigree: Pedigree, vals: np.ndarray, label: str) -> dict[str, Any]:
    """Run midparent-offspring regression of *vals* over *pedigree* and return result dict."""
    fit = parent_offspring(pedigree, Trait(vals, "continuous"), lines="midparent", inference=False).pooled.midparent
    assert fit is not None, "the midparent line was requested"
    if fit.n <= _MIN_PO_OFFSPRING:
        return _info("Not enough offspring with both parents in data")
    if fit.slope is not None:
        # r2 is None, with the slope set, when every offspring value is equal.
        r2 = "n/a" if fit.r2 is None else f"{fit.r2:.4f}"
        return _info(
            f"Midparent-offspring {label} regression: slope={fit.slope:.4f}, R²={r2}",
            slope=fit.slope,
            intercept=fit.intercept,
            r_squared=fit.r2,
        )
    return _info(f"Zero variance in midparent {label} values")


def _validate_mz_correlations(
    df: pd.DataFrame | pl.DataFrame,
    A_params: dict[int, float],
    comp_vals: dict[str, np.ndarray],
    ped: PedigreeArrays,
    results: dict[str, Any],
) -> tuple[dict[int, float | None], int]:
    """Validate MZ twin correlations. Returns (mz_pheno_corr, n_mz_pairs)."""
    twin_all = df["twin"].to_numpy()
    twin_mask = twin_all != -1
    twin_ids = df["id"].to_numpy()[twin_mask]
    twin_partners = twin_all[twin_mask]
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]

    mz_pheno_corr: dict[int, float | None] = {}
    if len(t1_arr) >= _MIN_PAIRS_FOR_CORR:
        idx1 = ped.positions(t1_arr)
        idx2 = ped.positions(t2_arr)

        for t in [1, 2]:
            col = f"A{t}"
            mz_v1, mz_v2 = comp_vals[col][idx1], comp_vals[col][idx2]
            mz_corr = safe_corrcoef(mz_v1, mz_v2)
            # MZ twins share identical A, so the A-component correlation equals
            # the relatedness 2*kinship == 1.0; 0.01 is the near-equality band.
            expected_mz = 2.0 * RELATIONSHIPS["MZ"].nominal_kinship
            mz_ok = mz_corr > expected_mz - 0.01 if not np.isnan(mz_corr) else A_params[t] == 0
            results[f"mz_twin_{col}_correlation"] = _result(
                mz_ok,
                f"MZ twin {col} correlation: {mz_corr:.4f} (expected: {expected_mz})",
                expected=expected_mz,
                observed=float(mz_corr),
                n_pairs=len(t1_arr),
            )

            P1 = mz_v1 + comp_vals[f"C{t}"][idx1] + comp_vals[f"E{t}"][idx1]
            P2 = mz_v2 + comp_vals[f"C{t}"][idx2] + comp_vals[f"E{t}"][idx2]
            pheno_corr = safe_corrcoef(P1, P2)
            mz_pheno_corr[t] = pheno_corr
            results[f"mz_twin_liability{t}_correlation"] = _info(
                f"MZ twin liability{t} correlation: {pheno_corr:.4f}",
                observed=float(pheno_corr),
                n_pairs=len(t1_arr),
            )
    else:
        for t in [1, 2]:
            results[f"mz_twin_A{t}_correlation"] = _result(
                True,
                f"Not enough MZ twin pairs ({len(t1_arr)}) to compute correlation",
            )
            mz_pheno_corr[t] = None

    return mz_pheno_corr, len(t1_arr)


def _validate_dz_correlations(
    mates: MateCorrelations,
    params: dict[str, Any],
    A_params: dict[int, float],
    sibling_moments: RelationshipMoments,
    results: dict[str, Any],
) -> tuple[dict[int, float | None], int]:
    """Validate DZ sibling correlations. Returns (dz_pheno_corr, n_dz_pairs).

    The full-sib A-component correlation is ``2·kinship`` (0.5) under random
    mating; under single-trait assortative mating it inflates to ``(1+mu_A)/2``
    (see :mod:`.am_relatedness`). Both-trait AM skips the scored check. The
    correlations are exact over every full-sib pair (ADR 0022).
    """
    full_sibs = category_cell(sibling_moments, "FS")
    n_dz_pairs = int(full_sibs.counts)
    dz_pheno_corr: dict[int, float | None] = {}

    if n_dz_pairs >= _MIN_PAIRS_FOR_CORR:
        for t in [1, 2]:
            col = f"A{t}"
            dz_corr = pair_correlation(full_sibs, col)
            # Full-sib (DZ) A correlation: 2*kinship under random mating,
            # AM-inflated to (1+mu_A)/2 under single-trait assortment.
            expected_dz, skip, info = resolve_expected_a_corr(
                mates, params, t, "FS", 2.0 * RELATIONSHIPS["FS"].nominal_kinship
            )
            if expected_dz is None:
                # Reported, not asserted: no single-trait formula under {skip}.
                results[f"dz_sibling_{col}_correlation"] = _info(
                    f"DZ sibling {col} correlation: {dz_corr:.4f} (not asserted — {skip})",
                    observed=float(dz_corr),
                    n_pairs=n_dz_pairs,
                )
            else:
                dz_tol = _corr_tolerance(expected_dz, n_dz_pairs)
                if np.isnan(dz_corr):
                    dz_ok = A_params[t] == 0
                else:
                    dz_ok = abs(dz_corr - expected_dz) < dz_tol
                results[f"dz_sibling_{col}_correlation"] = _result(
                    dz_ok,
                    f"DZ sibling {col} correlation: {dz_corr:.4f} (expected: ~{expected_dz:.4f}, tol: {dz_tol:.4f})",
                    expected=float(expected_dz),
                    observed=float(dz_corr),
                    n_pairs=n_dz_pairs,
                    **info,
                )

            pheno_corr = pair_correlation(full_sibs, f"P{t}")
            dz_pheno_corr[t] = pheno_corr
            results[f"dz_sibling_liability{t}_correlation"] = _info(
                f"DZ sibling liability{t} correlation: {pheno_corr:.4f}",
                observed=float(pheno_corr),
                n_pairs=n_dz_pairs,
            )

    if n_dz_pairs < _MIN_PAIRS_FOR_CORR:
        for t in [1, 2]:
            results[f"dz_sibling_A{t}_correlation"] = _result(
                True,
                f"Not enough DZ sibling pairs ({n_dz_pairs}) to compute correlation",
            )
            dz_pheno_corr[t] = None

    return dz_pheno_corr, n_dz_pairs


def _falconer_expected(
    mates: MateCorrelations,
    params: dict[str, Any],
    comp_vals: dict[str, np.ndarray],
    A_params: dict[int, float],
    t: int,
) -> tuple[float | None, str | None, str, dict[str, Any]]:
    """Resolve the expected Falconer estimate for trait ``t``.

    Returns ``(expected, skip_reason, label, info)``. Under random mating
    Falconer estimates the configured ``A``. Under single-trait assortative
    mating it is biased downward to ``Var(A)·(1 − mu_A)/V_P`` (the ``1 − mu_A``
    factor cancels the AM variance inflation, leaving the heritability shrunk by
    the inflated liability variance) — a known AM bias the check now asserts the
    simulator reproduces. Both-trait AM skips (cross-trait bias not modelled).
    """
    mode = am_relatedness_mode(params, t)
    if mode == "none":
        return A_params[t], None, f"expected ~{A_params[t]}", {}
    if mode == "bivariate":
        return None, "both-trait AM active (cross-trait Falconer bias not modelled)", "", {}
    mu_a, _r_ho, _n = observed_mate_correlations(mates, t)
    v_a = float(np.var(comp_vals[f"A{t}"]))
    v_p = float(np.var(comp_vals[f"A{t}"] + comp_vals[f"C{t}"] + comp_vals[f"E{t}"]))
    if v_p <= 0:
        return A_params[t], None, f"expected ~{A_params[t]}", {}
    expected = v_a * (1.0 - mu_a) / v_p
    info = {"mu_A": mu_a, "var_A": v_a, "var_P": v_p}
    return expected, None, f"AM-biased expected {expected:.4f} = Var(A)(1-mu_A)/V_P", info


def _validate_falconer(
    mates: MateCorrelations,
    params: dict[str, Any],
    comp_vals: dict[str, np.ndarray],
    A_params: dict[int, float],
    mz_pheno_corr: dict[int, float | None],
    dz_pheno_corr: dict[int, float | None],
    n_mz_pairs: int,
    n_dz_pairs: int,
    results: dict[str, Any],
) -> None:
    """Validate Falconer heritability estimates (AM-aware expected value)."""
    for t in [1, 2]:
        mz_c = mz_pheno_corr.get(t)
        dz_c = dz_pheno_corr.get(t)
        if mz_c is None or dz_c is None or np.isnan(mz_c) or np.isnan(dz_c):
            results[f"falconer_estimate_trait{t}"] = _result(
                True,
                "Cannot compute Falconer estimate without both MZ and DZ correlations",
            )
            continue

        falconer = 2 * (mz_c - dz_c)
        expected, skip, label, info = _falconer_expected(mates, params, comp_vals, A_params, t)
        if expected is None:
            # Reported, not asserted: cross-trait Falconer bias not modelled.
            results[f"falconer_estimate_trait{t}"] = _info(
                f"Falconer h²{chr(8320 + t)} = {falconer:.4f} (not asserted — {skip})",
                observed=float(falconer),
            )
            continue

        se_mz = _corr_se(mz_c, n_mz_pairs)
        se_dz = _corr_se(dz_c, n_dz_pairs)
        se_falconer = 2 * np.sqrt(se_mz**2 + se_dz**2)
        falconer_tol = max(4 * se_falconer, 0.05)
        results[f"falconer_estimate_trait{t}"] = _result(
            abs(falconer - expected) < falconer_tol,
            f"Falconer h²{chr(8320 + t)} = 2(r_MZ - r_DZ) = {falconer:.4f} ({label}, tol: {falconer_tol:.4f})",
            expected=float(expected),
            observed=float(falconer),
            **info,
        )


def _validate_parent_offspring(
    comp_vals: dict[str, np.ndarray],
    pedigree: Pedigree,
    results: dict[str, Any],
) -> None:
    """Validate parent-offspring regression."""
    for t in [1, 2]:
        results[f"parent_offspring_A{t}_regression"] = _midparent_regression(pedigree, comp_vals[f"A{t}"], f"A{t}")
        P_vals = comp_vals[f"A{t}"] + comp_vals[f"C{t}"] + comp_vals[f"E{t}"]
        results[f"parent_offspring_liability{t}_regression"] = _midparent_regression(pedigree, P_vals, f"liability{t}")


def validate_heritability(
    df: pd.DataFrame | pl.DataFrame,
    params: dict[str, Any],
    ped: PedigreeArrays,
    pedigree: Pedigree,
    sibling_moments: RelationshipMoments,
    mates: MateCorrelations,
) -> dict[str, Any]:
    """Validate heritability estimates for two-trait simulation.

    Computes MZ twin and DZ sibling liability correlations, Falconer
    heritability estimates ``h² = 2(r_MZ - r_DZ)``, and midparent-offspring
    regressions, comparing each to expected values derived from the
    configured A parameters.

    Args:
        df: Pedigree DataFrame.
        params: Scenario parameters; requires keys ``A1``, ``A2``.
        ped: The same pedigree as id-addressable arrays.
        pedigree: :func:`~simace.analysis.mate_correlation.mate_pedigree` of
            ``df``, for the midparent-offspring regressions.
        sibling_moments: Relationship moments of ``df`` over ``FS``, ``MHS``
            and ``PHS`` with the ``A{t}``, ``C{t}`` and ``P{t}`` columns
            (:func:`~simace.analysis.validate._common.sibling_moments`).
        mates: The pedigree's mate correlations, for the AM-aware expected
            values (:func:`~simace.analysis.validate._common.mate_correlations`).

    Returns:
        Dict of check-name to result dicts, including MZ/DZ correlations,
        Falconer estimates, and parent-offspring regression slopes.
    """
    results: dict[str, Any] = {}
    A_params = {1: params["A1"], 2: params["A2"]}
    comp_vals = _extract_comp_vals(ped)

    mz_pheno_corr, n_mz_pairs = _validate_mz_correlations(
        df,
        A_params,
        comp_vals,
        ped,
        results,
    )
    dz_pheno_corr, n_dz_pairs = _validate_dz_correlations(
        mates,
        params,
        A_params,
        sibling_moments,
        results,
    )
    _validate_falconer(
        mates,
        params,
        comp_vals,
        A_params,
        mz_pheno_corr,
        dz_pheno_corr,
        n_mz_pairs,
        n_dz_pairs,
        results,
    )
    _validate_parent_offspring(comp_vals, pedigree, results)

    return results
