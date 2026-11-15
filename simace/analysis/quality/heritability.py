"""Heritability checks: MZ/DZ correlations, Falconer, parent-offspring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pedigree_graph import RELATIONSHIPS
from pg_phenotype import Trait

from simace.analysis.midparent import midparent_offspring
from simace.core.numerics import safe_corrcoef

from ._common import _MIN_PAIRS_FOR_CORR, _corr_se, _extract_comp_vals, category_cell, pair_correlation
from .am_relatedness import am_relatedness_mode, observed_mate_correlations, resolve_expected_a_corr
from .check import FamilyResult, PairCorrelation, QualityCheck

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments
    from pg_phenotype import Pedigree

    from simace.core.pedigree_arrays import PedigreeArrays
    from simace.simulation.params import Assortment, SimulationParams

    from ._common import MateCorrelations

# The midparent-offspring regressions need more than this many offspring.
_MIN_PO_OFFSPRING = 100


@dataclass(frozen=True)
class MidparentRegression:
    """The midparent-offspring regression of one value over the pedigree's offspring with both parents."""

    slope: float
    intercept: float | None
    r_squared: float | None


@dataclass(frozen=True)
class HeritabilityTrait:
    """One trait's heritability measurements; a correlation is ``None`` with fewer than ten pairs.

    Attributes:
        mz_A_corr: The MZ twin A correlation.
        mz_liability_corr: The MZ twin liability correlation.
        dz_A_corr: The full-sib A correlation, with its AM-aware expectation.
        dz_liability_corr: The full-sib liability correlation.
        falconer: ``2(r_MZ - r_DZ)`` on liability; ``None`` without both correlations.
        po_A: The midparent-offspring regression of A; ``None`` with too few
            offspring or no midparent variance.
        po_liability: The same for liability.
    """

    mz_A_corr: PairCorrelation | None
    mz_liability_corr: PairCorrelation | None
    dz_A_corr: PairCorrelation | None
    dz_liability_corr: PairCorrelation | None
    falconer: float | None
    po_A: MidparentRegression | None
    po_liability: MidparentRegression | None


@dataclass(frozen=True)
class HeritabilityMeasurements:
    """What the heritability family measures, ``traits[t - 1]`` per trait."""

    traits: tuple[HeritabilityTrait, HeritabilityTrait]


def _midparent_regression(pedigree: Pedigree, vals: np.ndarray) -> MidparentRegression | None:
    """The midparent-offspring regression of *vals* over *pedigree*, or None with too few offspring or no variance."""
    fit = midparent_offspring(pedigree, Trait(vals, "continuous")).pooled.midparent
    assert fit is not None, "the midparent line was requested"
    if fit.n <= _MIN_PO_OFFSPRING or fit.slope is None:
        return None
    # r2 is None, with the slope set, when every offspring value is equal.
    return MidparentRegression(slope=fit.slope, intercept=fit.intercept, r_squared=fit.r2)


def _mz_correlations(
    df: pd.DataFrame | pl.DataFrame,
    A_params: dict[int, float],
    comp_vals: dict[str, np.ndarray],
    ped: PedigreeArrays,
    t: int,
) -> tuple[PairCorrelation | None, PairCorrelation | None, QualityCheck | None]:
    """Trait ``t``'s MZ twin A and liability correlations and the A check, or Nones with fewer than ten pairs."""
    twin_all = df["twin"].to_numpy()
    twin_mask = twin_all != -1
    twin_ids = df["id"].to_numpy()[twin_mask]
    twin_partners = twin_all[twin_mask]
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]
    n_pairs = len(t1_arr)
    if n_pairs < _MIN_PAIRS_FOR_CORR:
        return None, None, None

    idx1 = ped.positions(t1_arr)
    idx2 = ped.positions(t2_arr)
    col = f"A{t}"
    mz_v1, mz_v2 = comp_vals[col][idx1], comp_vals[col][idx2]
    mz_corr = safe_corrcoef(mz_v1, mz_v2)
    # MZ twins share identical A, so the A-component correlation equals
    # the relatedness 2*kinship == 1.0; 0.01 is the near-equality band.
    expected_mz = 2.0 * RELATIONSHIPS["MZ"].nominal_kinship
    mz_ok = mz_corr > expected_mz - 0.01 if not np.isnan(mz_corr) else A_params[t] == 0
    check = QualityCheck(
        f"mz_twin_{col}_correlation",
        mz_ok,
        f"MZ twin {col} correlation: {mz_corr:.4f} (expected: {expected_mz})",
        observed=float(mz_corr),
        expected=expected_mz,
    )
    P1 = mz_v1 + comp_vals[f"C{t}"][idx1] + comp_vals[f"E{t}"][idx1]
    P2 = mz_v2 + comp_vals[f"C{t}"][idx2] + comp_vals[f"E{t}"][idx2]
    a_corr = PairCorrelation(observed=float(mz_corr), n_pairs=n_pairs, expected=expected_mz)
    liability = PairCorrelation(observed=float(safe_corrcoef(P1, P2)), n_pairs=n_pairs)
    return a_corr, liability, check


def _dz_correlations(
    mates: MateCorrelations, assortment: Assortment, sibling_moments: RelationshipMoments, t: int
) -> tuple[PairCorrelation | None, PairCorrelation | None]:
    """Trait ``t``'s full-sib A and liability correlations, or Nones with fewer than ten pairs.

    The full-sib A-component correlation carries its expectation,
    ``2·kinship`` (0.5) under random mating and ``(1+mu_A)/2`` under
    single-trait assortative mating (see :mod:`.am_relatedness`), none under
    both-trait AM. It is not asserted: `simace validate`'s
    ``mendelian_residual`` tests the inheritance (ADR 0029). The correlations
    are exact over every full-sib pair (ADR 0022).
    """
    full_sibs = category_cell(sibling_moments, "FS")
    n_pairs = int(full_sibs.counts)
    if n_pairs < _MIN_PAIRS_FOR_CORR:
        return None, None
    expected_dz, mate_terms = resolve_expected_a_corr(
        mates, assortment, t, "FS", 2.0 * RELATIONSHIPS["FS"].nominal_kinship
    )
    a_corr = PairCorrelation(
        observed=float(pair_correlation(full_sibs, f"A{t}")),
        n_pairs=n_pairs,
        expected=None if expected_dz is None else float(expected_dz),
        mate_terms=mate_terms,
    )
    return a_corr, PairCorrelation(observed=float(pair_correlation(full_sibs, f"P{t}")), n_pairs=n_pairs)


def _falconer_expected(
    mates: MateCorrelations,
    assortment: Assortment,
    comp_vals: dict[str, np.ndarray],
    A_params: dict[int, float],
    t: int,
) -> tuple[float | None, str]:
    """Resolve the expected Falconer estimate for trait ``t``, and how it was resolved.

    Under random mating Falconer estimates the configured ``A``. Under
    single-trait assortative mating it is biased downward to
    ``Var(A)·(1 − mu_A)/V_P`` (the ``1 − mu_A`` factor cancels the AM variance
    inflation, leaving the heritability shrunk by the inflated liability
    variance) — a known AM bias the check asserts the simulator reproduces.
    Both-trait AM has no expectation (cross-trait bias not modelled).
    """
    mode = am_relatedness_mode(assortment, t)
    if mode == "none":
        return A_params[t], f"expected ~{A_params[t]}"
    if mode == "bivariate":
        return None, ""
    mu_a, _r_ho, _n = observed_mate_correlations(mates, t)
    v_a = float(np.var(comp_vals[f"A{t}"]))
    v_p = float(np.var(comp_vals[f"A{t}"] + comp_vals[f"C{t}"] + comp_vals[f"E{t}"]))
    if v_p <= 0:
        return A_params[t], f"expected ~{A_params[t]}"
    expected = v_a * (1.0 - mu_a) / v_p
    return expected, f"AM-biased expected {expected:.4f} = Var(A)(1-mu_A)/V_P"


def _falconer(
    mates: MateCorrelations,
    assortment: Assortment,
    comp_vals: dict[str, np.ndarray],
    A_params: dict[int, float],
    mz: PairCorrelation | None,
    dz: PairCorrelation | None,
    t: int,
) -> tuple[float | None, QualityCheck | None]:
    """Trait ``t``'s Falconer estimate and its check (AM-aware expectation), each None where it does not apply."""
    if mz is None or dz is None or np.isnan(mz.observed) or np.isnan(dz.observed):
        return None, None
    falconer = 2 * (mz.observed - dz.observed)
    expected, label = _falconer_expected(mates, assortment, comp_vals, A_params, t)
    if expected is None:
        return float(falconer), None
    se_falconer = 2 * np.sqrt(_corr_se(mz.observed, mz.n_pairs) ** 2 + _corr_se(dz.observed, dz.n_pairs) ** 2)
    falconer_tol = max(4 * se_falconer, 0.05)
    check = QualityCheck(
        f"falconer_estimate_trait{t}",
        abs(falconer - expected) < falconer_tol,
        f"Falconer h²{chr(8320 + t)} = 2(r_MZ - r_DZ) = {falconer:.4f} ({label}, tol: {falconer_tol:.4f})",
        observed=float(falconer),
        expected=float(expected),
    )
    return float(falconer), check


def validate_heritability(
    df: pd.DataFrame | pl.DataFrame,
    sim: SimulationParams,
    ped: PedigreeArrays,
    pedigree: Pedigree,
    sibling_moments: RelationshipMoments,
    mates: MateCorrelations,
) -> FamilyResult[HeritabilityMeasurements]:
    """Measure heritability for two-trait simulation, and check what has a closed form.

    Measures MZ twin and DZ sibling correlations, Falconer heritability
    estimates ``h² = 2(r_MZ - r_DZ)``, and midparent-offspring regressions.
    Checks the MZ A correlation (1) and the Falconer estimate against the
    configured A (AM-biased under single-trait AM).

    Args:
        df: Pedigree DataFrame.
        sim: The rep's parameters.
        ped: The same pedigree as id-addressable arrays.
        pedigree: :func:`~simace.analysis.mate_correlation.mate_pedigree` of
            ``df``, for the midparent-offspring regressions.
        sibling_moments: Relationship moments of ``df`` over ``FS``, ``MHS``
            and ``PHS`` with the ``A{t}``, ``C{t}`` and ``P{t}`` columns
            (:func:`~simace.analysis.quality._common.sibling_moments`).
        mates: The pedigree's mate correlations, for the AM-aware expected
            values (:func:`~simace.analysis.quality._common.mate_correlations`).
    """
    A_params = {1: sim.vc.A[0], 2: sim.vc.A[1]}
    comp_vals = _extract_comp_vals(ped)

    mz = {t: _mz_correlations(df, A_params, comp_vals, ped, t) for t in (1, 2)}
    dz = {t: _dz_correlations(mates, sim.assortment, sibling_moments, t) for t in (1, 2)}
    falconer = {t: _falconer(mates, sim.assortment, comp_vals, A_params, mz[t][1], dz[t][1], t) for t in (1, 2)}
    traits = tuple(
        HeritabilityTrait(
            mz_A_corr=mz[t][0],
            mz_liability_corr=mz[t][1],
            dz_A_corr=dz[t][0],
            dz_liability_corr=dz[t][1],
            falconer=falconer[t][0],
            po_A=_midparent_regression(pedigree, comp_vals[f"A{t}"]),
            po_liability=_midparent_regression(pedigree, comp_vals[f"A{t}"] + comp_vals[f"C{t}"] + comp_vals[f"E{t}"]),
        )
        for t in (1, 2)
    )
    checks = [mz[t][2] for t in (1, 2)] + [falconer[t][1] for t in (1, 2)]
    return FamilyResult(
        HeritabilityMeasurements(traits=(traits[0], traits[1])), tuple(c for c in checks if c is not None)
    )
