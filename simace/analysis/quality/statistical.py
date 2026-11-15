"""Variance-component statistical checks (founder variances, correlations).

Library-agnostic by design (ADR 0015): columns come out through
``.to_numpy()`` and all grouping/slicing runs in NumPy, so any frame exposing
that interface yields identical results. polars is the canonical caller; the
NumPy boundary is a deliberate contract, not a migration leftover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simace.core.numerics import safe_corrcoef

from ._common import _corr_tolerance, sorted_runs
from .am_relatedness import am_relatedness_mode
from .check import FamilyResult, QualityCheck

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import polars as pl

    from simace.simulation.params import SimulationParams


@dataclass(frozen=True)
class FounderVariances:
    """One trait's component and total variances in the recorded founders."""

    var_A: float
    var_C: float
    var_E: float
    var_liability: float


@dataclass(frozen=True)
class StatisticalMeasurements:
    """What the statistical family measures, all in the recorded founders.

    Attributes:
        traits: Each trait's variances, ``traits[t - 1]``.
        rA: The cross-trait A correlation (NaN where undefined).
        rC: The cross-trait C correlation.
        rE: The cross-trait E correlation.
    """

    traits: tuple[FounderVariances, FounderVariances]
    rA: float
    rC: float
    rE: float


def _founder_var(founders: Mapping[str, np.ndarray], col: str) -> float:
    """Variance of ``col`` in founders, accumulated in float64 (matching pandas ``Series.var`` on float32 columns)."""
    return float(np.var(founders[col], ddof=1, dtype=np.float64))


def _check_variance(var: float, col: str, expected: float, tol: float = 0.1) -> QualityCheck:
    """Check that the founders' variance ``var`` of ``col`` is close to ``expected``."""
    return QualityCheck(
        f"variance_{col}",
        abs(var - expected) < tol,
        f"Var({col}) in founders: {var:.4f} (expected: {expected})",
        observed=var,
        expected=expected,
    )


def validate_statistical(
    df: pd.DataFrame | pl.DataFrame, sim: SimulationParams
) -> FamilyResult[StatisticalMeasurements]:
    """Validate statistical properties of variance components for two traits.

    Checks founder variances for A, C, E against configured values,
    cross-trait correlations rA and rE, C sharing within households, and E
    independence between siblings. Under assortative mating on a trait, its
    ``Var(A)`` and the cross-trait ``rA`` are measured without being
    checked. The total variance and the cross-trait ``rC`` are measured
    only: `simace validate`'s ``total_variance`` and ``cross_trait_rC`` test
    them (ADR 0030).

    Args:
        df: Pedigree DataFrame with variance-component columns A1, C1, E1,
            A2, C2, E2.
        sim: The rep's parameters.

    Returns:
        The checks, and the founders' variances and cross-trait correlations.
    """
    checks = []

    vc = sim.vc
    rA_param = vc.rA

    mother_all = df["mother"].to_numpy()
    founder_mask = mother_all == -1
    comp_cols = [f"{c}{t}" for c in ("A", "C", "E") for t in (1, 2)]
    founders = {c: df[c].to_numpy()[founder_mask] for c in comp_cols}

    # The configured value of each component in the founders' (first recorded) generation.
    configured = {f"A{t}": vc.A[t - 1] for t in (1, 2)} | {f"C{t}": vc.C[t - 1] for t in (1, 2)}
    for t in (1, 2):
        E = vc.E[t - 1]
        configured[f"E{t}"] = vc.E_per_gen(t, sim.span.G_sim)[sim.span.burnin] if isinstance(E, dict) else E

    # Variance checks for both traits. Under assortative mating the additive
    # variance inflates across generations (Bulmer), so the recorded "founders"
    # (first recorded generation) no longer carry the configured A; the
    # am_equilibrium check validates that inflation instead. Skip the
    # A-variance check for any trait with active AM. C and E are unaffected
    # (drawn fresh each generation), so they remain checked.
    am_active = {t: am_relatedness_mode(sim.assortment, t) != "none" for t in (1, 2)}
    variances = {col: _founder_var(founders, col) for col in comp_cols}
    for t in [1, 2]:
        for comp in ["A", "C", "E"]:
            col = f"{comp}{t}"
            if not (comp == "A" and am_active[t]):
                checks.append(_check_variance(variances[col], col, configured[col]))

    # Total variances, measured only: `simace validate`'s total_variance tests
    # them (ADR 0030). Var(A + C + E) directly, not the sum of component
    # variances, because AM with C transmission gives A and C a covariance (ADR 0025).
    trait1, trait2 = (
        FounderVariances(
            var_A=variances[f"A{t}"],
            var_C=variances[f"C{t}"],
            var_E=variances[f"E{t}"],
            var_liability=float(
                np.var(founders[f"A{t}"] + founders[f"C{t}"] + founders[f"E{t}"], ddof=1, dtype=np.float64)
            ),
        )
        for t in (1, 2)
    )

    # Cross-trait correlations. Assortative mating on either trait builds
    # disequilibrium into the additive covariance as it does into Var(A), so
    # the recorded founders' rA drifts from the configured value whenever
    # rA != 0; there is no closed-form expectation to check it against.
    rA_obs = safe_corrcoef(founders["A1"], founders["A2"])
    if not (am_active[1] or am_active[2]):
        rA_ok = abs(rA_obs - rA_param) < 0.15 if not np.isnan(rA_obs) else rA_param == 0
        checks.append(
            QualityCheck(
                "cross_trait_rA",
                rA_ok,
                f"Cross-trait A correlation: {rA_obs:.4f} (expected: {rA_param})",
                observed=float(rA_obs),
                expected=rA_param,
            )
        )
    # Measured only: `simace validate`'s cross_trait_rC tests it (ADR 0030).
    rC_obs = safe_corrcoef(founders["C1"], founders["C2"])

    rE_param = vc.rE
    rE_obs = safe_corrcoef(founders["E1"], founders["E2"])
    rE_ok = abs(rE_obs - rE_param) < 0.15 if not np.isnan(rE_obs) else rE_param == 0
    checks.append(
        QualityCheck(
            "cross_trait_rE",
            rE_ok,
            f"Cross-trait E correlation: {rE_obs:.4f} (expected: {rE_param})",
            observed=float(rE_obs),
            expected=rE_param,
        )
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
        by_pair = sim.c_transmission.coefficients(t)[1] != 0.0
        if by_pair and not nf_mask.any():
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
        # The share stays in the message only: report.yaml has always carried
        # this check's observed as null.
        checks.append(
            QualityCheck(
                f"c{t}_inheritance", c_shared > 0.99, f"Proportion of {unit} with shared {col}: {c_shared:.4f}"
            )
        )

    # E independence between siblings, when there are enough sibling pairs
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
                checks.append(
                    QualityCheck(
                        "e1_independence",
                        abs(e_corr) < tol,
                        f"E1 correlation between siblings: {e_corr:.4f} (expected: ~0, tol: {tol:.4f})",
                        observed=float(e_corr),
                        tolerance=tol,
                    )
                )

    measurements = StatisticalMeasurements(
        traits=(trait1, trait2), rA=float(rA_obs), rC=float(rC_obs), rE=float(rE_obs)
    )
    return FamilyResult(measurements, tuple(checks))
