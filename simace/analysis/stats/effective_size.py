"""Per-rep effective population size (Ne) summary for the stats runner.

Thin wrapper that invokes
:func:`pedigree_graph.effective_size.estimate_effective_sizes` and attaches
scenario-level theoretical expectations from the rep's ``params.yaml`` when
the configuration matches the canonical random-mating, balanced-sex,
ZTP-family regime.

The result is YAML-ready and always carries all eight estimator keys. A key
that produced a result holds the record's ``to_dict()`` payload plus an
``expected`` field (``None`` when the config has non-standard knobs such as
assortative mating). A key the library refused holds that refusal verbatim
— :class:`~pedigree_graph.effective_size.UnavailableEffectiveSize`'s
``{reason, code, fields}`` — with no ``ne`` and no ``expected``, so the
reason survives into ``effective_size.yaml`` instead of being flattened to a null Ne.
"""

from __future__ import annotations

__all__ = [
    "cli",
    "compute_effective_size",
    "family_size_variance_expected_ztp",
    "main",
    "ne_drift_expected_ztp",
    "regression_estimator_regime_ok",
    "theoretical_expectations",
]

import argparse
import math
from typing import TYPE_CHECKING, Any

from pedigree_graph import PedigreeGraph
from pedigree_graph.effective_size import (
    ALL_EFFECTIVE_SIZE_ESTIMATORS,
    UnavailableEffectiveSize,
    estimate_effective_sizes,
)

from simace.analysis.quality.assortative_mating import assortment_schedule
from simace.core.cli_base import add_logging_args, init_logging
from simace.core.cohort import read_cohort, read_pedigree, selected_views
from simace.core.yaml_io import dump_yaml, load_yaml

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# Bias on regression-based Ne estimators (Ne_I, Ne_C, Ne_GC) scales as
# ``Ne / (N · G²)`` due to Jensen inversion of a noisy slope; we mark the
# regime as "ok" only when this implied bias is below about 20 %.  The
# constant is an empirical heuristic, not a calibrated guarantee.
_REGRESSION_REGIME_THRESHOLD = 120.0


def ne_drift_expected_ztp(n: float, mating_lambda: float) -> float:
    """Closed-form drift ``Ne`` under simACE's standard mating model.

    Under random mating with balanced 50/50 sex, ZTP(λ) mating counts
    per individual, and multinomial allocation of N offspring across the
    resulting matings, the per-individual total-offspring count has

        ``E[k] = 2``,
        ``V(k) = 2 + 4 · r``,   ``r = Var[m] / E[m]²``,

    where ``m ~ ZTP(λ)`` with

        ``E[m]   = λ / (1 − e^(−λ))``,
        ``Var[m] = E[m] · (1 + λ) − E[m]²``.

    Hill 1979 eq. 10 with ``M = F = N/2``, ``L = 1`` and
    ``σ²_m = σ²_f = V(k)`` is ``Ne = 8N / (2·V(k) + 4) = 4N / (2 + V(k))``,
    which is

        ``Ne = N / (1 + r)``.

    The ``+ 2`` beside ``V(k)`` is Mendelian segregation (Hill's eq. 7);
    the form ``2N/V(k)`` omits it and runs 13 % low at λ = 0.5 (issue #72).

    This is the large-N limit.  Trimming the larger sex's mating slots
    to the smaller's total (:func:`~simace.simulation.simulate.balance_mating_slots`)
    adds family-size variance of order ``N^(−1/2)``, so simulated
    pedigrees drift slightly faster than this.

    Limits:
        * ``λ → 0⁺`` (degenerate at m=1, monogamous): ``Ne = N``.
        * ``λ → ∞`` (Poisson, no truncation): ``Ne = N``.
        * Default ``λ = 0.5``: ``Ne ≈ 0.8472 · N``.
    """
    return float(n) / (1.0 + family_size_variance_expected_ztp(mating_lambda)["cov"])


def family_size_variance_expected_ztp(mating_lambda: float) -> dict[str, float]:
    """Closed-form ``(v, cov)`` per-sex-quadrant family-size decomposition.

    Splits the total offspring-count variance
    ``V(k) = 2 + 4 · Var[m] / E[m]²`` behind :func:`ne_drift_expected_ztp` into the per-(parent-sex ×
    offspring-sex) variance ``v`` and the between-offspring-sex
    covariance ``cov`` reported by :func:`pedigree_graph.ne_variance_family_size`.

    Under balanced 50/50 offspring sex assignment within each mating
    (``k_M | k ~ Binomial(k, 0.5)``), the law of total variance gives

        ``v   = E[k]/4 + V(k)/4 = 1 + Var[m] / E[m]²``,
        ``cov = -E[k]/4 + V(k)/4 = Var[m] / E[m]²``,

    so that ``2 · v + 2 · cov = V(k)``.  All four quadrants
    (``v_mm = v_mf = v_fm = v_ff``) share the same closed-form ``v``,
    and both covariances (``cov_m = cov_f``) share the same ``cov``,
    because the parent and offspring sex labels are exchangeable under
    the balanced-mating regime.

    Limits:
        * ``λ → 0⁺``: ``v = 1``, ``cov = 0`` (m=1 degenerate; pure
          binomial sex split, no overdispersion).
        * ``λ → ∞``: ``v = 1``, ``cov = 0`` (Poisson limit).
        * Default ``λ = 0.5``: ``v ≈ 1.180``, ``cov ≈ 0.180``.

    Returns a dict with keys ``"v"`` and ``"cov"``.
    """
    if mating_lambda <= 0:
        return {"v": 1.0, "cov": 0.0}
    p = 1.0 - math.exp(-mating_lambda)
    e_m = mating_lambda / p
    var_m = e_m * (1.0 + mating_lambda) - e_m * e_m
    ratio = var_m / (e_m * e_m)
    return {"v": 1.0 + ratio, "cov": ratio}


def regression_estimator_regime_ok(n: float, g_ped: int, ne: float) -> bool:
    """Whether the regression-based Ne estimators are reliable at this scale.

    The slope estimate in Ne_I, Ne_C, and Ne_GC has variance
    ``∝ 1/(N·G³)``; inverting the slope to get Ne incurs a Jensen bias
    that scales as ``Ne² / (N · G²)``, where ``Ne`` is the drift
    expectation.  We declare the regime acceptable when the implied bias
    on Ne is below ~20 % of Ne, which corresponds to ``N · G² ≥ 120 · Ne``.

    Returns ``False`` for ``g_ped < 2`` (no slope possible) regardless
    of ``N``.
    """
    if g_ped < 2:
        return False
    return n * g_ped * g_ped >= _REGRESSION_REGIME_THRESHOLD * ne


def theoretical_expectations(config: dict[str, Any] | None) -> dict[str, float | None]:
    """Closed-form Ne expectations under standard random-mating assumptions.

    Returns a per-estimator dict.  Every entry is ``None`` when ``config``
    is missing, ``N`` is unknown, or the configuration includes a
    non-standard knob (currently: assortative mating in any generation,
    through ``assort1``, ``assort2`` or ``assort_matrix``).

    The estimators that read the realized drift rate share one target,
    the drift ``Ne`` of :func:`ne_drift_expected_ztp`: Hill 1979 eq. 10,
    ``4N / (2 + V(k))``, which is ``N`` under Wright-Fisher (``V(k) = 2``).
    Issue #72 measured the realized rate on simACE pedigrees at
    ``N = 2000`` (5 reps, read from pair sharing and from the rise in mean
    kinship) at −1.6 % to +0.1 % of it under ZTP(0.5) and ZTP(2); the
    committed check is ``test_drift_expectation_matches_simulator_mc``.

    * **Ne_I, Ne_C, Ne_GC** carry a Jensen bias on the inverted slope of
      order ``Ne² / (N · G²)`` that typically dominates at simACE's default
      ``G_ped = 6``.  We return their expectation only when
      :func:`regression_estimator_regime_ok` is satisfied, otherwise
      ``None``.
    * **Ne_iΔF** carries a known upward bias of ``t/(t−1)`` on top of it,
      where ``t = G_ped − 1`` is the last recorded cohort's equivalent
      complete generations, because the recorded founders are unrelated by
      construction and so the cohort has drifted for one generation fewer
      than its pedigree is deep.  Its expectation is ``Ne · t/(t−1)``, and
      ``None`` below ``t = 3`` where that factor is untested.
    * **Ne_LTC** is the drift ``Ne`` itself.  Wray & Thompson 1990 eq. 31
      is ``Ne = 2N/(μ_r² + σ_r²)`` with ``μ_r = 1``, and ``σ_r² = V(k)/2``
      makes that ``4N/(2 + V(k))``.  It is an asymptotic target: the
      estimator reads the last cohort, so a shallow pedigree need not reach
      it.
    * **Ne_sr** stays at ``N`` (deterministic balanced sex ratio).
    * **Ne_V and Ne_H** get ``N`` under Wright-Fisher and ``None``
      otherwise.  pedigree-graph's Ne_V omits Hill's segregation term and
      returns ``2N/V(k)``, which equals the drift ``Ne`` only at
      ``V(k) = 2`` (pedigree-graph#36), and without ``birth_year`` Ne_H
      passes Ne_V through.  No expectation is stated for a number that is
      not the drift ``Ne``.

    Assumptions the config cannot check: the expectations describe the
    full simulated pedigree without twins.  MZ twinning (``p_mztwin``)
    changes birth allocation and shares a genome, and the analysis
    pedigree is the selected sample plus its ancestors, so a small or
    selective sample shifts the realized values.
    """
    if config is None:
        return dict.fromkeys(ALL_EFFECTIVE_SIZE_ESTIMATORS)

    N = config.get("N")
    if N is None:
        return dict.fromkeys(ALL_EFFECTIVE_SIZE_ESTIMATORS)

    mating_model = config.get("mating_model", "standard")
    n = float(N)

    if mating_model == "wright_fisher":
        # Sex-structured idealized WF: per-individual offspring count has
        # mean 2 and variance ≈ 2 (Poisson(2)), so V(k) = 2 and every drift
        # form gives N.  assort1/assort2/mating_lambda are no-ops.  The
        # regression gate reduces to G_ped² ≥ 120.
        ne = n
        family_size_expected: float | None = n
    else:
        # Standard model: AM in any generation or through assort_matrix
        # nullifies expectations; mating_lambda required.
        assort1, assort2 = assortment_schedule(config)
        if any(assort1) or any(assort2):
            return dict.fromkeys(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        mating_lambda = config.get("mating_lambda")
        if mating_lambda is None:
            return dict.fromkeys(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        ne = ne_drift_expected_ztp(n, float(mating_lambda))
        family_size_expected = None

    g_ped = config.get("G_ped")
    regression_ok = g_ped is not None and regression_estimator_regime_ok(n, int(g_ped), ne)
    regression_expected = ne if regression_ok else None

    # Ne_iΔF reads the last recorded cohort, whose equivalent complete
    # generations are G_ped - 1: simulate.py records exactly G_ped cohorts
    # labelled 0..G_ped-1 with cohort 0 marked founder.  Those founders are
    # unrelated as recorded, so the cohort carries t - 1 generations of drift
    # while Gutiérrez eq. 2 divides by t, inflating Ne by t/(t-1) in the
    # large-N limit (pedigree-graph ADR 0012).  Measured on Wright-Fisher
    # pedigrees at N=2000: +32.9% at t=4 against 33.3% predicted, +28.7% at
    # t=5 against 25.0%, +19.2% at t=7 against 16.7%.  Untested below t=3.
    delta_f_expected = None
    if g_ped is not None:
        t_ref = int(g_ped) - 1
        if t_ref >= 3:
            delta_f_expected = ne * t_ref / (t_ref - 1.0)

    return {
        "ne_inbreeding": regression_expected,
        "ne_coancestry": regression_expected,
        "ne_variance_family_size": family_size_expected,
        "ne_sex_ratio": n,
        "ne_individual_delta_f": delta_f_expected,
        "ne_long_term_contributions": ne,
        "ne_hill_overlapping": family_size_expected,
        "ne_group_coancestry": regression_expected,
    }


def compute_effective_size(
    pedigree: pd.DataFrame | pl.DataFrame | PedigreeGraph,
    config: dict[str, Any] | None = None,
    skip_ne_coancestry: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run the Ne estimators on ``pedigree`` and serialize to dicts.

    Args:
        pedigree: Either a frame with the standard pedigree columns or an
            already-built :class:`PedigreeGraph`.  Passing a graph avoids
            rebuilding it when the runner has already constructed one for
            relationship extraction.
        config: Per-rep params (e.g. loaded from ``params.yaml``).
            Used solely to derive theoretical expectations.
        skip_ne_coancestry: When True, ``ne_coancestry`` is dropped from the
            estimator selection and the full sparse kinship matrix is never
            built — required on very large pedigrees where K would OOM.  Its
            key is still present, carrying ``reason: not_requested``.

    Returns:
        Dict keyed on all eight estimator names.  A computed estimator's
        value is its record's ``to_dict()`` payload plus an ``expected``
        field (``float`` or ``None``); an unavailable one is its
        ``{reason, code, fields}`` payload with no ``ne`` and no
        ``expected``.
    """
    pg = pedigree if isinstance(pedigree, PedigreeGraph) else PedigreeGraph.from_frame(pedigree)
    estimators = ALL_EFFECTIVE_SIZE_ESTIMATORS
    if skip_ne_coancestry:
        estimators = tuple(name for name in estimators if name != "ne_coancestry")
    raw = estimate_effective_sizes(pg, estimators)
    expected = theoretical_expectations(config)
    # With pg.birth_year set, ne_hill_overlapping computes Hill 1979 eq. 10
    # with an estimated generation interval L, for which no closed-form
    # expectation is stated here.
    if pg.birth_year is not None:
        expected["ne_hill_overlapping"] = None
    out: dict[str, dict[str, Any]] = {}
    for name, result in raw.items():
        payload = result.to_dict()
        if not isinstance(result, UnavailableEffectiveSize):
            payload["expected"] = expected.get(name)
        out[name] = payload
    return out


def main(
    pedigree_path: str,
    cohort_path: str,
    params_path: str,
    output_path: str,
    skip_ne_coancestry: bool = False,
) -> None:
    """Compute Ne for one rep and write ``effective_size.yaml``.

    Rebuilds the analysis pedigree (the analysis sample plus its ancestor
    closure, links outside it severed) from the recorded ``pedigree_path``
    and ``cohort_path`` with :func:`~simace.core.cohort.selected_views`,
    builds a :class:`PedigreeGraph` on it, runs
    :func:`compute_effective_size`, and dumps the YAML-ready dict to
    ``output_path``.
    """
    df_observed = selected_views(read_pedigree(pedigree_path), read_cohort(cohort_path)).pedigree
    params = load_yaml(params_path)

    pg = PedigreeGraph.from_frame(df_observed)
    result = compute_effective_size(pg, config=params, skip_ne_coancestry=skip_ne_coancestry)

    dump_yaml(result, output_path)


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line entry point: compute Ne for one rep."""
    from simace.core.publish import publish

    parser = argparse.ArgumentParser(prog=prog, description="Compute Ne estimators")
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True, help="Recorded pedigree parquet (pedigree.parquet)")
    parser.add_argument("--cohort", required=True, help="The rep's cohort.parquet (defines the analysis pedigree)")
    parser.add_argument("--params", required=True, help="Per-rep params.yaml")
    parser.add_argument("--output", required=True, help="Output effective_size.yaml")
    parser.add_argument(
        "--ne-coancestry",
        action="store_true",
        help="Include the coancestry-rate Ne_C estimator alongside the other seven. Off by "
        "default because its kinship DP dominates memory on large pedigrees, matching the "
        "analysis.skip_ne_coancestry pipeline default. Without it ne_coancestry carries "
        "reason: not_requested instead of a result.",
    )
    args = parser.parse_args(argv)
    init_logging(args)
    with publish(args.output) as (tmp,):
        main(
            args.pedigree,
            args.cohort,
            args.params,
            str(tmp),
            skip_ne_coancestry=not args.ne_coancestry,
        )
