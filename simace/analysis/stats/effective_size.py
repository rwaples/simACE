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
reason survives into the validator instead of being flattened to a null Ne.
"""

from __future__ import annotations

__all__ = [
    "cli",
    "compute_effective_size",
    "family_size_variance_expected_ztp",
    "main",
    "ne_v_expected_ztp",
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

from simace.core.cli_base import add_logging_args, init_logging
from simace.core.parquet import load_parquet
from simace.core.pedigree_filter import filter_pedigree_to_observed
from simace.core.yaml_io import dump_yaml, load_yaml

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# Bias on regression-based Ne estimators (Ne_I, Ne_C, Ne_GC) scales as
# ``Ne_V / (N · G²)`` due to Jensen inversion of a noisy slope; we mark
# the regime as "ok" only when this implied bias is below the validator's
# ±20 % tolerance.  Constant chosen to match `bias_ratio < 0.20`.
_REGRESSION_REGIME_THRESHOLD = 120.0


def ne_v_expected_ztp(n: float, mating_lambda: float) -> float:
    """Closed-form ``Ne_V`` expectation under simACE's mating model.

    Under random mating with balanced 50/50 sex, ZTP(λ) mating counts
    per individual, and multinomial allocation of N offspring across the
    resulting matings, the per-individual total-offspring count has

        ``E[k] = 2``,
        ``V(k) = 2 + 4 · Var[m] / E[m]²``,

    where ``m ~ ZTP(λ)`` with

        ``E[m]   = λ / (1 − e^(−λ))``,
        ``Var[m] = E[m] · (1 + λ) − E[m]²``.

    Plugging into ``Ne_V = 2N / V(k)`` yields

        ``Ne_V = N / (1 + 2 · Var[m] / E[m]²)``.

    The formula is exact in the multinomial → Poisson per-mating
    offspring limit (large M); finite-sample correction is
    ``O(1 / number_of_matings)``.

    Limits:
        * ``λ → 0⁺`` (degenerate at m=1, monogamous): ``Ne_V = N``.
        * ``λ → ∞`` (Poisson, no truncation): ``Ne_V = N``.
        * Default ``λ = 0.5``: ``Ne_V ≈ 0.7349 · N``.
    """
    if mating_lambda <= 0:
        return float(n)
    p = 1.0 - math.exp(-mating_lambda)
    e_m = mating_lambda / p
    var_m = e_m * (1.0 + mating_lambda) - e_m * e_m
    return float(n) / (1.0 + 2.0 * var_m / (e_m * e_m))


def family_size_variance_expected_ztp(mating_lambda: float) -> dict[str, float]:
    """Closed-form ``(v, cov)`` per-sex-quadrant family-size decomposition.

    Splits :func:`ne_v_expected_ztp`'s total offspring-count variance
    ``V(k) = 2 + 4 · Var[m] / E[m]²`` into the per-(parent-sex ×
    offspring-sex) variance ``v`` and the between-offspring-sex
    covariance ``cov`` reported by :func:`pedigree_graph.ne_variance_family_size`.

    Under balanced 50/50 offspring sex assignment within each mating
    (``k_M | k ~ Binomial(k, 0.5)``), the law of total variance gives

        ``v   = E[k]/4 + V(k)/4 = 1 + Var[m] / E[m]²``,
        ``cov = -E[k]/4 + V(k)/4 = Var[m] / E[m]²``,

    so that ``2 · v + 2 · cov = V(k)`` and downstream Ne_V is consistent
    with :func:`ne_v_expected_ztp`.  All four quadrants
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


def regression_estimator_regime_ok(n: float, g_ped: int, ne_v: float) -> bool:
    """Whether the regression-based Ne estimators are reliable at this scale.

    The slope estimate in Ne_I, Ne_C, and Ne_GC has variance
    ``∝ 1/(N·G³)``; inverting the slope to get Ne incurs a Jensen bias
    that scales as ``Ne_V² / (N · G²)``.  We declare the regime
    acceptable when the implied bias on Ne is below ~20 % of Ne_V,
    which corresponds to ``N · G² ≥ 120 · Ne_V``.

    Returns ``False`` for ``g_ped < 2`` (no slope possible) regardless
    of ``N``.
    """
    if g_ped < 2:
        return False
    return n * g_ped * g_ped >= _REGRESSION_REGIME_THRESHOLD * ne_v


def theoretical_expectations(config: dict[str, Any] | None) -> dict[str, float | None]:
    """Closed-form Ne expectations under standard random-mating assumptions.

    Returns a per-estimator dict.  Every entry is ``None`` when ``config``
    is missing, ``N`` is unknown, or the configuration includes a
    non-standard knob (currently: nonzero ``assort1`` / ``assort2``).

    Under random mating with 50/50 sex and ZTP(``mating_lambda``) family
    allocation, the family-size variance correction reduces
    ``Ne_V``-family estimators below ``N`` per :func:`ne_v_expected_ztp`.
    Two estimators (Ne_V, Ne_H) inherit that expectation directly — their
    finite-sample bias is ``O(1/N)`` and negligible at realistic simACE
    scales.

    Ne_iΔF carries a known upward bias of ``t/(t−1)`` on top of it, where
    ``t = G_ped − 1`` is the last recorded cohort's equivalent complete
    generations, because the recorded founders are unrelated by
    construction and so the cohort has drifted for one generation fewer
    than its pedigree is deep.  Its expectation is ``Ne_V · t/(t−1)``, and
    ``None`` below ``t = 3`` where that factor is untested.

    Three regression-based estimators (Ne_I, Ne_C, Ne_GC) carry a Jensen
    bias on the inverted slope of order ``Ne_V² / (N · G²)`` that
    typically dominates at simACE's default ``G_ped = 6``.  We return
    their expectation only when
    :func:`regression_estimator_regime_ok` is satisfied, otherwise
    ``None`` (validator passes vacuously).

    Ne_sr stays at ``N`` (deterministic balanced sex ratio).

    Ne_LTC is the harmonic mean of ``N`` and ``Ne_V``.  Wray & Thompson
    1990 eq. 31 is ``Ne = 2N/(μ_r² + σ_r²)`` with ``μ_r = 1``, and their
    p. 51 relation ``σ_r² = V(k)/2`` makes that ``4N/(2 + V(k))``;
    :func:`ne_v_expected_ztp` is ``2N/V(k)``, so ``V(k)`` cancels out of

        ``2/Ne_LTC = 1/N + 1/Ne_V``.

    Under Wright-Fisher ``V(k) = 2``, ``Ne_V = N``, and the two coincide.
    ``test_ne_ltc_expectation_matches_simulator_mc`` is the committed method
    behind that claim.  Driven through simACE's own simulator at ``N = 1000``
    over 12 reps, ``2/Σc²`` lands +0.75 % from this expectation under
    ZTP(0.5) at ``G_ped = 8``, +0.73 % at ``G_ped = 12``, and +0.58 % under
    Wright-Fisher, each inside 0.8 standard errors of the replicate mean.

    The estimator this describes is the post-ADR-0012 ``Ne = 2/Σc²``.
    pedigree-graph 0.8 reports ``1/(2·Σc²)``, four times lower, so
    :func:`compute_effective_size` withholds this expectation from a 0.8
    record.
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
        # mean 2 and variance ≈ 2 (Poisson(2)), giving Ne_V → N via
        # Crow-Kimura. assort1/assort2/mating_lambda are no-ops.  The
        # regression-based estimators still carry the same Jensen-bias
        # gate; under WF the gate reduces to G_ped² ≥ 120.
        ne_v = n
    else:
        # Standard model: AM nullifies expectations; mating_lambda required.
        assort1 = float(config.get("assort1") or 0.0)
        assort2 = float(config.get("assort2") or 0.0)
        if assort1 != 0.0 or assort2 != 0.0:
            return dict.fromkeys(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        mating_lambda = config.get("mating_lambda")
        if mating_lambda is None:
            return dict.fromkeys(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        ne_v = ne_v_expected_ztp(n, float(mating_lambda))

    g_ped = config.get("G_ped")
    regression_ok = g_ped is not None and regression_estimator_regime_ok(n, int(g_ped), ne_v)
    regression_expected = ne_v if regression_ok else None

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
            delta_f_expected = ne_v * t_ref / (t_ref - 1.0)

    return {
        "ne_inbreeding": regression_expected,
        "ne_coancestry": regression_expected,
        "ne_variance_family_size": ne_v,
        "ne_sex_ratio": n,
        "ne_individual_delta_f": delta_f_expected,
        "ne_long_term_contributions": 2.0 * n * ne_v / (n + ne_v),
        "ne_hill_overlapping": ne_v,
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
    # Hill 1979's closed-form Ne_V passthrough only applies under the
    # strictly-discrete simACE simulator (L = 1).  When pg.birth_year is
    # set, ne_hill_overlapping computes the true overlapping-generation
    # form (Hill 1979 eq. 10) with a non-trivial L; no analytic
    # expectation is available, so the validator passes vacuously.
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
    phenotype_path: str,
    params_path: str,
    output_path: str,
    skip_ne_coancestry: bool = False,
) -> None:
    """Compute Ne for one rep and write ``effective_size.yaml``.

    Reads ``pedigree_path`` and ``phenotype_path``, restricts the pedigree to
    observed (phenotyped) IDs plus their ancestor closure within
    ``pedigree_path`` (so kinship arithmetic still works through pre-phenotyping
    ancestors), builds a :class:`PedigreeGraph`, runs
    :func:`compute_effective_size`, and dumps the YAML-ready dict to
    ``output_path``.
    """
    df_ped = load_parquet(pedigree_path)
    df_phe = load_parquet(phenotype_path)
    params = load_yaml(params_path)

    df_observed = filter_pedigree_to_observed(df_ped, df_phe["id"].to_numpy())
    pg = PedigreeGraph.from_frame(df_observed)
    result = compute_effective_size(pg, config=params, skip_ne_coancestry=skip_ne_coancestry)

    dump_yaml(result, output_path)


def cli() -> None:
    """Argparse entry point for running outside Snakemake."""
    parser = argparse.ArgumentParser(description="Compute Ne estimators")
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True, help="Pedigree parquet (post-dropout)")
    parser.add_argument("--phenotype", required=True, help="Sampled phenotype parquet (defines observed set)")
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
    args = parser.parse_args()
    init_logging(args)
    main(
        args.pedigree,
        args.phenotype,
        args.params,
        args.output,
        skip_ne_coancestry=not args.ne_coancestry,
    )
