"""Per-rep effective population size (Ne) summary for the stats runner.

Thin wrapper that invokes :func:`pedigree_graph.compute_all_ne` and
attaches scenario-level theoretical expectations from the rep's
``params.yaml`` when the configuration matches the canonical
random-mating, balanced-sex, ZTP-family regime.

The result is YAML-ready: each estimator key holds the dataclass
``to_dict()`` payload plus an ``expected`` field (``None`` when the
config has non-standard knobs such as assortative mating).
"""

from __future__ import annotations

__all__ = [
    "compute_effective_size",
    "ne_v_expected_ztp",
    "regression_estimator_regime_ok",
    "theoretical_expectations",
]

import math
from typing import TYPE_CHECKING, Any

from pedigree_graph import PedigreeGraph, compute_all_ne

if TYPE_CHECKING:
    import pandas as pd


# Bias on regression-based Ne estimators (Ne_I, Ne_C, Ne_CT) scales as
# ``Ne_V / (N · G²)`` due to Jensen inversion of a noisy slope; we mark
# the regime as "ok" only when this implied bias is below the validator's
# ±20 % tolerance.  Constant chosen to match `bias_ratio < 0.20`.
_REGRESSION_REGIME_THRESHOLD = 120.0


_NE_KEYS = (
    "ne_inbreeding",
    "ne_coancestry",
    "ne_variance_family_size",
    "ne_sex_ratio",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_hill_overlapping",
    "ne_caballero_toro",
)


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


def regression_estimator_regime_ok(n: float, g_ped: int, ne_v: float) -> bool:
    """Whether the regression-based Ne estimators are reliable at this scale.

    The slope estimate in Ne_I, Ne_C, and Ne_CT has variance
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
    Three estimators (Ne_V, Ne_iΔF, Ne_H) inherit that expectation
    directly — their finite-sample bias is ``O(1/N)`` and negligible at
    realistic simACE scales.

    Three regression-based estimators (Ne_I, Ne_C, Ne_CT) carry a Jensen
    bias on the inverted slope of order ``Ne_V² / (N · G²)`` that
    typically dominates at simACE's default ``G_ped = 6``.  We return
    their expectation only when
    :func:`regression_estimator_regime_ok` is satisfied, otherwise
    ``None`` (validator passes vacuously).

    Ne_sr stays at ``N`` (deterministic balanced sex ratio).  Ne_LTC
    under the ``Ne = 1/(2·Σc²)`` form is approximated as
    ``Ne_V_expected / 2`` — consistent with the WF limit where
    ``Ne_V → N`` and ``Ne_LTC → N/2``.  In practice the observed
    Ne_LTC is typically ``None`` (asymptote not reached within
    ``G_ped`` generations under realized WF noise), so the validator
    passes vacuously.
    """
    if config is None:
        return dict.fromkeys(_NE_KEYS)

    assort1 = float(config.get("assort1") or 0.0)
    assort2 = float(config.get("assort2") or 0.0)
    if assort1 != 0.0 or assort2 != 0.0:
        return dict.fromkeys(_NE_KEYS)

    N = config.get("N")
    if N is None:
        return dict.fromkeys(_NE_KEYS)

    mating_lambda = config.get("mating_lambda")
    if mating_lambda is None:
        return dict.fromkeys(_NE_KEYS)

    n = float(N)
    ne_v = ne_v_expected_ztp(n, float(mating_lambda))

    g_ped = config.get("G_ped")
    regression_ok = g_ped is not None and regression_estimator_regime_ok(n, int(g_ped), ne_v)
    regression_expected = ne_v if regression_ok else None

    return {
        "ne_inbreeding": regression_expected,
        "ne_coancestry": regression_expected,
        "ne_variance_family_size": ne_v,
        "ne_sex_ratio": n,
        "ne_individual_delta_f": ne_v,
        "ne_long_term_contributions": ne_v / 2.0,
        "ne_hill_overlapping": ne_v,
        "ne_caballero_toro": regression_expected,
    }


def compute_effective_size(
    pedigree: pd.DataFrame | PedigreeGraph,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all eight Ne estimators on ``pedigree`` and serialize to dicts.

    Args:
        pedigree: Either a pandas DataFrame with the standard pedigree
            columns or an already-built :class:`PedigreeGraph`.  Passing
            a graph avoids rebuilding it when the runner has already
            constructed one for relationship extraction.
        config: Per-rep params (e.g. loaded from ``params.yaml``).
            Used solely to derive theoretical expectations.

    Returns:
        Dict keyed on estimator name; each value is the matching
        dataclass's ``to_dict()`` payload merged with an ``expected``
        field (``float`` or ``None``).
    """
    pg = pedigree if isinstance(pedigree, PedigreeGraph) else PedigreeGraph(pedigree)
    raw = compute_all_ne(pg)
    expected = theoretical_expectations(config)
    out: dict[str, dict[str, Any]] = {}
    for name, result in raw.items():
        d = result.to_dict()
        d["expected"] = expected.get(name)
        out[name] = d
    return out
