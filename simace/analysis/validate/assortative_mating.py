"""Mate-correlation (assortative-mating) validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from simace.simulation.assortment import _parent_ce_gen
from simace.simulation.mate_correlation import expected_mate_corr_matrix

from ._common import _corr_se, _info, _result

if TYPE_CHECKING:
    from ._common import MateCorrelations


def validate_assortative_mating(params: dict[str, Any], mates: MateCorrelations) -> dict[str, Any]:
    """Validate mate correlation on liability when assortative mating is configured.

    Checks the Pearson correlation of mother and father liability over the
    unique mating pairs, per trait, against the configured ``assort1`` /
    ``assort2`` parameters. When both traits assort, also checks the
    cross-trait mate correlations against
    :func:`~simace.simulation.mate_correlation.expected_mate_corr_matrix`.

    Args:
        params: Scenario parameters; uses keys ``assort1``, ``assort2``, and
            for the cross-trait check ``assort_matrix``, ``rA``, ``rC``,
            ``rE``, ``A1``, ``C1``, ``E1``, ``A2``, ``C2``, ``E2``.
        mates: The pedigree's mate correlations
            (:func:`~simace.analysis.validate._common.mate_correlations`).

    Returns:
        Dict of check-name to result dicts.
    """
    results: dict[str, Any] = {}
    assort1 = params.get("assort1", 0.0)
    assort2 = params.get("assort2", 0.0)

    # Per-gen AM (dict) is not currently validated — observed pairs span
    # multiple generations with potentially different per-gen AM strengths,
    # so a single-value tolerance check is ill-defined. Skip with a
    # passing result. A future improvement: stratify pairs by generation
    # and validate each cohort against its per-gen target.
    if isinstance(assort1, dict) or isinstance(assort2, dict):
        msg = (
            "Per-generation assort1/assort2 (dict-valued); skipping pooled "
            "mate-correlation check (would conflate cohort-varying targets)."
        )
        results["mate_corr_liability1"] = _result(True, msg)
        results["mate_corr_liability2"] = _result(True, msg)
        return results

    n_pairs = mates.n_pairs
    if n_pairs == 0:
        results["mate_corr_liability1"] = _result(True, "No mating pairs with both parents present")
        results["mate_corr_liability2"] = _result(True, "No mating pairs with both parents present")
        return results

    for t, expected in [(1, assort1), (2, assort2)]:
        obs = float(mates.liability[t - 1, t - 1])

        if np.isnan(obs):
            results[f"mate_corr_liability{t}"] = _result(
                True,
                f"Cannot compute mate correlation for trait {t} (zero variance)",
                expected=float(expected),
                observed=float(obs),
            )
            continue

        se = _corr_se(expected, n_pairs)
        tol = max(0.1, 3 * se)
        ok = abs(obs - expected) < tol
        results[f"mate_corr_liability{t}"] = _result(
            ok,
            f"Mate correlation liability{t}: {obs:.4f} (expected: {expected}, tol: {tol:.4f})",
            expected=float(expected),
            observed=float(obs),
            n_pairs=n_pairs,
        )

    # Genetic (A-component) mate correlation mu_A — informational; consumed by
    # the AM-corrected relative-correlation reference lines (see am_relatedness
    # and plot_A_correlations). Reduces to ~0 with no assortment.
    for t in [1, 2]:
        obs_a = mates.genetic[t - 1]
        results[f"mate_corr_A{t}"] = _info(
            f"Genetic mate correlation A{t} (mu_A): {obs_a:.4f}",
            observed=None if np.isnan(obs_a) else float(obs_a),
            n_pairs=n_pairs,
        )

    # Cross-trait validation (only when both traits assort)
    if assort1 != 0 and assort2 != 0:
        targets = _cross_trait_targets(params, assort1, assort2)
        if max(targets) - min(targets) > 1e-12:
            msg = (
                f"Cross-trait mate-correlation target varies by generation "
                f"({min(targets):.4f} to {max(targets):.4f}, from per-generation C/E); "
                f"skipping the pooled check."
            )
            results["mate_corr_cross_12"] = _result(True, msg)
            results["mate_corr_cross_21"] = _result(True, msg)
        else:
            results.update(_check_cross_trait(targets[0], mates))

    return results


def _cross_trait_targets(params: dict[str, Any], assort1: float, assort2: float) -> list[float]:
    """The simulator's cross-trait mate-correlation target for each parental C/E generation of a recorded mating.

    Mating iteration ``i`` mates parental C/E generation ``_parent_ce_gen(i)``
    (:class:`~simace.simulation.assortment.AssortmentPlan`), whose ``rho_w``
    sets the target unless ``assort_matrix`` fixes it. Only iterations from
    ``burnin + 1`` on give Mating Pairs: iteration ``burnin`` makes the
    recorded founders, whose parents are not rows, and burn-in matings are
    not recorded.
    With one recorded generation there are no pairs, and the founders'
    iteration stands in.
    """
    from simace.simulation.simulate import resolve_per_gen_param

    G_sim = int(params.get("G_sim") or params.get("G_ped") or 1)
    G_ped = int(params.get("G_ped") or G_sim)
    first_recorded_mating = min(G_sim - G_ped + 1, G_sim - 1)
    per_gen = {k: resolve_per_gen_param(params.get(k, 0), G_sim, name=k) for k in ("C1", "C2", "E1", "E2")}
    return [
        float(
            expected_mate_corr_matrix(
                assort1,
                assort2,
                rA=params.get("rA", 0),
                rC=params.get("rC", 0),
                A1=params.get("A1", 0),
                C1=per_gen["C1"][g],
                A2=params.get("A2", 0),
                C2=per_gen["C2"][g],
                assort_matrix=params.get("assort_matrix"),
                rE=params.get("rE", 0),
                E1=per_gen["E1"][g],
                E2=per_gen["E2"][g],
            )[0, 1]
        )
        for g in sorted({_parent_ce_gen(i) for i in range(first_recorded_mating, G_sim)})
    ]


def _check_cross_trait(c_expected: float, mates: MateCorrelations) -> dict[str, Any]:
    """Check both cross-trait mate correlations against *c_expected*."""
    results: dict[str, Any] = {}
    n_pairs = mates.n_pairs
    for label, mi, fi in [("cross_12", 1, 2), ("cross_21", 2, 1)]:
        obs = float(mates.liability[mi - 1, fi - 1])

        if np.isnan(obs):
            results[f"mate_corr_{label}"] = _result(
                True,
                f"Cannot compute mate correlation {label} (zero variance)",
                expected=float(c_expected),
                observed=float(obs),
            )
            continue

        se = _corr_se(c_expected, n_pairs)
        tol = max(0.1, 3 * se)
        ok = abs(obs - c_expected) < tol
        results[f"mate_corr_{label}"] = _result(
            ok,
            f"Mate correlation {label}: {obs:.4f} (expected: {c_expected:.4f}, tol: {tol:.4f})",
            expected=float(c_expected),
            observed=float(obs),
            n_pairs=n_pairs,
        )
    return results
