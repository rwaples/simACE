"""Onset-τ check: observed onset τ against its realized expectation (ADR 0027).

An on-demand check, run by ``simace validate`` on finished reps, not by
``simace run``: :func:`run_onset_tau` regenerates the phenotyped population's
raw onsets, :func:`simace.analysis.stats.onset_tau.compute_onset_tau`
computes every cell, and :func:`validate_onset_tau` scores them. A cell is scored when
its effective number of components, its numerical bound plus tie share, and
its SE allow a normal comparison. The scored cells of one rep share one
Bonferroni family at a familywise 5%: a cell passes when
``|observed − expected| <= z·SE`` with ``z = Φ⁻¹(1 − 0.025/K)``.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import TYPE_CHECKING, Any

from ._common import _info, _result

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

#: Fewer effective components (entry-weighted) than this leaves the normal approximation unreliable.
MIN_EFFECTIVE_COMPONENTS = 100

#: Ties plus the target's numerical bound may shift the comparison by at most this many SEs.
APPROXIMATION_SE_SHARE = 0.05

#: Familywise false-failure rate of one rep's scored cells.
FAMILYWISE_ALPHA = 0.05

SCOPE = "phenotyped_population"

_FIELDS = (
    "n_pairs",
    "sampled_fraction",
    "n_entries",
    "n_entries_population",
    "n_people",
    "n_components",
    "n_eff",
    "max_component_share",
    "max_component_people",
    "shared_workload",
    "grid",
    "numerical_bound",
    "tie_share",
    "se",
    "r_eta",
    "s1",
    "s2",
    "tau_oracle",
)


def _cells(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    cells = summary.get("cells") or {}
    out = []
    for trait in ("trait1", "trait2"):
        out.extend((f"{trait}_{code}", cell) for code, cell in (cells.get(trait) or {}).items())
    if "within_person" in cells:
        out.append(("within_person", cells["within_person"]))
    return out


def _unscored_reason(cell: dict[str, Any]) -> str | None:
    if cell.get("expected") is None or cell.get("observed") is None:
        return cell.get("note") or "not computed"
    if cell["n_eff"] < MIN_EFFECTIVE_COMPONENTS:
        return f"effective components {cell['n_eff']:.0f} < {MIN_EFFECTIVE_COMPONENTS}"
    se = cell.get("se")
    if not se or se <= 0:
        return "SE is zero"
    approximation = cell["tie_share"] + cell["numerical_bound"]
    if approximation > APPROXIMATION_SE_SHARE * se:
        return f"ties plus numerical bound {approximation:.2g} > {APPROXIMATION_SE_SHARE}·SE ({se:.2g})"
    return None


def validate_onset_tau(summary: dict[str, Any]) -> dict[str, Any]:
    """One result per onset-τ cell: a scored check, or information with the reason it is not scored.

    Args:
        summary: :func:`~simace.analysis.stats.onset_tau.compute_onset_tau`'s result.

    Returns:
        ``{name: result}`` with names ``trait{t}_{code}`` and
        ``within_person``. Every result carries the cell's statistics, its
        ``status`` (``scored`` or ``information``) and, unscored, its
        ``reason``; all are scoped to the phenotyped population.
    """
    cells = _cells(summary)
    reasons = {name: _unscored_reason(cell) for name, cell in cells}
    n_scored = sum(reason is None for reason in reasons.values())
    z = NormalDist().inv_cdf(1 - FAMILYWISE_ALPHA / (2 * n_scored)) if n_scored else None

    results: dict[str, Any] = {}
    for name, cell in cells:
        stats = {key: cell.get(key) for key in _FIELDS}
        observed, expected = cell.get("observed"), cell.get("expected")
        reason = reasons[name]
        if reason is not None:
            results[name] = _info(
                f"onset τ {name}: not scored ({reason})",
                scope=SCOPE,
                status="information",
                reason=reason,
                observed=observed,
                expected=expected,
                **stats,
            )
            continue
        # Scored: both are numbers, and z exists because at least this cell is scored.
        observed, expected = cell["observed"], cell["expected"]
        tolerance = z * cell["se"]
        passed = abs(observed - expected) <= tolerance
        results[name] = _result(
            passed,
            f"onset τ {name}: observed {observed:.4f}, expected {expected:.4f}, "
            f"|diff| {abs(observed - expected):.4f} vs {z:.2f}·SE = {tolerance:.4f} (K = {n_scored})",
            scope=SCOPE,
            status="scored",
            observed=observed,
            expected=expected,
            tolerance=tolerance,
            z_crit=z,
            n_scored=n_scored,
            **stats,
        )
    for trait, why in (summary.get("skipped_traits") or {}).items():
        results[f"{trait}_skipped"] = _info(f"onset τ {trait}: {why}", scope=SCOPE, status="information", reason=why)
    return results


def run_onset_tau(pedigree: pl.DataFrame, *, seed: int, phenotype: Mapping[str, Any]) -> dict[str, Any]:
    """Regenerate a rep's raw onsets and score its onset τ.

    The phenotype stage is deterministic in the rep seed, so ``run_phenotype``
    on the recorded pedigree reproduces, in float64, the raw onsets the
    cohort stage drew for the phenotyped population (ADR 0021).

    Args:
        pedigree: The rep's recorded pedigree (``pedigree.parquet``).
        seed: The rep seed.
        phenotype: ``run_phenotype``'s keyword arguments other than ``seed``.

    Returns:
        ``{"cells": ..., "checks": ..., "summary": {"passed", "n_scored", "n_failed"}}``:
        :func:`~simace.analysis.stats.onset_tau.compute_onset_tau`'s cells and
        :func:`validate_onset_tau`'s results.
    """
    from simace.analysis.stats.onset_tau import compute_onset_tau
    from simace.phenotype.runner import run_phenotype

    raw = run_phenotype(pedigree, seed=seed, **phenotype)
    cells = compute_onset_tau(raw, pedigree, phenotype, seed=seed)
    checks = validate_onset_tau(cells)
    scored = [check for check in checks.values() if check.get("status") == "scored"]
    n_failed = sum(not check["passed"] for check in scored)
    return {
        "cells": cells,
        "checks": checks,
        "summary": {"passed": n_failed == 0, "n_scored": len(scored), "n_failed": n_failed},
    }
