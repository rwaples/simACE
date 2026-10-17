"""Onset-τ calibrated test: observed onset τ against its realized expectation (ADR 0027, 0028).

Run by ``simace validate`` on finished reps, not by ``simace run``:
:func:`run_onset_tau` regenerates the phenotyped population's raw onsets,
:func:`simace.analysis.stats.onset_tau.compute_onset_tau` computes every
cell, and :func:`score_onset_tau` scores them. A cell is scored when its
effective number of components, its numerical bound plus tie share, and its
SE allow a normal comparison; its z is ``(observed − expected) / SE``. A rep
records its largest ``|z|`` and the per-rep Bonferroni bound over its ``K``
scored cells, ``z_bound = Φ⁻¹(1 − 0.025/K)``, which a correct simulation
exceeds in about 5% of reps. It records no pass: the verdict is across reps.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import TYPE_CHECKING, Any

from .calibrated import information_cell, scored_cell

if TYPE_CHECKING:
    from .calibrated import RepInputs

#: Fewer effective components (entry-weighted) than this leaves the normal approximation unreliable.
MIN_EFFECTIVE_COMPONENTS = 100

#: Ties plus the target's numerical bound may shift the comparison by at most this many SEs.
APPROXIMATION_SE_SHARE = 0.05

#: Share of correct reps whose largest |z| exceeds the per-rep bound.
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


def score_onset_tau(summary: dict[str, Any]) -> dict[str, Any]:
    """Score one rep's onset-τ cells.

    Args:
        summary: :func:`~simace.analysis.stats.onset_tau.compute_onset_tau`'s result.

    Returns:
        ``{"scope", "cells": {name: cell}, "summary": {...}}``. Cell names are
        ``trait{t}_{code}`` and ``within_person``; each cell carries its
        statistics and is scored (with ``z``) or information (with its
        ``reason``). ``summary`` holds ``n_scored``, ``max_abs_z`` and
        ``z_bound`` (both None with nothing scored), ``skipped_traits`` and
        ``entry_cap``.
    """
    results: dict[str, Any] = {}
    for name, cell in _cells(summary):
        stats = {key: cell.get(key) for key in _FIELDS}
        reason = _unscored_reason(cell)
        if reason is None:
            results[name] = scored_cell(cell["observed"], cell["expected"], cell["se"], **stats)
        else:
            results[name] = information_cell(
                reason, observed=cell.get("observed"), expected=cell.get("expected"), se=cell.get("se"), **stats
            )
    abs_z = [abs(cell["z"]) for cell in results.values() if cell["status"] == "scored"]
    return {
        "scope": SCOPE,
        "cells": results,
        "summary": {
            "n_scored": len(abs_z),
            "max_abs_z": max(abs_z) if abs_z else None,
            "z_bound": NormalDist().inv_cdf(1 - FAMILYWISE_ALPHA / (2 * len(abs_z))) if abs_z else None,
            "skipped_traits": summary.get("skipped_traits") or {},
            "entry_cap": summary.get("entry_cap"),
        },
    }


def run_onset_tau(inputs: RepInputs) -> dict[str, Any]:
    """Regenerate a rep's raw onsets and score its onset τ.

    The phenotype stage is deterministic in the rep seed, so ``run_phenotype``
    on the recorded pedigree reproduces, in float64, the raw onsets the
    cohort stage drew for the phenotyped population (ADR 0021).

    Returns:
        :func:`score_onset_tau`'s result.
    """
    from simace.analysis.stats.onset_tau import compute_onset_tau
    from simace.phenotype.runner import run_phenotype

    raw = run_phenotype(inputs.pedigree, seed=inputs.seed, **inputs.phenotype)
    return score_onset_tau(compute_onset_tau(raw, inputs.pedigree, inputs.phenotype, seed=inputs.seed))
