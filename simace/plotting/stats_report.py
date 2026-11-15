"""Adapter from the curated v2 report (+ plot payload) to the flat plotting view.

The plot helpers consume a single flat dict of keys (``cumulative_incidence``,
``censoring``, ``family_size``, ``tetrachoric``, ``per_generation``,
``parameters``, …). This module rebuilds that view from the v2 ``report.yaml``
scope-organized groups and the dense arrays in ``plot_payload.yaml``, so the
plot code does not need to know about the v2 scientific grouping.
"""

from typing import Any

import numpy as np

from simace.analysis.sample_stats import SampleStats
from simace.simulation.params import SimulationParams

__all__ = [
    "plotting_report_view",
    "plotting_report_views",
    "report_per_generation",
    "truth_entry",
    "truth_mean",
]


def report_per_generation(report: dict[str, Any] | None) -> dict[str, Any]:
    """Reconstruct the flat ``generation_N -> {A1_var, C1_var, …}`` table from truth.

    Inverse of the per-trait ``truth.recorded_pedigree.traits.*.realized_by_generation``
    layout, matching what the per-generation heritability plots expect.
    """
    traits = (((report or {}).get("truth") or {}).get("recorded_pedigree") or {}).get("traits", {})
    out: dict[str, Any] = {}
    for t in (1, 2):
        realized_by_generation = (traits.get(f"trait{t}") or {}).get("realized_by_generation", {})
        for gen_key, vals in realized_by_generation.items():
            entry = out.setdefault(gen_key, {})
            if vals.get("n") is not None:
                entry["n"] = vals.get("n")
            entry[f"A{t}_var"] = vals.get("var_A")
            entry[f"C{t}_var"] = vals.get("var_C")
            entry[f"E{t}_var"] = vals.get("var_E")
            entry[f"liability{t}_variance"] = vals.get("var_liability")
            entry[f"A{t}_cov_non_genetic"] = vals.get("cov_A_non_genetic")
            entry[f"A{t}_cov_C"] = vals.get("cov_A_C")
            entry[f"A{t}_cov_E"] = vals.get("cov_A_E")
    return out


def plotting_report_view(report: dict[str, Any], plot_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the flat plotting view from a v2 report and its plot payload.

    ``plot_payload`` may be omitted when only scalar fields are needed (e.g.
    ad-hoc cross-scenario tools); curve arrays will then be absent.
    """
    stats = SampleStats.from_report(report, plot_payload)
    sample = stats.analysis_sample
    pedigree = stats.analysis_pedigree

    view: dict[str, Any] = {
        "n_individuals": stats.sample_size.n_individuals,
        "n_generations": stats.sample_size.n_generations,
    }
    view |= {key: value for key, value in sample.to_report().items() if not key.startswith("censoring_")}
    # The plot helpers want the windows curve under the bare key "censoring".
    if sample.censoring_windows is not None:
        view["censoring"] = sample.censoring_windows
    if sample.censoring_confusion is not None:
        view["censoring_confusion"] = sample.censoring_confusion
    if sample.censoring_cascade is not None:
        view["censoring_cascade"] = sample.censoring_cascade
    view["pair_counts"] = sample.relationship_pair_counts

    view["parent_status"] = pedigree.parent_status
    view["mate_correlation"] = pedigree.mate_correlation
    view["pair_counts_ped"] = pedigree.relationship_pair_counts
    view["n_individuals_ped"] = stats.pedigree_size.n_individuals
    view["n_generations_ped"] = stats.pedigree_size.n_generations

    view["observed_h2_estimators"] = stats.observed_h2_estimators
    view["per_generation"] = report_per_generation(report)
    view["parameters"] = (report.get("inputs") or {}).get("parameters", {})
    # The parameters as the record the reference lines read; None for a report without them.
    view["sim"] = SimulationParams.from_mapping(view["parameters"]) if view["parameters"] else None
    # Recorded-pedigree truth, present only under C transmission (ADR 0025).
    traits = ((report.get("truth") or {}).get("recorded_pedigree") or {}).get("traits", {})
    view["truth_relationships"] = {name: tr["relationships"] for name, tr in traits.items() if "relationships" in tr}
    view["truth_midparent_slope"] = {
        name: tr["midparent_slope"] for name, tr in traits.items() if "midparent_slope" in tr
    }
    return view


def plotting_report_views(
    reports: list[dict[str, Any]],
    plot_payloads: list[dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Build plotting views for a list of v2 reports and their plot payloads."""
    if plot_payloads is None:
        plot_payloads = [None] * len(reports)
    return [plotting_report_view(r, pl) for r, pl in zip(reports, plot_payloads, strict=True)]


def truth_entry(entry: dict[str, Any] | None, generation: int | None) -> dict[str, Any] | None:
    """A view's recorded-pedigree truth *entry*, or its ``by_generation`` entry for *generation*."""
    if entry is None or generation is None:
        return entry
    return (entry.get("by_generation") or {}).get(generation)


def truth_mean(entries: list[Any], key: str) -> float | None:
    """Mean of ``entry[key]`` over the reps' truth entries that carry one, else None."""
    vals = [float(e[key]) for e in entries if e and e.get(key) is not None]
    return float(np.mean(vals)) if vals else None
