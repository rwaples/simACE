"""The plotting view of one rep: its v2 report (+ plot payload), parsed once into a :class:`ReportView`.

The figures read a rep's stats as attributes of the same
:class:`~simace.analysis.sample_stats.SampleStats` records Analyze wrote the
report from (ADR 0008), so a stat cannot exist on one side only. A field is
``None`` only where a correct run can omit it: the censoring tables without
per-generation censoring windows, and a recorded-pedigree truth entry
without C transmission. Without a plot payload the dense curve arrays are
absent from their tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from simace.analysis.sample_stats import SampleStats
from simace.simulation.params import SimulationParams

if TYPE_CHECKING:
    from simace.analysis.sample_stats import AnalysisPedigreeStats, AnalysisSampleStats, ScopeSize

__all__ = [
    "ReportView",
    "plotting_report_view",
    "plotting_report_views",
    "report_per_generation",
    "truth_entry",
    "truth_mean",
]


@dataclass(frozen=True)
class ReportView:
    """One rep's report, as the figures read it.

    Attributes:
        sample_size: The analysis sample's size.
        pedigree_size: The analysis pedigree's size.
        sample: ``observed.analysis_sample``, dense arrays merged back from the payload.
        pedigree: ``observed.analysis_pedigree``.
        observed_h2_estimators: ``estimators.heritability.observed_scale``.
        per_generation: The recorded pedigree's realized moments by
            generation, in the flat ``generation_N -> {A1_var, ...}`` shape
            (:func:`report_per_generation`).
        parameters: ``inputs.parameters``, as written.
        sim: The parameters as the record the reference lines read.
        truth_relationships: Per trait, the recorded-pedigree relationship
            truth; empty without C transmission (ADR 0025).
        truth_midparent_slope: Per trait, the recorded-pedigree midparent
            slope truth; empty without C transmission.
    """

    sample_size: ScopeSize
    pedigree_size: ScopeSize
    sample: AnalysisSampleStats
    pedigree: AnalysisPedigreeStats
    observed_h2_estimators: dict[str, Any]
    per_generation: dict[str, dict[str, Any]]
    parameters: dict[str, Any]
    sim: SimulationParams
    truth_relationships: dict[str, Any]
    truth_midparent_slope: dict[str, Any]


def report_per_generation(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reconstruct the flat ``generation_N -> {A1_var, C1_var, …}`` table from truth.

    Inverse of the per-trait ``truth.recorded_pedigree.traits.*.realized_by_generation``
    layout, matching what the per-generation heritability plots expect.
    """
    traits = report["truth"]["recorded_pedigree"]["traits"]
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


def plotting_report_view(report: dict[str, Any], plot_payload: dict[str, Any] | None = None) -> ReportView:
    """Parse a v2 report and its plot payload into the plotting view.

    ``plot_payload`` may be omitted when only scalar fields are needed (e.g.
    ad-hoc cross-scenario tools); curve arrays will then be absent.
    """
    stats = SampleStats.from_report(report, plot_payload)
    parameters = report["inputs"]["parameters"]
    traits = report["truth"]["recorded_pedigree"]["traits"]
    return ReportView(
        sample_size=stats.sample_size,
        pedigree_size=stats.pedigree_size,
        sample=stats.analysis_sample,
        pedigree=stats.analysis_pedigree,
        observed_h2_estimators=stats.observed_h2_estimators,
        per_generation=report_per_generation(report),
        parameters=parameters,
        sim=SimulationParams.from_mapping(parameters),
        truth_relationships={name: tr["relationships"] for name, tr in traits.items() if "relationships" in tr},
        truth_midparent_slope={name: tr["midparent_slope"] for name, tr in traits.items() if "midparent_slope" in tr},
    )


def plotting_report_views(
    reports: list[dict[str, Any]],
    plot_payloads: list[dict[str, Any] | None] | None = None,
) -> list[ReportView]:
    """Parse a list of v2 reports and their plot payloads into plotting views."""
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
