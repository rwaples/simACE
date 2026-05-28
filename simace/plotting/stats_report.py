"""Adapter from the curated v2 report (+ plot payload) to the flat plotting view.

The plot helpers consume a single flat dict of keys (``cumulative_incidence``,
``censoring``, ``family_size``, ``tetrachoric``, ``per_generation``,
``parameters``, …). This module rebuilds that view from the v2 ``report.yaml``
scope-organized groups and the dense arrays in ``plot_payload.yaml``, so the
plot code does not need to know about the v2 scientific grouping.
"""

from typing import Any

__all__ = [
    "plotting_report_view",
    "plotting_report_views",
    "report_per_generation",
]

# analysis_sample keys that map straight through to the flat view by name.
_SAMPLE_PASSTHROUGH = (
    "prevalence",
    "mortality",
    "regression",
    "cumulative_incidence",
    "cumulative_incidence_by_sex",
    "cumulative_incidence_by_sex_generation",
    "cumulative_incidence_aj",
    "cumulative_incidence_aj_by_sex",
    "cumulative_incidence_aj_by_sex_generation",
    "joint_affection",
    "liability_correlations",
    "affected_correlations",
    "parent_offspring_corr",
    "parent_offspring_corr_by_sex",
    "parent_offspring_affected_corr",
    "tetrachoric",
    "tetrachoric_by_generation",
    "tetrachoric_by_sex",
    "cross_trait_tetrachoric",
)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into a copy of ``base`` (``extra`` wins)."""
    merged = dict(base)
    for key, value in extra.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


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
            entry[f"A{t}_var"] = vals.get("var_A")
            entry[f"C{t}_var"] = vals.get("var_C")
            entry[f"E{t}_var"] = vals.get("var_E")
            entry[f"liability{t}_variance"] = vals.get("var_liability")
    return out


def plotting_report_view(report: dict[str, Any] | None, plot_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the flat plotting view from a v2 report and its plot payload.

    ``plot_payload`` may be omitted when only scalar fields are needed (e.g.
    ad-hoc cross-scenario tools); curve arrays will then be absent.
    """
    report = report or {}
    observed = report.get("observed") or {}
    scopes = report.get("scopes") or {}

    sample = observed.get("analysis_sample") or {}
    if plot_payload:
        sample = _deep_merge(sample, plot_payload.get("analysis_sample") or {})
    pedigree = observed.get("analysis_pedigree") or {}

    view: dict[str, Any] = {}
    view["n_individuals"] = (scopes.get("analysis_sample") or {}).get("n_individuals")
    view["n_generations"] = (scopes.get("analysis_sample") or {}).get("n_generations")

    for key in _SAMPLE_PASSTHROUGH:
        if key in sample:
            view[key] = sample[key]

    view["person_years"] = sample.get("person_years")
    if "censoring_windows" in sample:
        view["censoring"] = sample["censoring_windows"]
    if "censoring_confusion" in sample:
        view["censoring_confusion"] = sample["censoring_confusion"]
    if "censoring_cascade" in sample:
        view["censoring_cascade"] = sample["censoring_cascade"]
    view["family_size"] = sample.get("family_size")
    view["pair_counts"] = sample.get("relationship_pair_counts")

    view["parent_status"] = pedigree.get("parent_status")
    if "mate_correlation" in pedigree:
        view["mate_correlation"] = pedigree["mate_correlation"]
    full_counts = pedigree.get("relationship_pair_counts")
    if full_counts is not None:
        view["pair_counts_ped"] = full_counts
        view["n_individuals_ped"] = (scopes.get("analysis_pedigree") or {}).get("n_individuals")
        view["n_generations_ped"] = (scopes.get("analysis_pedigree") or {}).get("n_generations")

    view["observed_h2_estimators"] = ((report.get("estimators") or {}).get("heritability") or {}).get("observed_scale")
    view["per_generation"] = report_per_generation(report)
    view["parameters"] = (report.get("inputs") or {}).get("parameters", {})
    return view


def plotting_report_views(
    reports: list[dict[str, Any] | None],
    plot_payloads: list[dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Build plotting views for a list of v2 reports and their plot payloads."""
    if plot_payloads is None:
        plot_payloads = [None] * len(reports)
    return [plotting_report_view(r, pl) for r, pl in zip(reports, plot_payloads, strict=True)]
