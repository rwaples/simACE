"""Helpers for reading grouped Stats-stage reports in plotting code."""

from typing import Any

REPORT_GROUPS = ("metadata", "incidence", "censoring", "pedigree", "correlations", "heritability")


def require_grouped_stats_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Return a grouped stats report or raise for old flat report files."""
    if not isinstance(report, dict):
        raise ValueError("Stats report is empty or invalid")
    missing = [group for group in REPORT_GROUPS if group not in report]
    if missing:
        raise ValueError(
            "Stats report does not use the grouped schema; missing top-level groups: " + ", ".join(missing)
        )
    return report


def plotting_stats_view(report: dict[str, Any] | None) -> dict[str, Any]:
    """Build the internal plotting view from a grouped stats report.

    This is not an old-schema compatibility layer: input must already be the
    grouped `stats_report.yaml` shape. The returned view preserves the
    existing plotting helpers' flat key names while report files remain
    grouped at the Interface.
    """
    grouped = require_grouped_stats_report(report)
    metadata = grouped["metadata"] or {}
    incidence = grouped["incidence"] or {}
    censoring = grouped["censoring"] or {}
    pedigree = grouped["pedigree"] or {}
    correlations = grouped["correlations"] or {}
    heritability = grouped["heritability"] or {}

    view: dict[str, Any] = {}
    view.update(metadata)
    view.update(incidence)
    view["person_years"] = censoring.get("person_years")
    if "windows" in censoring:
        view["censoring"] = censoring.get("windows")
    if "confusion" in censoring:
        view["censoring_confusion"] = censoring.get("confusion")
    if "cascade" in censoring:
        view["censoring_cascade"] = censoring.get("cascade")

    view["family_size"] = pedigree.get("family_size")
    view["pair_counts"] = pedigree.get("relationship_pair_counts")
    view["parent_status"] = pedigree.get("parent_status")
    full = pedigree.get("full") or {}
    if full:
        view["pair_counts_ped"] = full.get("relationship_pair_counts")
        view["n_individuals_ped"] = full.get("n_individuals")
        view["n_generations_ped"] = full.get("n_generations")

    view.update(correlations)
    view.update(heritability)
    return view


def plotting_stats_views(reports: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Build plotting views for a list of grouped stats reports."""
    return [plotting_stats_view(report) for report in reports]
