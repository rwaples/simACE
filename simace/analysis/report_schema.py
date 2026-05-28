"""Schema constants and contract checks for the curated Analyze report (v2).

The Analyze stage writes two durable per-replicate artifacts:

- ``report.yaml`` — a curated scientific report (this module's ``v2`` schema):
  ``schema``, ``replicate``, ``inputs``, ``scopes``, ``quality_checks``,
  ``truth``, ``observed``, ``estimators``. Scalars, small categorical tables,
  and by-generation summaries only — never dense plot arrays.
- ``plot_payload.yaml`` — the dense incidence/censoring arrays needed only to
  render plots, organized by scope to mirror ``observed``.

See ``docs/adr/0008-curated-analyze-report.md``.
"""

from __future__ import annotations

__all__ = [
    "DENSE_ARRAY_KEYS",
    "PLOT_PAYLOAD_SCHEMA_NAME",
    "PLOT_PAYLOAD_SCHEMA_VERSION",
    "REPORT_SCHEMA_NAME",
    "REPORT_SCHEMA_VERSION",
    "REPORT_TOP_LEVEL_GROUPS",
    "SCOPES",
    "assert_report_contract",
    "find_dense_keys",
    "partition_dense",
]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

REPORT_SCHEMA_NAME = "simace_report"
REPORT_SCHEMA_VERSION = 2
PLOT_PAYLOAD_SCHEMA_NAME = "simace_plot_payload"
PLOT_PAYLOAD_SCHEMA_VERSION = 1

REPORT_TOP_LEVEL_GROUPS = (
    "schema",
    "replicate",
    "inputs",
    "scopes",
    "quality_checks",
    "truth",
    "observed",
    "estimators",
)

# The four canonical population scopes (CONTEXT.md vocabulary).
SCOPES = (
    "recorded_pedigree",
    "phenotyped_population",
    "analysis_sample",
    "analysis_pedigree",
)

# Leaf keys whose values are dense per-replicate arrays (incidence curves,
# censoring-window incidence) used only to render plots. They live in
# plot_payload.yaml, never in report.yaml.
DENSE_ARRAY_KEYS = frozenset(
    {
        "ages",
        "values",
        "observed_values",
        "true_values",
        "aj_values",
        "aj_death_values",
        "aj_survival",
        "aj_se",
        "censoring_ages",
        "true_incidence",
        "observed_incidence",
    }
)


def partition_dense(node: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recursively split a nested dict into (scalar_part, dense_part).

    Leaf keys in ``DENSE_ARRAY_KEYS`` go to ``dense_part``; everything else
    (scalars, small tables, decade-rate lists) stays in ``scalar_part``. Both
    halves mirror the input nesting; empty branches are dropped.
    """
    scalar: dict[str, Any] = {}
    dense: dict[str, Any] = {}
    for key, value in node.items():
        if key in DENSE_ARRAY_KEYS:
            dense[key] = value
        elif isinstance(value, dict):
            sub_scalar, sub_dense = partition_dense(value)
            if sub_scalar:
                scalar[key] = sub_scalar
            if sub_dense:
                dense[key] = sub_dense
        else:
            scalar[key] = value
    return scalar, dense


def find_dense_keys(node: Any, path: str = "") -> Iterator[str]:
    """Yield dotted paths of any ``DENSE_ARRAY_KEYS`` found anywhere in ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in DENSE_ARRAY_KEYS:
                yield here
            yield from find_dense_keys(value, here)


def assert_report_contract(report: dict[str, Any]) -> None:
    """Validate a report against the v2 contract, raising ``ValueError`` on breach.

    Checks the schema name/version, that all top-level groups are present, and
    that no dense plot-array keys leaked into the scientific report.
    """
    schema = report.get("schema") or {}
    if schema.get("name") != REPORT_SCHEMA_NAME or schema.get("version") != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Report schema must be {REPORT_SCHEMA_NAME} v{REPORT_SCHEMA_VERSION}, got {schema!r}"
        )
    missing = [group for group in REPORT_TOP_LEVEL_GROUPS if group not in report]
    if missing:
        raise ValueError("Report is missing top-level groups: " + ", ".join(missing))
    dense = list(find_dense_keys(report))
    if dense:
        raise ValueError("Report contains dense plot-array keys (belong in plot_payload): " + ", ".join(dense))
