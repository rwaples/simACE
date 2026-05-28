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
    "REPORT_SUMMARY_REGISTRY",
    "REPORT_TOP_LEVEL_GROUPS",
    "SCOPES",
    "MetricSpec",
    "assert_report_contract",
    "find_dense_keys",
    "partition_dense",
]

from typing import TYPE_CHECKING, Any, NamedTuple

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


# ---------------------------------------------------------------------------
# Report-summary registry: wide TSV columns -> v2 report paths.
#
# `gather.extract_metrics` walks REPORT_SUMMARY_REGISTRY to emit one column per
# entry, each path relative to the report root. The contract is enforced by
# tests/analysis/test_report_summary.py: every path must resolve to a non-None
# value against a fully populated coverage report. Identity (folder, scenario,
# rep, seed), quality counts, and benchmark timing are added inline by
# extract_metrics — they don't go through the registry tree.
# ---------------------------------------------------------------------------


class MetricSpec(NamedTuple):
    """One row of REPORT_SUMMARY_REGISTRY: TSV column name and report path."""

    column: str
    path: tuple[str, ...]


_TRUTH = ("truth", "recorded_pedigree")
_FS = (*_TRUTH, "family_structure")
_LIAB = ("estimators", "heritability", "liability_scale")
_SCOPES = ("scopes",)
_ASC = ("observed", "ascertainment")

# fmt: off
REPORT_SUMMARY_REGISTRY: list[MetricSpec] = [
    # ── scope sizes ───────────────────────────────────────────────────────
    MetricSpec("recorded_pedigree_n", (*_SCOPES, "recorded_pedigree", "n_individuals")),
    MetricSpec("phenotyped_population_n", (*_SCOPES, "phenotyped_population", "n_individuals")),
    MetricSpec("analysis_sample_n", (*_SCOPES, "analysis_sample", "n_individuals")),
    MetricSpec("analysis_pedigree_n", (*_SCOPES, "analysis_pedigree", "n_individuals")),
    MetricSpec("ancestor_closure_ratio", (*_SCOPES, "analysis_pedigree", "ancestor_closure_ratio")),

    # ── ascertainment distortion ──────────────────────────────────────────
    MetricSpec("retained_fraction", (*_ASC, "counts", "retained_fraction")),
    MetricSpec("trait1_affected_before", (*_ASC, "trait_enrichment", "trait1", "affected_fraction_before")),
    MetricSpec("trait1_affected_after", (*_ASC, "trait_enrichment", "trait1", "affected_fraction_after")),
    MetricSpec("trait2_affected_before", (*_ASC, "trait_enrichment", "trait2", "affected_fraction_before")),
    MetricSpec("trait2_affected_after", (*_ASC, "trait_enrichment", "trait2", "affected_fraction_after")),

    # ── twins ─────────────────────────────────────────────────────────────
    MetricSpec("observed_twin_rate", (*_FS, "twin_rate", "observed")),
    MetricSpec("expected_twin_rate", (*_FS, "twin_rate", "expected")),

    # ── truth: realized variance components ───────────────────────────────
    MetricSpec("variance_A1", (*_TRUTH, "traits", "trait1", "realized", "var_A")),
    MetricSpec("variance_C1", (*_TRUTH, "traits", "trait1", "realized", "var_C")),
    MetricSpec("variance_E1", (*_TRUTH, "traits", "trait1", "realized", "var_E")),
    MetricSpec("variance_A2", (*_TRUTH, "traits", "trait2", "realized", "var_A")),
    MetricSpec("variance_C2", (*_TRUTH, "traits", "trait2", "realized", "var_C")),
    MetricSpec("variance_E2", (*_TRUTH, "traits", "trait2", "realized", "var_E")),

    # ── truth: cross-trait correlations ───────────────────────────────────
    MetricSpec("observed_rA", (*_TRUTH, "cross_trait", "rA")),
    MetricSpec("observed_rC", (*_TRUTH, "cross_trait", "rC")),
    MetricSpec("observed_rE", (*_TRUTH, "cross_trait", "rE")),

    # ── estimators: MZ twin correlations ──────────────────────────────────
    MetricSpec("mz_twin_A1_corr", (*_LIAB, "trait1", "mz_twin_A_corr")),
    MetricSpec("mz_twin_liability1_corr", (*_LIAB, "trait1", "mz_twin_liability_corr")),
    MetricSpec("mz_twin_A2_corr", (*_LIAB, "trait2", "mz_twin_A_corr")),
    MetricSpec("mz_twin_liability2_corr", (*_LIAB, "trait2", "mz_twin_liability_corr")),

    # ── estimators: DZ sibling correlations ───────────────────────────────
    MetricSpec("dz_sibling_A1_corr", (*_LIAB, "trait1", "dz_sibling_A_corr")),
    MetricSpec("dz_sibling_liability1_corr", (*_LIAB, "trait1", "dz_sibling_liability_corr")),
    MetricSpec("dz_sibling_A2_corr", (*_LIAB, "trait2", "dz_sibling_A_corr")),
    MetricSpec("dz_sibling_liability2_corr", (*_LIAB, "trait2", "dz_sibling_liability_corr")),

    # ── estimators: Falconer + parent-offspring regressions ───────────────
    MetricSpec("falconer_h2_trait1", (*_LIAB, "trait1", "falconer")),
    MetricSpec("falconer_h2_trait2", (*_LIAB, "trait2", "falconer")),
    MetricSpec("parent_offspring_A1_slope", (*_LIAB, "trait1", "parent_offspring_A_slope")),
    MetricSpec("parent_offspring_A1_r2", (*_LIAB, "trait1", "parent_offspring_A_r2")),
    MetricSpec("parent_offspring_liability1_slope", (*_LIAB, "trait1", "parent_offspring_liability_slope")),
    MetricSpec("parent_offspring_liability1_r2", (*_LIAB, "trait1", "parent_offspring_liability_r2")),
    MetricSpec("parent_offspring_A2_slope", (*_LIAB, "trait2", "parent_offspring_A_slope")),
    MetricSpec("parent_offspring_A2_r2", (*_LIAB, "trait2", "parent_offspring_A_r2")),
    MetricSpec("parent_offspring_liability2_slope", (*_LIAB, "trait2", "parent_offspring_liability_slope")),
    MetricSpec("parent_offspring_liability2_r2", (*_LIAB, "trait2", "parent_offspring_liability_r2")),

    # ── truth: half-sib structure ─────────────────────────────────────────
    MetricSpec("half_sib_prop_observed", (*_FS, "half_sibs", "pair_proportion")),
    MetricSpec("offspring_with_half_sib_observed", (*_FS, "half_sibs", "offspring_with_half_sib")),
    MetricSpec("half_sib_A1_corr", (*_FS, "half_sibs", "trait1", "A_corr")),
    MetricSpec("half_sib_liability1_corr", (*_FS, "half_sibs", "trait1", "liability_corr")),
    MetricSpec("half_sib_shared_C1", (*_FS, "half_sibs", "trait1", "shared_C")),
    MetricSpec("half_sib_A2_corr", (*_FS, "half_sibs", "trait2", "A_corr")),
    MetricSpec("half_sib_liability2_corr", (*_FS, "half_sibs", "trait2", "liability_corr")),
    MetricSpec("half_sib_shared_C2", (*_FS, "half_sibs", "trait2", "shared_C")),

    # ── truth: assortative mating ─────────────────────────────────────────
    MetricSpec("mate_corr_liability1", (*_TRUTH, "assortative_mating", "mate_corr_liability1")),
    MetricSpec("mate_corr_liability2", (*_TRUTH, "assortative_mating", "mate_corr_liability2")),

    # ── truth: offspring distribution ─────────────────────────────────────
    MetricSpec("mother_mean_offspring", (*_FS, "offspring_distribution", "mother", "mean")),
    MetricSpec("father_mean_offspring", (*_FS, "offspring_distribution", "father", "mean")),

    # ── truth: consanguineous matings ─────────────────────────────────────
    MetricSpec("n_half_sib_matings", (*_FS, "consanguineous_matings", "n_half_sib_matings")),
    MetricSpec("n_full_sib_matings", (*_FS, "consanguineous_matings", "n_full_sib_matings")),
    MetricSpec("missing_gp_links", (*_FS, "consanguineous_matings", "total_missing_gp_links")),
    MetricSpec("gp_reconciled", (*_FS, "consanguineous_matings", "grandparent_reconciliation_passed")),
]
# fmt: on
