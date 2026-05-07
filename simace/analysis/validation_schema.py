"""Single source of truth mapping TSV columns to validation YAML paths.

`gather.extract_metrics` walks `METRIC_REGISTRY` to produce one column per entry.
The contract is enforced by `tests/analysis/test_validation_schema.py`: every
registered path must resolve to a non-`None` value when applied to a fully
populated coverage validation YAML, so a producer-side rename in
`simace/analysis/validate.py` cannot silently empty a TSV column.

Path-derived fields (scenario, rep, benchmark timing) and shallow
``parameters.*`` / ``summary.*`` extractions remain inline in
`extract_metrics` — they don't go through the YAML check tree.
"""

from __future__ import annotations

__all__ = ["METRIC_REGISTRY", "MetricSpec"]

from typing import NamedTuple


class MetricSpec(NamedTuple):
    """One row of METRIC_REGISTRY: TSV column name and nested YAML path."""

    column: str
    path: tuple[str, ...]


# fmt: off
METRIC_REGISTRY: list[MetricSpec] = [
    # ── twins ─────────────────────────────────────────────────────────────
    MetricSpec("observed_twin_rate", ("twins", "twin_rate", "observed_rate")),

    # ── statistical: founder variance components ──────────────────────────
    MetricSpec("variance_A1", ("statistical", "variance_A1", "observed")),
    MetricSpec("variance_C1", ("statistical", "variance_C1", "observed")),
    MetricSpec("variance_E1", ("statistical", "variance_E1", "observed")),
    MetricSpec("variance_A2", ("statistical", "variance_A2", "observed")),
    MetricSpec("variance_C2", ("statistical", "variance_C2", "observed")),
    MetricSpec("variance_E2", ("statistical", "variance_E2", "observed")),

    # ── statistical: cross-trait correlations ─────────────────────────────
    MetricSpec("observed_rA", ("statistical", "cross_trait_rA", "observed")),
    MetricSpec("observed_rC", ("statistical", "cross_trait_rC", "observed")),
    MetricSpec("observed_rE", ("statistical", "cross_trait_rE", "observed")),

    # ── heritability: MZ twin correlations ────────────────────────────────
    MetricSpec("mz_twin_A1_corr", ("heritability", "mz_twin_A1_correlation", "observed")),
    MetricSpec("mz_twin_liability1_corr", ("heritability", "mz_twin_liability1_correlation", "observed")),
    MetricSpec("mz_twin_A2_corr", ("heritability", "mz_twin_A2_correlation", "observed")),
    MetricSpec("mz_twin_liability2_corr", ("heritability", "mz_twin_liability2_correlation", "observed")),

    # ── heritability: DZ sibling correlations ─────────────────────────────
    MetricSpec("dz_sibling_A1_corr", ("heritability", "dz_sibling_A1_correlation", "observed")),
    MetricSpec("dz_sibling_liability1_corr", ("heritability", "dz_sibling_liability1_correlation", "observed")),
    MetricSpec("dz_sibling_A2_corr", ("heritability", "dz_sibling_A2_correlation", "observed")),
    MetricSpec("dz_sibling_liability2_corr", ("heritability", "dz_sibling_liability2_correlation", "observed")),

    # ── heritability: Falconer + parent-offspring regressions ─────────────
    MetricSpec("falconer_h2_trait1", ("heritability", "falconer_estimate_trait1", "observed")),
    MetricSpec("falconer_h2_trait2", ("heritability", "falconer_estimate_trait2", "observed")),
    MetricSpec("parent_offspring_A1_slope", ("heritability", "parent_offspring_A1_regression", "slope")),
    MetricSpec("parent_offspring_A1_r2", ("heritability", "parent_offspring_A1_regression", "r_squared")),
    MetricSpec("parent_offspring_liability1_slope", ("heritability", "parent_offspring_liability1_regression", "slope")),
    MetricSpec("parent_offspring_liability1_r2", ("heritability", "parent_offspring_liability1_regression", "r_squared")),
    MetricSpec("parent_offspring_A2_slope", ("heritability", "parent_offspring_A2_regression", "slope")),
    MetricSpec("parent_offspring_A2_r2", ("heritability", "parent_offspring_A2_regression", "r_squared")),
    MetricSpec("parent_offspring_liability2_slope", ("heritability", "parent_offspring_liability2_regression", "slope")),
    MetricSpec("parent_offspring_liability2_r2", ("heritability", "parent_offspring_liability2_regression", "r_squared")),

    # ── half-sibs: structural counts ──────────────────────────────────────
    MetricSpec("half_sib_prop_observed", ("half_sibs", "half_sib_pair_proportion", "observed")),
    MetricSpec("offspring_with_half_sib_observed", ("half_sibs", "offspring_with_half_sib", "observed")),

    # ── half-sibs: variance-component correlations ────────────────────────
    # See `_validate_half_sib_correlations` in simace.analysis.validate for the
    # MHS+PHS pooling rule and PHS-only restriction.
    MetricSpec("half_sib_A1_corr", ("half_sibs", "half_sib_A1_correlation", "observed")),
    MetricSpec("half_sib_liability1_corr", ("half_sibs", "half_sib_liability1_correlation", "observed")),
    MetricSpec("half_sib_shared_C1", ("half_sibs", "half_sib_shared_C1", "observed")),
    MetricSpec("half_sib_A2_corr", ("half_sibs", "half_sib_A2_correlation", "observed")),
    MetricSpec("half_sib_liability2_corr", ("half_sibs", "half_sib_liability2_correlation", "observed")),
    MetricSpec("half_sib_shared_C2", ("half_sibs", "half_sib_shared_C2", "observed")),

    # ── assortative mating: mate correlations ─────────────────────────────
    MetricSpec("mate_corr_liability1", ("assortative_mating", "mate_corr_liability1", "observed")),
    MetricSpec("mate_corr_liability2", ("assortative_mating", "mate_corr_liability2", "observed")),

    # ── family size distribution ──────────────────────────────────────────
    MetricSpec("mother_mean_offspring", ("family_size_distribution", "mother", "mean")),
    MetricSpec("father_mean_offspring", ("family_size_distribution", "father", "mean")),

    # ── consanguineous matings ────────────────────────────────────────────
    MetricSpec("n_half_sib_matings", ("consanguineous_matings", "consanguineous_count", "n_half_sib_matings")),
    MetricSpec("n_full_sib_matings", ("consanguineous_matings", "consanguineous_count", "n_full_sib_matings")),
    MetricSpec("missing_gp_links", ("consanguineous_matings", "consanguineous_count", "total_missing_gp_links")),
    MetricSpec("gp_reconciled", ("consanguineous_matings", "grandparent_reconciliation", "passed")),
]
# fmt: on
