"""Single source of truth mapping TSV columns to v2 ``report.yaml`` paths.

`gather.extract_metrics` walks `METRIC_REGISTRY` to produce one column per entry.
Every registered path is relative to the report root and resolves into the
curated ``truth`` / ``estimators`` groups (ADR 0008). The contract is enforced
by `tests/analysis/test_validation_schema.py`: every registered path must
resolve to a non-`None` value against a fully populated coverage report, so a
producer-side rename in `validate.py` / `report.py` cannot silently empty a
column.

Path-derived fields (scenario, rep, benchmark timing) and shallow
``inputs.parameters.*`` / ``quality_checks.summary.*`` extractions remain inline
in `extract_metrics` — they don't go through the registry tree.
"""

from __future__ import annotations

__all__ = ["METRIC_REGISTRY", "MetricSpec"]

from typing import NamedTuple


class MetricSpec(NamedTuple):
    """One row of METRIC_REGISTRY: TSV column name and nested YAML path."""

    column: str
    path: tuple[str, ...]


_TRUTH = ("truth", "recorded_pedigree")
_FS = (*_TRUTH, "family_structure")
_LIAB = ("estimators", "heritability", "liability_scale")

# fmt: off
METRIC_REGISTRY: list[MetricSpec] = [
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
