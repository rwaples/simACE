"""Assemble the curated v2 Analyze report from quality-check + stats outputs.

Pure functions that re-home the raw quality report and the analysis sample's
:class:`~simace.analysis.sample_stats.SampleStats` into the v2 scientific
shape (``inputs``, ``scopes``, ``quality_checks``, ``truth``, ``observed``,
``estimators``) plus the scope-organized ``plot_payload``. See ``docs/adr/0008-curated-analyze-report.md`` and
``report_schema.py`` for the contract.
"""

from __future__ import annotations

__all__ = ["assemble_report"]

from typing import TYPE_CHECKING, Any

from .report_schema import (
    PLOT_PAYLOAD_SCHEMA_NAME,
    PLOT_PAYLOAD_SCHEMA_VERSION,
    REPORT_SCHEMA_NAME,
    REPORT_SCHEMA_VERSION,
    assert_report_contract,
    partition_dense,
)

if TYPE_CHECKING:
    from .quality.assortative_mating import MateCorrelation
    from .quality.check import PairCorrelation
    from .quality.runner import QualityReport
    from .sample_stats import SampleStats


def normalize_quality_checks(quality: QualityReport) -> dict[str, Any]:
    """The ``quality_checks`` group: one uniform row per quality check, and their tally.

    All quality checks run on the full pre-ascertainment pedigree, so every
    row carries ``scope="recorded_pedigree"``. No check has a warn level, so
    ``n_warn`` is always 0.
    """
    checks = quality.rows()
    n_failed = sum(row["status"] == "fail" for row in checks)
    return {
        "summary": {
            "passed": n_failed == 0,
            "n_passed": len(checks) - n_failed,
            "n_failed": n_failed,
            "n_warn": 0,
        },
        "checks": checks,
    }


def extract_truth(
    quality: QualityReport, params: dict[str, Any], recorded_truth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Re-home generated/realized ground-truth quantities under ``recorded_pedigree``.

    ``recorded_truth`` is :func:`simace.analysis.truth.recorded_pedigree_truth`'s
    result, present only under C transmission: its per-trait
    ``relationships`` and ``midparent_slope`` join each trait, and its
    ``limitations`` list sits beside the traits (ADR 0025).
    """
    stat = quality.statistical.measurements
    per_gen = quality.per_generation
    pop = quality.population.measurements
    twins = quality.twins.measurements
    half = quality.half_sibs.measurements
    am = quality.assortative_mating.measurements
    cons = quality.consanguineous_matings
    reconciliation = cons.find("grandparent_reconciliation")

    traits: dict[str, Any] = {}
    for t in (1, 2):
        founders = stat.traits[t - 1]
        var_a, var_c, var_e, var_liability = founders.var_A, founders.var_C, founders.var_E, founders.var_liability
        h2 = var_a / var_liability if var_liability else None
        realized_by_generation = {
            gen: {
                "n": g.get("n"),
                "var_A": g.get(f"A{t}_var"),
                "var_C": g.get(f"C{t}_var"),
                "var_E": g.get(f"E{t}_var"),
                "var_liability": g.get(f"liability{t}_variance"),
                "cov_A_non_genetic": g.get(f"A{t}_cov_non_genetic"),
                "cov_A_C": g.get(f"A{t}_cov_C"),
                "cov_A_E": g.get(f"A{t}_cov_E"),
            }
            for gen, g in per_gen.items()
        }
        traits[f"trait{t}"] = {
            "configured": {"A": params.get(f"A{t}"), "C": params.get(f"C{t}"), "E": params.get(f"E{t}")},
            "realized": {
                "var_A": var_a,
                "var_C": var_c,
                "var_E": var_e,
                "var_liability": var_liability,
                "h2_liability": h2,
            },
            "realized_by_generation": realized_by_generation,
        }

    half_sibs: dict[str, Any] = {
        "pair_proportion": half.pair_proportion,
        "offspring_with_half_sib": half.offspring_with_half_sib,
    }
    for t, sibs in enumerate(half.traits, start=1):
        half_sibs[f"trait{t}"] = {
            "A_corr": _observed(sibs.A_corr),
            "liability_corr": _observed(sibs.liability_corr),
            "shared_C": _observed(sibs.shared_C),
        }

    family_structure = {
        "twin_rate": {"observed": twins.twin_rate, "expected": twins.expected_twin_rate},
        "half_sibs": half_sibs,
        "consanguineous_matings": {
            "n_half_sib_matings": cons.measurements.n_half_sib_matings,
            "n_full_sib_matings": cons.measurements.n_full_sib_matings,
            "total_missing_gp_links": cons.measurements.total_missing_gp_links,
            "grandparent_reconciliation_passed": None if reconciliation is None else reconciliation.passed,
        },
        "offspring_distribution": quality.family_size_distribution,
    }

    population = {
        "generation_sizes": pop.generation_sizes,
        "n_generations": pop.n_generations,
        "sex_ratio_male": quality.structural.measurements.sex_ratio_male,
    }

    recorded: dict[str, Any] = {}
    if recorded_truth is not None:
        for name, trait in recorded_truth["traits"].items():
            traits[name].update(trait)
        recorded["limitations"] = recorded_truth["limitations"]
        if "c_transmission_clips" in recorded_truth:
            recorded["c_transmission_clips"] = recorded_truth["c_transmission_clips"]

    return {
        "recorded_pedigree": {
            **recorded,
            "population": population,
            "traits": traits,
            "family_structure": family_structure,
            "cross_trait": {"rA": stat.rA, "rC": stat.rC, "rE": stat.rE},
            "assortative_mating": {
                "mate_corr_liability1": _observed(am.liability[0]),
                "mate_corr_liability2": _observed(am.liability[1]),
                "mate_corr_A1": am.genetic[0],
                "mate_corr_A2": am.genetic[1],
            },
        }
    }


def _observed(measured: PairCorrelation | MateCorrelation | None) -> float | None:
    return None if measured is None else measured.observed


def extract_estimators(quality: QualityReport, observed_h2_estimators: dict[str, Any]) -> dict[str, Any]:
    """Heritability estimators: relationship/outcome-derived, by scale."""
    liability_scale: dict[str, Any] = {}
    for t, her in enumerate(quality.heritability.measurements.traits, start=1):
        liability_scale[f"trait{t}"] = {
            "mz_twin_A_corr": _observed(her.mz_A_corr),
            "mz_twin_liability_corr": _observed(her.mz_liability_corr),
            "dz_sibling_A_corr": _observed(her.dz_A_corr),
            "dz_sibling_liability_corr": _observed(her.dz_liability_corr),
            "falconer": her.falconer,
            "parent_offspring_A_slope": None if her.po_A is None else her.po_A.slope,
            "parent_offspring_A_r2": None if her.po_A is None else her.po_A.r_squared,
            "parent_offspring_liability_slope": None if her.po_liability is None else her.po_liability.slope,
            "parent_offspring_liability_r2": None if her.po_liability is None else her.po_liability.r_squared,
        }

    return {"heritability": {"observed_scale": observed_h2_estimators, "liability_scale": liability_scale}}


def build_ascertainment_summary(
    prevalence_phenotyped: dict[str, Any],
    prevalence_sample: dict[str, Any],
    scope_counts: dict[str, Any],
) -> dict[str, Any]:
    """Summarize how ascertainment distorts the phenotyped population.

    Carries only the distortion story (retained fraction, per-trait
    case-enrichment). Raw scope sizes and the ancestor-closure ratio live once
    in ``scopes`` (their canonical home) and are not restated here. Configured
    knobs likewise live once in ``inputs.ascertainment``.
    """
    pheno_n = (scope_counts.get("phenotyped_population") or {}).get("n_individuals")
    sample_n = (scope_counts.get("analysis_sample") or {}).get("n_individuals")
    trait_enrichment: dict[str, Any] = {}
    for t in (1, 2):
        before = prevalence_phenotyped.get(f"trait{t}")
        after = prevalence_sample.get(f"trait{t}")
        trait_enrichment[f"trait{t}"] = {
            "affected_fraction_before": before,
            "affected_fraction_after": after,
            "enrichment_ratio": (after / before) if before else None,
        }
    return {
        "retained_fraction": (sample_n / pheno_n) if pheno_n else None,
        "trait_enrichment": trait_enrichment,
    }


def _build_inputs(params: dict[str, Any], case_ascertainment_ratio: float) -> dict[str, Any]:
    trait_model = {
        f"trait{t}": {"A": params.get(f"A{t}"), "C": params.get(f"C{t}"), "E": params.get(f"E{t}")} for t in (1, 2)
    }
    return {
        "parameters": params,
        "trait_model": trait_model,
        "ascertainment": {
            "dropout_rate": params.get("dropout_rate"),
            "case_ascertainment_ratio": case_ascertainment_ratio,
            "N_sample": params.get("N_sample"),
        },
    }


def assemble_report(
    *,
    replicate: dict[str, Any],
    params: dict[str, Any],
    case_ascertainment_ratio: float,
    quality: QualityReport,
    stats: SampleStats,
    prevalence_phenotyped: dict[str, Any],
    scope_counts: dict[str, Any],
    recorded_truth: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the v2 ``report.yaml`` dict and its scope-organized ``plot_payload``.

    ``recorded_truth`` is the recorded-pedigree relationship truth, present
    only under C transmission (see :func:`extract_truth`).

    Returns ``(report, plot_payload)``.
    """
    prevalence_sample = stats.analysis_sample.prevalence
    ascertainment = build_ascertainment_summary(prevalence_phenotyped, prevalence_sample, scope_counts)

    # Split dense plot arrays out of every scope in one pass: the scalar half is
    # the scientific report, the dense half becomes the plot payload (keyed by
    # the same scopes, so the payload mirrors observed). This makes the
    # report's no-dense-arrays guarantee structural rather than per-scope.
    observed_full = {
        "ascertainment": ascertainment,
        "phenotyped_population": {"prevalence": prevalence_phenotyped},
        "analysis_sample": stats.analysis_sample.to_report(),
        "analysis_pedigree": stats.analysis_pedigree.to_report(),
    }
    observed, observed_dense = partition_dense(observed_full)

    report = {
        "schema": {"name": REPORT_SCHEMA_NAME, "version": REPORT_SCHEMA_VERSION},
        "replicate": replicate,
        "inputs": _build_inputs(params, case_ascertainment_ratio),
        "scopes": scope_counts,
        "quality_checks": normalize_quality_checks(quality),
        "truth": extract_truth(quality, params, recorded_truth),
        "observed": observed,
        "estimators": extract_estimators(quality, stats.observed_h2_estimators),
    }
    assert_report_contract(report)

    plot_payload = {
        "schema": {"name": PLOT_PAYLOAD_SCHEMA_NAME, "version": PLOT_PAYLOAD_SCHEMA_VERSION},
        "replicate": {k: replicate.get(k) for k in ("folder", "scenario", "rep")},
        **observed_dense,
    }
    return report, plot_payload
