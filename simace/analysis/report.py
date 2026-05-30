"""Assemble the curated v2 Analyze report from validation + stats outputs.

Pure functions that re-home the raw validation report and the six stats groups
into the v2 scientific shape (``inputs``, ``scopes``, ``quality_checks``,
``truth``, ``observed``, ``estimators``) plus the scope-organized
``plot_payload``. See ``docs/adr/0008-curated-analyze-report.md`` and
``report_schema.py`` for the contract.
"""

from __future__ import annotations

__all__ = ["assemble_report"]

from typing import Any

from .report_schema import (
    ANALYSIS_SAMPLE_CORRELATION_KEYS,
    ANALYSIS_SAMPLE_INCIDENCE_KEYS,
    ANALYSIS_SAMPLE_PEDIGREE_KEYS,
    CENSORING_RENAME,
    PLOT_PAYLOAD_SCHEMA_NAME,
    PLOT_PAYLOAD_SCHEMA_VERSION,
    REPORT_SCHEMA_NAME,
    REPORT_SCHEMA_VERSION,
    assert_report_contract,
    partition_dense,
)

# Skip these when normalizing validation checks — they carry no pass/fail rows.
_NON_CHECK_CATEGORIES = frozenset({"per_generation", "summary", "family_size_distribution", "parameters"})


def _single(result: dict[str, Any], prefix: str) -> Any:
    """Return the value of the one key starting with ``prefix`` (or None).

    Validator results carry at most one ``observed_*`` / ``expected_*`` field
    each, so a uniform check row can record it unambiguously. If a result ever
    grows two matching keys, picking one by dict order would be silently
    arbitrary — raise instead so the field map is made explicit.
    """
    hits = [value for key, value in result.items() if key.startswith(prefix)]
    if len(hits) > 1:
        matched = sorted(k for k in result if k.startswith(prefix))
        raise ValueError(f"Ambiguous {prefix!r} fields in check result: {matched}")
    return hits[0] if hits else None


def normalize_quality_checks(validation_report: dict[str, Any]) -> dict[str, Any]:
    """Flatten heterogeneous validation results into uniform check rows.

    All validation checks run on the full pre-ascertainment pedigree, so every
    row carries ``scope="recorded_pedigree"``. The current validator has no
    warn level, so ``n_warn`` is always 0.
    """
    checks: list[dict[str, Any]] = []
    n_passed = 0
    n_failed = 0
    for category, items in validation_report.items():
        if category in _NON_CHECK_CATEGORIES or not isinstance(items, dict):
            continue
        for name, result in items.items():
            # Skip informational metrics (no closed-form pass/fail): those
            # flagged via validate._info, or any dict lacking a ``passed`` key.
            if not isinstance(result, dict) or result.get("informational") or "passed" not in result:
                continue
            passed = bool(result["passed"])
            n_passed += passed
            n_failed += not passed
            checks.append(
                {
                    "id": f"{category}.{name}",
                    "scope": "recorded_pedigree",
                    "severity": "error",
                    "status": "pass" if passed else "fail",
                    "observed": _single(result, "observed"),
                    "expected": _single(result, "expected"),
                    "tolerance": result.get("tolerance"),
                    "message": result.get("details"),
                }
            )
    return {
        "summary": {
            "passed": n_failed == 0,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "n_warn": 0,
        },
        "checks": checks,
    }


def extract_truth(validation_report: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Re-home generated/realized ground-truth quantities under ``recorded_pedigree``."""
    stat = validation_report.get("statistical", {})
    per_gen = validation_report.get("per_generation", {})
    pop = validation_report.get("population", {})
    twins = validation_report.get("twins", {})
    half = validation_report.get("half_sibs", {})
    am = validation_report.get("assortative_mating", {})
    cons = validation_report.get("consanguineous_matings", {})
    structural = validation_report.get("structural", {})

    def obs(group: dict[str, Any], key: str) -> Any:
        return (group.get(key) or {}).get("observed")

    traits: dict[str, Any] = {}
    for t in (1, 2):
        var_a = obs(stat, f"variance_A{t}")
        var_c = obs(stat, f"variance_C{t}")
        var_e = obs(stat, f"variance_E{t}")
        var_liability = obs(stat, f"total_variance_trait{t}")
        h2 = var_a / var_liability if (var_a is not None and var_liability) else None
        realized_by_generation = {
            gen: {
                "var_A": g.get(f"A{t}_var"),
                "var_C": g.get(f"C{t}_var"),
                "var_E": g.get(f"E{t}_var"),
                "var_liability": g.get(f"liability{t}_variance"),
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

    half_sibs = {
        "pair_proportion": obs(half, "half_sib_pair_proportion"),
        "offspring_with_half_sib": obs(half, "offspring_with_half_sib"),
    }
    for t in (1, 2):
        half_sibs[f"trait{t}"] = {
            "A_corr": obs(half, f"half_sib_A{t}_correlation"),
            "liability_corr": obs(half, f"half_sib_liability{t}_correlation"),
            "shared_C": obs(half, f"half_sib_shared_C{t}"),
        }

    cons_count = cons.get("consanguineous_count") or {}
    family_structure = {
        "twin_rate": {
            "observed": (twins.get("twin_rate") or {}).get("observed_rate"),
            "expected": (twins.get("twin_rate") or {}).get("expected_rate"),
        },
        "half_sibs": half_sibs,
        "consanguineous_matings": {
            "n_half_sib_matings": cons_count.get("n_half_sib_matings"),
            "n_full_sib_matings": cons_count.get("n_full_sib_matings"),
            "total_missing_gp_links": cons_count.get("total_missing_gp_links"),
            "grandparent_reconciliation_passed": (cons.get("grandparent_reconciliation") or {}).get("passed"),
        },
        "offspring_distribution": validation_report.get("family_size_distribution", {}),
    }

    population = {
        "generation_sizes": obs(pop, "generation_sizes"),
        "n_generations": obs(pop, "generation_count"),
        "sex_ratio_male": (structural.get("sex_distribution") or {}).get("observed_ratio"),
    }

    return {
        "recorded_pedigree": {
            "population": population,
            "traits": traits,
            "family_structure": family_structure,
            "cross_trait": {
                "rA": obs(stat, "cross_trait_rA"),
                "rC": obs(stat, "cross_trait_rC"),
                "rE": obs(stat, "cross_trait_rE"),
            },
            "assortative_mating": {
                "mate_corr_liability1": obs(am, "mate_corr_liability1"),
                "mate_corr_liability2": obs(am, "mate_corr_liability2"),
            },
        }
    }


def extract_estimators(validation_report: dict[str, Any], stats_report: dict[str, Any]) -> dict[str, Any]:
    """Heritability estimators: relationship/outcome-derived, by scale."""
    her = validation_report.get("heritability", {})

    def obs(key: str) -> Any:
        return (her.get(key) or {}).get("observed")

    def reg(key: str, field: str) -> Any:
        return (her.get(key) or {}).get(field)

    liability_scale: dict[str, Any] = {}
    for t in (1, 2):
        liability_scale[f"trait{t}"] = {
            "mz_twin_A_corr": obs(f"mz_twin_A{t}_correlation"),
            "mz_twin_liability_corr": obs(f"mz_twin_liability{t}_correlation"),
            "dz_sibling_A_corr": obs(f"dz_sibling_A{t}_correlation"),
            "dz_sibling_liability_corr": obs(f"dz_sibling_liability{t}_correlation"),
            "falconer": obs(f"falconer_estimate_trait{t}"),
            "parent_offspring_A_slope": reg(f"parent_offspring_A{t}_regression", "slope"),
            "parent_offspring_A_r2": reg(f"parent_offspring_A{t}_regression", "r_squared"),
            "parent_offspring_liability_slope": reg(f"parent_offspring_liability{t}_regression", "slope"),
            "parent_offspring_liability_r2": reg(f"parent_offspring_liability{t}_regression", "r_squared"),
        }

    observed_scale = (stats_report.get("heritability") or {}).get("observed_h2_estimators", {})
    return {"heritability": {"observed_scale": observed_scale, "liability_scale": liability_scale}}


def _rebucket_analysis_sample(stats_report: dict[str, Any]) -> dict[str, Any]:
    """Collect the analysis-sample (post-ascertainment) stats into one block."""
    inc = stats_report.get("incidence", {})
    cen = stats_report.get("censoring", {})
    ped = stats_report.get("pedigree", {})
    cor = stats_report.get("correlations", {})
    sample: dict[str, Any] = {}
    sample.update({k: inc[k] for k in ANALYSIS_SAMPLE_INCIDENCE_KEYS if k in inc})
    if "person_years" in cen:
        sample["person_years"] = cen["person_years"]
    for src, dst in CENSORING_RENAME.items():
        if src in cen:
            sample[dst] = cen[src]
    sample.update({k: ped[k] for k in ANALYSIS_SAMPLE_PEDIGREE_KEYS if k in ped})
    sample.update({k: cor[k] for k in ANALYSIS_SAMPLE_CORRELATION_KEYS if k in cor})
    return sample


def _rebucket_analysis_pedigree(stats_report: dict[str, Any]) -> dict[str, Any]:
    """Collect the ancestor-closure-pedigree stats into one block."""
    ped = stats_report.get("pedigree", {})
    cor = stats_report.get("correlations", {})
    out: dict[str, Any] = {}
    full = ped.get("full") or {}
    if full.get("relationship_pair_counts") is not None:
        out["relationship_pair_counts"] = full["relationship_pair_counts"]
    if "parent_status" in ped:
        out["parent_status"] = ped["parent_status"]
    if "mate_correlation" in cor:
        out["mate_correlation"] = cor["mate_correlation"]
    return out


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
    validation_report: dict[str, Any],
    stats_report: dict[str, Any],
    prevalence_phenotyped: dict[str, Any],
    scope_counts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the v2 ``report.yaml`` dict and its scope-organized ``plot_payload``.

    Returns ``(report, plot_payload)``.
    """
    prevalence_sample = (stats_report.get("incidence") or {}).get("prevalence", {})
    ascertainment = build_ascertainment_summary(prevalence_phenotyped, prevalence_sample, scope_counts)

    # Split dense plot arrays out of every scope in one pass: the scalar half is
    # the scientific report, the dense half becomes the plot payload (keyed by
    # the same scopes, so the payload mirrors observed). This makes the
    # report's no-dense-arrays guarantee structural rather than per-scope.
    observed_full = {
        "ascertainment": ascertainment,
        "phenotyped_population": {"prevalence": prevalence_phenotyped},
        "analysis_sample": _rebucket_analysis_sample(stats_report),
        "analysis_pedigree": _rebucket_analysis_pedigree(stats_report),
    }
    observed, observed_dense = partition_dense(observed_full)

    report = {
        "schema": {"name": REPORT_SCHEMA_NAME, "version": REPORT_SCHEMA_VERSION},
        "replicate": replicate,
        "inputs": _build_inputs(params, case_ascertainment_ratio),
        "scopes": scope_counts,
        "quality_checks": normalize_quality_checks(validation_report),
        "truth": extract_truth(validation_report, params),
        "observed": observed,
        "estimators": extract_estimators(validation_report, stats_report),
    }
    assert_report_contract(report)

    plot_payload = {
        "schema": {"name": PLOT_PAYLOAD_SCHEMA_NAME, "version": PLOT_PAYLOAD_SCHEMA_VERSION},
        "replicate": {k: replicate.get(k) for k in ("folder", "scenario", "rep")},
        **observed_dense,
    }
    return report, plot_payload
