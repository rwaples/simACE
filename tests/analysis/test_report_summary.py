"""Contract tests for REPORT_SUMMARY_REGISTRY ↔ the curated v2 report.

The registry in `simace.analysis.report_schema` is the single source of truth
for the folder-level ``report_summary.tsv`` columns. These tests run a small,
parameter-tuned coverage scenario through the full Analyze stage, then assert
that **every** registered path resolves to a non-None value. If a producer-side
rename in `validate.py` / `report.py` empties a registry-tracked column, this
suite fails — the silent-drop bug class this refactor closes.

A full report (not just validation) is built so the scope-count and
ascertainment columns resolve alongside the truth/estimators columns.
"""

from __future__ import annotations

import pytest

from simace.analysis.gather import _get_nested, extract_metrics
from simace.analysis.report_schema import REPORT_SUMMARY_REGISTRY
from simace.core.yaml_io import dump_yaml, load_yaml

# Coverage parameters mirror config/test.yaml::coverage_scenario. Tuned so every
# registered metric is populated: half-sib pairs (mating_lambda=0.5, N=2000), MZ
# twins (p_mztwin=0.05), and non-zero mate liability correlation (assort=0.3).
_COVERAGE_PARAMS = dict(
    seed=1234,
    N=2000,
    G_ped=4,
    G_sim=4,
    mating_lambda=0.5,
    p_mztwin=0.05,
    A1=0.5,
    C1=0.2,
    E1=0.3,
    A2=0.4,
    C2=0.2,
    E2=0.4,
    rA=0.0,
    rC=0.0,
    rE=0.0,
    assort1=0.3,
    assort2=0.3,
)


@pytest.fixture(scope="session")
def coverage_report(tmp_path_factory) -> dict:
    """Run the full Analyze stage on a coverage scenario; return its report.yaml.

    No ascertainment is applied (trait.full == trait, pedigree == pedigree.full),
    so the ascertainment columns are identity but still populated. Round-trips
    through `dump_yaml` / `load_yaml` so the test sees what `gather` reads.
    """
    from simace.analysis.analyze import run_analysis
    from simace.censoring.censor import run_censor
    from simace.phenotype import run_phenotype
    from simace.simulation.simulate import run_simulation

    work = tmp_path_factory.mktemp("coverage_scenario")
    pedigree = run_simulation(**_COVERAGE_PARAMS)
    phenotype = run_phenotype(
        pedigree,
        G_pheno=_COVERAGE_PARAMS["G_ped"],
        seed=_COVERAGE_PARAMS["seed"],
        standardize=True,
        phenotype_model1="frailty",
        phenotype_params1={"distribution": "weibull", "scale": 2160, "rho": 0.8},
        beta1=1.0,
        beta_sex1=0.0,
        phenotype_model2="frailty",
        phenotype_params2={"distribution": "weibull", "scale": 333, "rho": 1.2},
        beta2=1.0,
        beta_sex2=0.0,
    )
    censored = run_censor(
        phenotype,
        pedigree,
        censor_age=80,
        seed=_COVERAGE_PARAMS["seed"],
        gen_censoring={},
        death_scale=164,
        death_rho=2.73,
    )

    ped_full = work / "pedigree.full.parquet"
    ped = work / "pedigree.parquet"
    trait_full = work / "trait.full.parquet"
    trait = work / "trait.parquet"
    params_path = work / "params.yaml"
    pedigree.to_parquet(ped_full)
    pedigree.to_parquet(ped)
    censored.to_parquet(trait_full)
    censored.to_parquet(trait)
    dump_yaml(_COVERAGE_PARAMS, params_path)

    report_yaml = work / "report.yaml"
    run_analysis(
        pedigree_full_path=str(ped_full),
        params_path=str(params_path),
        trait_full_path=str(trait_full),
        trait_path=str(trait),
        pedigree_path=str(ped),
        report_output=str(report_yaml),
        plot_payload_output=str(work / "plot_payload.yaml"),
        samples_output=str(work / "plotting_sample.parquet"),
        folder="test",
        scenario="coverage",
        rep=1,
        seed=_COVERAGE_PARAMS["seed"],
        censor_age=80.0,
        max_degree=2,
    )
    return load_yaml(report_yaml)


def test_unique_columns():
    columns = [spec.column for spec in REPORT_SUMMARY_REGISTRY]
    duplicates = {c for c in columns if columns.count(c) > 1}
    assert not duplicates, f"Duplicate columns in REPORT_SUMMARY_REGISTRY: {sorted(duplicates)}"


def test_every_registry_path_resolves(coverage_report):
    """Each registered path must hit a non-None leaf in the coverage report."""
    missing = []
    for spec in REPORT_SUMMARY_REGISTRY:
        value = _get_nested(coverage_report, *spec.path)
        if value is None:
            missing.append((spec.column, "/".join(spec.path)))
    assert not missing, (
        "Registry paths that did not resolve in the coverage report "
        "(producer must emit these keys, or registry must be updated): " + repr(missing)
    )


def test_extract_metrics_populates_registry_columns(tmp_path, coverage_report):
    """End-to-end: extract_metrics returns a non-None value for every registered column."""
    val_dir = tmp_path / "results" / "test" / "coverage_scenario" / "rep1"
    val_dir.mkdir(parents=True)
    report_path = val_dir / "report.yaml"
    dump_yaml(coverage_report, report_path)

    row = extract_metrics(str(report_path))
    missing = [spec.column for spec in REPORT_SUMMARY_REGISTRY if row.get(spec.column) is None]
    assert not missing, f"extract_metrics returned None for registry columns: {missing}"
