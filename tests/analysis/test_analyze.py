"""End-to-end smoke test for the combined Analyze stage (curated v2 report)."""

import pytest
import yaml

from simace.analysis.analyze import run_analysis
from simace.analysis.report_schema import (
    REPORT_SCHEMA_VERSION,
    REPORT_TOP_LEVEL_GROUPS,
    assert_report_contract,
    find_dense_keys,
)
from simace.plotting.stats_report import plotting_report_view

# ---------------------------------------------------------------------------
# Module-scoped fixture: simulate -> phenotype -> censor, once per file.
#
# Validate runs on the full simulated pedigree; the phenotyped-population and
# analysis-sample phases run on the censored trait (no ascertainment in this
# fixture, so trait.full == trait). Sizing matches test_validate (N=1000,
# G_ped=3, seed=42) so the validation checks pass deterministically.
# ---------------------------------------------------------------------------

_SIM_PARAMS = dict(
    seed=42,
    N=1000,
    G_ped=3,
    G_sim=3,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.2,
    E1=0.3,
    A2=0.5,
    C2=0.2,
    E2=0.3,
    rA=0.3,
    rC=0.5,
    assort1=0.0,
    assort2=0.0,
)


@pytest.fixture(scope="module")
def analyze_data():
    from simace.censoring.censor import run_censor
    from simace.phenotype import run_phenotype
    from simace.simulation.simulate import run_simulation

    pedigree = run_simulation(**_SIM_PARAMS)
    phenotype = run_phenotype(
        pedigree,
        G_pheno=3,
        seed=42,
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
        censor_age=80,
        seed=42,
        gen_censoring={},
        death_scale=164,
        death_rho=2.73,
    )
    params = {**_SIM_PARAMS, "rE": 0.0}
    return pedigree, censored, params


@pytest.fixture
def analyze_outputs(tmp_path, analyze_data):
    pedigree, censored, params = analyze_data
    ped_full = tmp_path / "pedigree.full.parquet"
    ped = tmp_path / "pedigree.parquet"
    trait_full = tmp_path / "trait.full.parquet"
    trait = tmp_path / "trait.parquet"
    params_path = tmp_path / "params.yaml"
    pedigree.to_parquet(ped_full)
    pedigree.to_parquet(ped)
    censored.to_parquet(trait_full)
    censored.to_parquet(trait)
    with open(params_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(params, fh)

    report_yaml = tmp_path / "report.yaml"
    plot_payload_yaml = tmp_path / "plot_payload.yaml"
    samples_pq = tmp_path / "plotting_sample.parquet"

    report = run_analysis(
        pedigree_full_path=str(ped_full),
        params_path=str(params_path),
        trait_full_path=str(trait_full),
        trait_path=str(trait),
        pedigree_path=str(ped),
        report_output=str(report_yaml),
        plot_payload_output=str(plot_payload_yaml),
        samples_output=str(samples_pq),
        folder="test",
        scenario="analyze_unit",
        rep=1,
        seed=42,
        censor_age=80.0,
        max_degree=2,
    )
    return {
        "report": report,
        "report_yaml": report_yaml,
        "plot_payload_yaml": plot_payload_yaml,
        "samples_pq": samples_pq,
    }


class TestRunAnalysis:
    def test_outputs_written(self, analyze_outputs):
        assert analyze_outputs["report_yaml"].exists()
        assert analyze_outputs["plot_payload_yaml"].exists()
        assert analyze_outputs["samples_pq"].exists()

    def test_schema_is_v2(self, analyze_outputs):
        assert analyze_outputs["report"]["schema"] == {
            "name": "simace_report",
            "version": REPORT_SCHEMA_VERSION,
        }

    def test_top_level_groups_match_contract(self, analyze_outputs):
        assert set(analyze_outputs["report"]) == set(REPORT_TOP_LEVEL_GROUPS)

    def test_report_passes_contract(self, analyze_outputs):
        # Validates schema, required groups, and absence of dense plot arrays.
        assert_report_contract(analyze_outputs["report"])

    def test_quality_checks_normalized(self, analyze_outputs):
        qc = analyze_outputs["report"]["quality_checks"]
        assert qc["summary"]["passed"] is True
        assert qc["summary"]["n_failed"] == 0
        assert qc["checks"], "expected at least one normalized check row"
        row = qc["checks"][0]
        assert set(row) >= {"id", "scope", "severity", "status", "observed", "expected", "tolerance", "message"}
        assert all(c["scope"] == "recorded_pedigree" for c in qc["checks"])

    def test_truth_realized_variances(self, analyze_outputs):
        realized = analyze_outputs["report"]["truth"]["recorded_pedigree"]["traits"]["trait1"]["realized"]
        assert realized["var_A"] is not None
        assert realized["h2_liability"] is not None
        assert "realized_by_generation" in analyze_outputs["report"]["truth"]["recorded_pedigree"]["traits"]["trait1"]

    def test_observed_ascertainment_before_after(self, analyze_outputs):
        asc = analyze_outputs["report"]["observed"]["ascertainment"]
        enr = asc["trait_enrichment"]["trait1"]
        assert "affected_fraction_before" in enr
        assert "affected_fraction_after" in enr
        assert asc["retained_fraction"] is not None

    def test_estimators_present(self, analyze_outputs):
        her = analyze_outputs["report"]["estimators"]["heritability"]
        assert "observed_scale" in her
        assert her["liability_scale"]["trait1"]["falconer"] is not None

    def test_scopes_cover_four_populations(self, analyze_outputs):
        scopes = analyze_outputs["report"]["scopes"]
        assert set(scopes) == {
            "recorded_pedigree",
            "phenotyped_population",
            "analysis_sample",
            "analysis_pedigree",
        }
        assert scopes["analysis_pedigree"]["ancestor_closure_ratio"] is not None

    def test_written_yaml_matches_returned_report(self, analyze_outputs):
        with open(analyze_outputs["report_yaml"], encoding="utf-8") as fh:
            report = yaml.safe_load(fh)
        assert set(report) == set(REPORT_TOP_LEVEL_GROUPS)
        assert list(find_dense_keys(report)) == []

    def test_plot_payload_has_dense_arrays(self, analyze_outputs):
        with open(analyze_outputs["plot_payload_yaml"], encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
        # gen_censoring is None for this fixture, so only the incidence curves
        # are dense; they live under analysis_sample, carrying ages.
        assert "analysis_sample" in payload
        dense = set(find_dense_keys(payload))
        assert any(p.endswith(".ages") for p in dense)
        assert any(p.endswith(".observed_values") for p in dense)

    def test_adapter_round_trip_reunites_scalars_and_arrays(self, analyze_outputs):
        report = analyze_outputs["report"]
        with open(analyze_outputs["plot_payload_yaml"], encoding="utf-8") as fh:
            payload = yaml.safe_load(fh)
        view = plotting_report_view(report, payload)
        ci = view["cumulative_incidence"]["trait1"]
        # Array (from payload) and scalar landmark (from report) sit together.
        assert "ages" in ci
        assert "half_target_age" in ci
        assert len(ci["ages"]) == len(ci["observed_values"])
        # Validation-derived per-generation table is reconstructed for plots.
        assert view["per_generation"]
        assert view["parameters"].get("A1") is not None
