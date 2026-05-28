"""End-to-end smoke test for the combined Analyze stage (Validate + Stats)."""

import pytest
import yaml

from simace.analysis.analyze import run_analysis

# ---------------------------------------------------------------------------
# Module-scoped fixture: simulate -> phenotype -> censor, once per file.
#
# Validate runs on the full simulated pedigree; Stats runs on the censored
# trait + pedigree subsample. Sizing matches test_validate (N=1000, G_ped=3,
# seed=42) so the validation checks pass deterministically.
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
    trait = tmp_path / "trait.parquet"
    params_path = tmp_path / "params.yaml"
    pedigree.to_parquet(ped_full)
    pedigree.to_parquet(ped)
    censored.to_parquet(trait)
    with open(params_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(params, fh)

    report_yaml = tmp_path / "report.yaml"
    samples_pq = tmp_path / "plotting_sample.parquet"

    report = run_analysis(
        pedigree_full_path=str(ped_full),
        params_path=str(params_path),
        trait_path=str(trait),
        pedigree_path=str(ped),
        report_output=str(report_yaml),
        samples_output=str(samples_pq),
        seed=42,
        censor_age=80.0,
        max_degree=2,
    )
    return {
        "report": report,
        "report_yaml": report_yaml,
        "samples_pq": samples_pq,
    }


_REPORT_GROUPS = {
    "metadata",
    "incidence",
    "censoring",
    "pedigree",
    "correlations",
    "heritability",
    "validation",
}


class TestRunAnalysis:
    def test_outputs_written(self, analyze_outputs):
        assert analyze_outputs["report_yaml"].exists()
        assert analyze_outputs["samples_pq"].exists()

    def test_validation_summary_passes(self, analyze_outputs):
        summary = analyze_outputs["report"]["validation"]["summary"]
        assert summary["checks_total"] == summary["checks_passed"] + summary["checks_failed"]
        assert summary["passed"] is True

    def test_report_has_six_stats_groups_plus_validation(self, analyze_outputs):
        assert set(analyze_outputs["report"]) == _REPORT_GROUPS

    def test_written_yaml_matches_returned_report(self, analyze_outputs):
        with open(analyze_outputs["report_yaml"], encoding="utf-8") as fh:
            report = yaml.safe_load(fh)
        assert set(report) == _REPORT_GROUPS
        # Validation folded in as its own group.
        assert "summary" in report["validation"]
        assert "structural" in report["validation"]
        # Stats ran on the (sub)sampled pedigree, so the full-pedigree branch is present.
        assert "full" in report["pedigree"]
        assert "mate_correlation" in report["correlations"]
