"""Smoke test for simace.analysis.stats.runner.main end-to-end."""

import sys

import pandas as pd
import pytest
import yaml

from simace.analysis.stats.runner import build_stats_report
from simace.analysis.stats.runner import cli as run_stats_cli
from simace.analysis.stats.runner import main as run_stats


@pytest.fixture(scope="module")
def tiny_phenotype():
    """Build a tiny censored phenotype DataFrame via simulate → phenotype → censor."""
    from simace.censoring.censor import run_censor
    from simace.phenotype import run_phenotype
    from simace.simulation.simulate import run_simulation

    sim_params = dict(
        seed=7,
        N=200,
        G_ped=2,
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
    pedigree = run_simulation(**sim_params)
    phenotype = run_phenotype(
        pedigree,
        G_pheno=2,
        seed=7,
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
        seed=7,
        gen_censoring={},
        death_scale=164,
        death_rho=2.73,
    )
    return pedigree, censored


@pytest.fixture
def runner_outputs(tmp_path, tiny_phenotype):
    pedigree, phenotype = tiny_phenotype
    ped_path = tmp_path / "pedigree.parquet"
    phe_path = tmp_path / "trait.parquet"
    pedigree.to_parquet(ped_path)
    phenotype.to_parquet(phe_path)

    stats_yaml = tmp_path / "stats_report.yaml"
    samples_pq = tmp_path / "plotting_sample.parquet"

    run_stats(
        phenotype_path=str(phe_path),
        censor_age=80.0,
        stats_output=str(stats_yaml),
        samples_output=str(samples_pq),
        seed=42,
        pedigree_path=str(ped_path),
        max_degree=2,
    )
    return stats_yaml, samples_pq


class TestRunnerMain:
    def test_writes_stats_yaml(self, runner_outputs):
        stats_yaml, _ = runner_outputs
        assert stats_yaml.exists()

    def test_writes_samples_parquet(self, runner_outputs):
        _, samples_pq = runner_outputs
        assert samples_pq.exists()
        df = pd.read_parquet(samples_pq)
        assert len(df) > 0

    def test_build_stats_report_returns_grouped_schema(self, tiny_phenotype):
        pedigree, phenotype = tiny_phenotype
        stats = build_stats_report(phenotype, 80.0, df_ped=pedigree, max_degree=2)
        assert set(stats) == {"metadata", "incidence", "censoring", "pedigree", "correlations", "heritability"}
        flat_keys = {
            "prevalence",
            "pair_counts",
            "pair_counts_ped",
            "mate_correlation",
            "observed_h2_estimators",
        }
        assert flat_keys.isdisjoint(stats)

    def test_stats_yaml_has_expected_grouped_keys(self, runner_outputs):
        stats_yaml, _ = runner_outputs
        with open(stats_yaml, encoding="utf-8") as fh:
            stats = yaml.safe_load(fh)
        assert set(stats) == {"metadata", "incidence", "censoring", "pedigree", "correlations", "heritability"}
        assert "n_individuals" in stats["metadata"]
        assert "prevalence" in stats["incidence"]
        assert "person_years" in stats["censoring"]
        assert "relationship_pair_counts" in stats["pedigree"]
        assert "affected_correlations" in stats["correlations"]
        assert "tetrachoric" in stats["correlations"]
        assert "observed_h2_estimators" in stats["heritability"]

    def test_pedigree_keys_when_pedigree_provided(self, runner_outputs):
        stats_yaml, _ = runner_outputs
        with open(stats_yaml, encoding="utf-8") as fh:
            stats = yaml.safe_load(fh)
        # Pedigree branch — present only because pedigree_path is supplied.
        assert "full" in stats["pedigree"]
        assert "relationship_pair_counts" in stats["pedigree"]["full"]
        assert "n_individuals" in stats["pedigree"]["full"]
        assert "mate_correlation" in stats["correlations"]

    def test_runs_without_pedigree(self, tmp_path, tiny_phenotype):
        _, phenotype = tiny_phenotype
        phe_path = tmp_path / "trait.parquet"
        phenotype.to_parquet(phe_path)
        stats_yaml = tmp_path / "stats.yaml"
        samples_pq = tmp_path / "samples.parquet"
        run_stats(
            phenotype_path=str(phe_path),
            censor_age=80.0,
            stats_output=str(stats_yaml),
            samples_output=str(samples_pq),
            seed=42,
            pedigree_path=None,
            max_degree=2,
        )
        with open(stats_yaml, encoding="utf-8") as fh:
            stats = yaml.safe_load(fh)
        # The pedigree-dependent branches are skipped.
        assert "full" not in stats["pedigree"]
        assert "in_pedigree" not in stats["pedigree"]["parent_status"]
        assert "mate_correlation" not in stats["correlations"]

    def test_case_ascertainment_ratio_recorded(self, tmp_path, tiny_phenotype):
        _, phenotype = tiny_phenotype
        phe_path = tmp_path / "trait.parquet"
        phenotype.to_parquet(phe_path)
        stats_yaml = tmp_path / "stats.yaml"
        samples_pq = tmp_path / "samples.parquet"
        run_stats(
            phenotype_path=str(phe_path),
            censor_age=80.0,
            stats_output=str(stats_yaml),
            samples_output=str(samples_pq),
            seed=42,
            pedigree_path=None,
            case_ascertainment_ratio=0.7,
        )
        with open(stats_yaml, encoding="utf-8") as fh:
            stats = yaml.safe_load(fh)
        assert stats["metadata"]["case_ascertainment_ratio"] == pytest.approx(0.7)

    def test_gen_censoring_branch(self, tmp_path, tiny_phenotype):
        _, phenotype = tiny_phenotype
        phe_path = tmp_path / "trait.parquet"
        phenotype.to_parquet(phe_path)
        stats_yaml = tmp_path / "stats.yaml"
        samples_pq = tmp_path / "samples.parquet"
        run_stats(
            phenotype_path=str(phe_path),
            censor_age=80.0,
            stats_output=str(stats_yaml),
            samples_output=str(samples_pq),
            seed=42,
            gen_censoring={0: [80, 80], 1: [0, 80]},
            pedigree_path=None,
        )
        with open(stats_yaml, encoding="utf-8") as fh:
            stats = yaml.safe_load(fh)
        # gen_censoring populates these branch keys
        assert "windows" in stats["censoring"]
        assert "confusion" in stats["censoring"]
        assert "cascade" in stats["censoring"]


class TestRunnerCli:
    def test_cli_invokes_main(self, tmp_path, tiny_phenotype, monkeypatch):
        pedigree, phenotype = tiny_phenotype
        ped_path = tmp_path / "pedigree.parquet"
        phe_path = tmp_path / "trait.parquet"
        pedigree.to_parquet(ped_path)
        phenotype.to_parquet(phe_path)
        stats_yaml = tmp_path / "stats.yaml"
        samples_pq = tmp_path / "samples.parquet"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "stats",
                str(phe_path),
                "80",
                str(stats_yaml),
                str(samples_pq),
                "--pedigree",
                str(ped_path),
                "--max-degree",
                "2",
            ],
        )
        run_stats_cli()
        assert stats_yaml.exists()
        assert samples_pq.exists()

    def test_cli_with_gen_censoring(self, tmp_path, tiny_phenotype, monkeypatch):
        pedigree, phenotype = tiny_phenotype
        ped_path = tmp_path / "pedigree.parquet"
        phe_path = tmp_path / "trait.parquet"
        pedigree.to_parquet(ped_path)
        phenotype.to_parquet(phe_path)
        stats_yaml = tmp_path / "stats.yaml"
        samples_pq = tmp_path / "samples.parquet"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "stats",
                str(phe_path),
                "80",
                str(stats_yaml),
                str(samples_pq),
                "--pedigree",
                str(ped_path),
                "--gen-censoring",
                '{"0": [80, 80], "1": [0, 80]}',
            ],
        )
        run_stats_cli()
        with open(stats_yaml, encoding="utf-8") as fh:
            stats = yaml.safe_load(fh)
        assert "censoring" in stats
