"""Tests for Snakemake workflow helper paths."""

from workflow.common import (
    get_scenario_sim_outputs,
    get_scenario_simulation_outputs,
)


def test_simulate_only_outputs_use_pre_ascertainment_pedigrees():
    config = {
        "defaults": {"folder": "test", "replicates": 2},
        "scenarios": {"small": {}},
    }

    assert get_scenario_simulation_outputs(config, "small") == [
        "results/test/small/rep1/pedigree.full.parquet",
        "results/test/small/rep2/pedigree.full.parquet",
    ]


def test_full_scenario_outputs_keep_canonical_ascertained_pedigrees():
    config = {
        "defaults": {"folder": "test", "replicates": 1},
        "scenarios": {"small": {}},
    }

    outputs = get_scenario_sim_outputs(config, "small")

    assert "results/test/small/rep1/pedigree.parquet" in outputs
    assert "results/test/small/rep1/pedigree.full.parquet" not in outputs
    assert "results/test/small/rep1/report.yaml" in outputs
    assert "results/test/small/rep1/plot_payload.yaml" in outputs
    assert "results/test/small/rep1/trait.full.parquet" in outputs
    assert "results/test/small/rep1/stats_report.yaml" not in outputs
    assert "results/test/small/rep1/validation.yaml" not in outputs
    assert "results/test/small/rep1/phenotype_stats.yaml" not in outputs
