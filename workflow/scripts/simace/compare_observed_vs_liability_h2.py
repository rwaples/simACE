"""Snakemake wrapper: observed-scale vs liability-scale h² by phenotype model."""

from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_observed_vs_liability_h2


def _regroup(flat_inputs: list[str], reps_per_scenario: list[int]) -> list[list[Path]]:
    grouped: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        grouped.append([Path(p) for p in flat_inputs[offset : offset + n_reps]])
        offset += n_reps
    return grouped


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag="examples/observed_vs_liability_h2")

    labels = snakemake.params.labels
    reps_per_scenario = snakemake.params.reps_per_scenario
    input_h2 = snakemake.params.input_h2

    pedigree_paths = _regroup(list(snakemake.input.pedigree), reps_per_scenario)
    phenotype_stats_paths = _regroup(list(snakemake.input.phenotype_stats), reps_per_scenario)

    compare_observed_vs_liability_h2(
        pedigree_paths_per_scenario=pedigree_paths,
        phenotype_stats_paths_per_scenario=phenotype_stats_paths,
        labels=labels,
        output_path=Path(snakemake.output[0]),
        trait=1,
        input_h2=input_h2,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError as exc:
        raise SystemExit(
            "This script is intended to be invoked via Snakemake; for ad-hoc "
            "rendering call simace.plotting.compare_scenarios.compare_observed_vs_liability_h2() "
            "directly."
        ) from exc
    else:
        _run_snakemake()
