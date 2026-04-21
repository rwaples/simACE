"""Snakemake wrapper: realized-vA trajectory comparison across AM levels."""

from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag="examples/am_heritability")

    labels = snakemake.params.labels
    reps_per_scenario = snakemake.params.reps_per_scenario

    # Regroup the flat input list into per-scenario replicate lists.
    # Rule order matches zip(scenarios, reps_per_scenario).
    inputs = list(snakemake.input)
    scenario_paths: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        scenario_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps

    main(
        scenario_paths=scenario_paths,
        labels=labels,
        output_path=Path(snakemake.output[0]),
        trait=1,
        expected_A=snakemake.params.expected_A,
        expected_C=snakemake.params.expected_C,
        expected_E=snakemake.params.expected_E,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        from simace.plotting.compare_scenarios import cli

        cli()
    else:
        _run_snakemake()
