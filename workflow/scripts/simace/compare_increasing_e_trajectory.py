"""Snakemake wrapper: realized vA/vC/vE/h² trajectory across E schedules."""

from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_realized_variance_trajectory


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag="examples/increasing_e_trajectory")

    labels = snakemake.params.labels
    reps_per_scenario = snakemake.params.reps_per_scenario

    inputs = list(snakemake.input)
    scenario_paths: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        scenario_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps

    compare_realized_variance_trajectory(
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
    except NameError as exc:
        raise SystemExit(
            "This script is intended to be invoked via Snakemake; for ad-hoc "
            "rendering call "
            "simace.plotting.compare_scenarios.compare_realized_variance_trajectory() "
            "directly."
        ) from exc
    else:
        _run_snakemake()
