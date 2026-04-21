"""Snakemake wrapper: naive h² estimators across AM levels."""

from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_naive_estimators


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag="examples/am_naive_estimators")

    labels = snakemake.params.labels
    reps_per_scenario = snakemake.params.reps_per_scenario
    input_h2 = snakemake.params.input_h2
    min_generation = snakemake.params.min_generation

    inputs = list(snakemake.input)
    pedigree_paths: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        pedigree_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps

    compare_naive_estimators(
        pedigree_paths_per_scenario=pedigree_paths,
        labels=labels,
        output_path=Path(snakemake.output[0]),
        trait=1,
        input_h2=input_h2,
        min_generation=min_generation,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError as exc:
        raise SystemExit(
            "This script is intended to be invoked via Snakemake; for ad-hoc "
            "rendering call simace.plotting.compare_scenarios.compare_naive_estimators() "
            "directly."
        ) from exc
    else:
        _run_snakemake()
