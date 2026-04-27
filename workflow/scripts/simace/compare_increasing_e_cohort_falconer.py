"""Snakemake wrapper: per-cohort vs pooled-across-gens Falconer across E schedules."""

from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_cohort_falconer


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag="examples/increasing_e_cohort_falconer")

    labels = snakemake.params.labels
    reps_per_scenario = snakemake.params.reps_per_scenario

    inputs = list(snakemake.input)
    pedigree_paths: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        pedigree_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps

    compare_cohort_falconer(
        pedigree_paths_per_scenario=pedigree_paths,
        labels=labels,
        output_path=Path(snakemake.output[0]),
        trait=1,
        min_generation=snakemake.params.min_generation,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError as exc:
        raise SystemExit(
            "This script is intended to be invoked via Snakemake; for ad-hoc "
            "rendering call "
            "simace.plotting.compare_scenarios.compare_cohort_falconer() "
            "directly."
        ) from exc
    else:
        _run_snakemake()
