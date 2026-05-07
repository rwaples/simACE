"""Snakemake wrapper: per-gen prevalence under standardize=global, none, and per_generation."""

from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_prevalence_drift


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag="examples/increasing_e_prevalence")

    labels = snakemake.params.labels
    reps_per_trajectory = snakemake.params.reps_per_trajectory

    # Inputs are ordered: for each trajectory, all _std reps, then all _nostd
    # reps, then all _pergen reps; then move to the next trajectory.
    inputs = list(snakemake.input)
    std_paths: list[list[Path]] = []
    nostd_paths: list[list[Path]] = []
    pergen_paths: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_trajectory:
        std_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps
        nostd_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps
        pergen_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps

    compare_prevalence_drift(
        std_paths_per_trajectory=std_paths,
        nostd_paths_per_trajectory=nostd_paths,
        pergen_paths_per_trajectory=pergen_paths,
        labels=labels,
        output_path=Path(snakemake.output[0]),
        trait=1,
        target_prevalence=snakemake.params.target_prevalence,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError as exc:
        raise SystemExit(
            "This script is intended to be invoked via Snakemake; for ad-hoc "
            "rendering call "
            "simace.plotting.compare_scenarios.compare_prevalence_drift() "
            "directly."
        ) from exc
    else:
        _run_snakemake()
