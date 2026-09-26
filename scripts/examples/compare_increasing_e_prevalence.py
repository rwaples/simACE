r"""Per-gen prevalence under standardize=global, none, and per_generation.

Inputs are trajectory-major: for each trajectory, all ``_std`` reps, then all
``_nostd`` reps, then all ``_pergen`` reps. Example::

    python scripts/examples/compare_increasing_e_prevalence.py \
        --inputs results/examples/e_{flat,rise_mild,rise_steep,fall_steep}_{std,nostd,pergen}/rep{1,2,3}/report.yaml \
        --reps-per-trajectory 3 3 3 3
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_prevalence_drift

INCREASING_E_LABELS = [
    "E flat at 0.5",
    "E rising 0.5→0.6",
    "E rising 0.5→0.7",
    "E falling 0.5→0.3",
]


def main(argv: list[str] | None = None) -> None:
    """Parse flags, regroup the flat input list per trajectory and standardization, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="report.yaml files, trajectory-major")
    parser.add_argument("--output", default="docs/images/examples/increasing_e/prevalence_drift.png")
    parser.add_argument("--labels", nargs="+", default=INCREASING_E_LABELS)
    parser.add_argument("--target-prevalence", type=float, default=0.1)
    parser.add_argument("--reps-per-trajectory", nargs="+", type=int, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/increasing_e_prevalence")

    labels = args.labels
    reps_per_trajectory = args.reps_per_trajectory

    inputs = list(args.inputs)
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
        output_path=Path(args.output),
        trait=1,
        target_prevalence=args.target_prevalence,
    )


if __name__ == "__main__":
    main()
