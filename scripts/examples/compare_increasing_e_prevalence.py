r"""Per-gen prevalence under standardize=global, none, and per_generation.

Inputs are trajectory-major: for each trajectory, all ``_std`` reps, then all
``_nostd`` reps, then all ``_pergen`` reps. Example::

    python scripts/examples/compare_increasing_e_prevalence.py \
        --inputs results/examples/e_{flat,rise_mild,rise_steep,fall_steep}_{std,nostd,pergen}/rep{1,2,3} \
        --reps-per-trajectory 3 3 3 3
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare import PrevalenceSeries, group_reps
from simace.plotting.compare.increasing_e import compare_prevalence_drift

INCREASING_E_LABELS = [
    "E flat at 0.5",
    "E rising 0.5→0.6",
    "E rising 0.5→0.7",
    "E falling 0.5→0.3",
]
#: The standardize modes, in the order each trajectory's inputs list them.
STANDARDIZE_LABELS = ["standardize=global", "standardize=none", "standardize=per_generation"]


def main(argv: list[str] | None = None) -> None:
    """Parse flags, group the replicate directories per trajectory and standardization, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="replicate directories, trajectory-major")
    parser.add_argument("--output", default="docs/images/examples/increasing_e/prevalence_drift.png")
    parser.add_argument("--labels", nargs="+", default=INCREASING_E_LABELS)
    parser.add_argument("--target-prevalence", type=float, default=0.1)
    parser.add_argument("--reps-per-trajectory", nargs="+", type=int, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/increasing_e_prevalence")

    n_modes = len(STANDARDIZE_LABELS)
    groups = group_reps(
        [label for label in args.labels for _ in range(n_modes)],
        args.inputs,
        [n for n in args.reps_per_trajectory for _ in range(n_modes)],
    )
    series = [PrevalenceSeries(mode, tuple(groups[k::n_modes])) for k, mode in enumerate(STANDARDIZE_LABELS)]
    compare_prevalence_drift(
        series,
        Path(args.output),
        trait=1,
        target_prevalence=args.target_prevalence,
    )


if __name__ == "__main__":
    main()
