r"""Realized vA/vC/vE/h² trajectory across E schedules.

Example (``--expected-*`` are the first scenario's ``A1``/``C1`` and its
per-generation ``E1`` schedule in generation order)::

    python scripts/examples/compare_increasing_e_trajectory.py \
        --inputs results/examples/e_{flat,rise_mild,rise_steep,fall_steep}_std/rep{1,2,3} \
        --reps-per-scenario 3 3 3 3 \
        --expected-A 0.5 --expected-C 0.0 --expected-E 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare import group_reps
from simace.plotting.compare.trajectory import compare_realized_variance_trajectory

INCREASING_E_LABELS = [
    "E flat at 0.5",
    "E rising 0.5→0.6",
    "E rising 0.5→0.7",
    "E falling 0.5→0.3",
]


def main(argv: list[str] | None = None) -> None:
    """Parse flags, group the replicate directories per scenario, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="replicate directories, scenario-major")
    parser.add_argument("--output", default="docs/images/examples/increasing_e/realized_components_trajectory.png")
    parser.add_argument("--labels", nargs="+", default=INCREASING_E_LABELS)
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--expected-A", type=float, required=True)
    parser.add_argument("--expected-C", type=float, required=True)
    parser.add_argument("--expected-E", nargs="+", type=float, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/increasing_e_trajectory")

    scenarios = group_reps(args.labels, args.inputs, args.reps_per_scenario)
    compare_realized_variance_trajectory(
        scenarios,
        Path(args.output),
        trait=1,
        expected_A=args.expected_A,
        expected_C=args.expected_C,
        expected_E=args.expected_E,
    )


if __name__ == "__main__":
    main()
