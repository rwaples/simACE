r"""Realized-vA trajectory comparison across AM levels.

Example (``--expected-*`` are the first scenario's ``A1``/``C1``/``E1``)::

    python scripts/examples/compare_am_heritability.py \
        --inputs results/examples/{am_none,am_weak,am_strong}/rep{1,2,3} \
        --reps-per-scenario 3 3 3 \
        --expected-A 0.5 --expected-C 0.0 --expected-E 0.5
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare import group_reps
from simace.plotting.compare.trajectory import compare_realized_variance_trajectory


def main(argv: list[str] | None = None) -> None:
    """Parse flags, group the replicate directories per scenario, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="replicate directories, scenario-major")
    parser.add_argument("--output", default="docs/images/examples/am/vA_trajectory.png")
    parser.add_argument("--labels", nargs="+", default=["no AM", "weak AM (0.2)", "strong AM (0.4)"])
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--expected-A", type=float, required=True)
    parser.add_argument("--expected-C", type=float, required=True)
    parser.add_argument("--expected-E", type=float, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/am_heritability")

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
