r"""Per-gen A and liability distributions across E schedules.

Example::

    python scripts/examples/compare_increasing_e_components_by_gen.py \
        --inputs results/examples/e_{flat,rise_mild,rise_steep,fall_steep}_std/rep{1,2,3} \
        --reps-per-scenario 3 3 3 3
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare import group_reps
from simace.plotting.compare.increasing_e import compare_components_by_generation

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
    parser.add_argument("--output", default="docs/images/examples/increasing_e/components_by_gen.png")
    parser.add_argument("--labels", nargs="+", default=INCREASING_E_LABELS)
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--show-generations", nargs="+", type=int, default=[1, 5, 9])
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/increasing_e_components_by_gen")

    scenarios = group_reps(args.labels, args.inputs, args.reps_per_scenario)
    compare_components_by_generation(
        scenarios,
        Path(args.output),
        trait=1,
        show_generations=tuple(args.show_generations),
    )


if __name__ == "__main__":
    main()
