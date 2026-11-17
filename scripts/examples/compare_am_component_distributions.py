r"""Overlaid A / liability histograms across AM levels.

Example (``--min-generation`` is ``max(1, G_pheno // 2)`` of the first scenario)::

    python scripts/examples/compare_am_component_distributions.py \
        --inputs results/examples/{am_none,am_weak,am_strong}/rep{1,2,3} \
        --reps-per-scenario 3 3 3 --min-generation 5
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare import group_reps
from simace.plotting.compare.am import compare_component_distributions


def main(argv: list[str] | None = None) -> None:
    """Parse flags, group the replicate directories per scenario, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="replicate directories, scenario-major")
    parser.add_argument("--output", default="docs/images/examples/am/component_distributions.png")
    parser.add_argument("--labels", nargs="+", default=["no AM", "weak AM (0.2)", "strong AM (0.4)"])
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--min-generation", type=int, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/am_component_distributions")

    scenarios = group_reps(args.labels, args.inputs, args.reps_per_scenario)
    compare_component_distributions(
        scenarios,
        Path(args.output),
        trait=1,
        min_generation=args.min_generation,
    )


if __name__ == "__main__":
    main()
