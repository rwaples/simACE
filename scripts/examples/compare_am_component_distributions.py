r"""Overlaid A / liability histograms across AM levels.

Example (``--min-generation`` is ``max(1, G_pheno // 2)`` of the first scenario)::

    python scripts/examples/compare_am_component_distributions.py \
        --inputs results/examples/{am_none,am_weak,am_strong}/rep{1,2,3}/pedigree.parquet \
        --reps-per-scenario 3 3 3 --min-generation 5
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_component_distributions


def main(argv: list[str] | None = None) -> None:
    """Parse flags, regroup the flat input list per scenario, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="pedigree.parquet files, scenario-major")
    parser.add_argument("--output", default="docs/images/examples/am/component_distributions.png")
    parser.add_argument("--labels", nargs="+", default=["no AM", "weak AM (0.2)", "strong AM (0.4)"])
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--min-generation", type=int, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/am_component_distributions")

    labels = args.labels
    reps_per_scenario = args.reps_per_scenario
    min_generation = args.min_generation

    inputs = list(args.inputs)
    scenario_paths: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        scenario_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps

    compare_component_distributions(
        scenario_paths=scenario_paths,
        labels=labels,
        output_path=Path(args.output),
        trait=1,
        min_generation=min_generation,
    )


if __name__ == "__main__":
    main()
