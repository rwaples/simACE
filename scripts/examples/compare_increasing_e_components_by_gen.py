r"""Per-gen A and liability distributions across E schedules.

Example::

    python scripts/examples/compare_increasing_e_components_by_gen.py \
        --inputs results/examples/e_{flat,rise_mild,rise_steep,fall_steep}_std/rep{1,2,3}/pedigree.parquet \
        --reps-per-scenario 3 3 3 3
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_components_by_generation

INCREASING_E_LABELS = [
    "E flat at 0.5",
    "E rising 0.5→0.6",
    "E rising 0.5→0.7",
    "E falling 0.5→0.3",
]


def main(argv: list[str] | None = None) -> None:
    """Parse flags, regroup the flat input list per scenario, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="pedigree.parquet files, scenario-major")
    parser.add_argument("--output", default="docs/images/examples/increasing_e/components_by_gen.png")
    parser.add_argument("--labels", nargs="+", default=INCREASING_E_LABELS)
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--show-generations", nargs="+", type=int, default=[1, 5, 9])
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/increasing_e_components_by_gen")

    labels = args.labels
    reps_per_scenario = args.reps_per_scenario
    show_generations = tuple(args.show_generations)

    inputs = list(args.inputs)
    pedigree_paths: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        pedigree_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps

    compare_components_by_generation(
        pedigree_paths_per_scenario=pedigree_paths,
        labels=labels,
        output_path=Path(args.output),
        trait=1,
        show_generations=show_generations,
    )


if __name__ == "__main__":
    main()
