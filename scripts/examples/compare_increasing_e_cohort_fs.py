r"""Per-cohort full-sib liability correlation across E schedules.

Example (``--expected-*`` are the first scenario's ``A1``/``C1``)::

    python scripts/examples/compare_increasing_e_cohort_fs.py \
        --inputs results/examples/e_{flat,rise_mild,rise_steep,fall_steep}_std/rep{1,2,3}/pedigree.parquet \
        --reps-per-scenario 3 3 3 3 --expected-A 0.5 --expected-C 0.0
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_cohort_fs_correlations

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
    parser.add_argument("--output", default="docs/images/examples/increasing_e/fs_corr_by_gen.png")
    parser.add_argument("--labels", nargs="+", default=INCREASING_E_LABELS)
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--expected-A", type=float, required=True)
    parser.add_argument("--expected-C", type=float, required=True)
    parser.add_argument("--min-generation", type=int, default=1)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/increasing_e_cohort_fs")

    labels = args.labels
    reps_per_scenario = args.reps_per_scenario

    inputs = list(args.inputs)
    pedigree_paths: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        pedigree_paths.append([Path(p) for p in inputs[offset : offset + n_reps]])
        offset += n_reps

    compare_cohort_fs_correlations(
        pedigree_paths_per_scenario=pedigree_paths,
        labels=labels,
        output_path=Path(args.output),
        trait=1,
        expected_A=args.expected_A,
        expected_C=args.expected_C,
        min_generation=args.min_generation,
    )


if __name__ == "__main__":
    main()
