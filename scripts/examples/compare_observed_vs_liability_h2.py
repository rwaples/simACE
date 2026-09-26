r"""Observed-scale vs liability-scale h² by phenotype model.

Example (``--input-h2`` is ``A1 / (A1 + C1 + E1)`` of the first scenario)::

    python scripts/examples/compare_observed_vs_liability_h2.py \
        --pedigree results/examples/{model_ltm,model_cure_frailty_ln,model_frailty_wb}/rep{1,2,3}/pedigree.parquet \
        --report results/examples/{model_ltm,model_cure_frailty_ln,model_frailty_wb}/rep{1,2,3}/report.yaml \
        --reps-per-scenario 3 3 3 --input-h2 0.5
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare_scenarios import compare_observed_vs_liability_h2


def _regroup(flat_inputs: list[str], reps_per_scenario: list[int]) -> list[list[Path]]:
    grouped: list[list[Path]] = []
    offset = 0
    for n_reps in reps_per_scenario:
        grouped.append([Path(p) for p in flat_inputs[offset : offset + n_reps]])
        offset += n_reps
    return grouped


def main(argv: list[str] | None = None) -> None:
    """Parse flags, regroup the flat input lists per scenario, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pedigree", nargs="+", required=True, help="pedigree.parquet files, scenario-major")
    parser.add_argument("--report", nargs="+", required=True, help="report.yaml files, scenario-major")
    parser.add_argument("--output", default="docs/images/examples/models/observed_vs_liability.png")
    parser.add_argument("--labels", nargs="+", default=["LTM", "Cure-frailty (lognormal)", "Frailty (Weibull)"])
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--input-h2", type=float, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/observed_vs_liability_h2")

    labels = args.labels
    reps_per_scenario = args.reps_per_scenario
    input_h2 = args.input_h2

    pedigree_paths = _regroup(list(args.pedigree), reps_per_scenario)
    report_paths = _regroup(list(args.report), reps_per_scenario)

    compare_observed_vs_liability_h2(
        pedigree_paths_per_scenario=pedigree_paths,
        report_paths_per_scenario=report_paths,
        labels=labels,
        output_path=Path(args.output),
        trait=1,
        input_h2=input_h2,
    )


if __name__ == "__main__":
    main()
