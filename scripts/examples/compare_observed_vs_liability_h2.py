r"""Observed-scale vs liability-scale h² by phenotype model.

Example (``--input-h2`` is ``A1 / (A1 + C1 + E1)`` of the first scenario)::

    python scripts/examples/compare_observed_vs_liability_h2.py \
        --inputs results/examples/{model_ltm,model_cure_frailty_ln,model_frailty_wb}/rep{1,2,3} \
        --reps-per-scenario 3 3 3 --input-h2 0.5
"""

import argparse
from pathlib import Path

from simace import setup_logging
from simace.plotting.compare import group_reps
from simace.plotting.compare.observed_scale import compare_observed_vs_liability_h2


def main(argv: list[str] | None = None) -> None:
    """Parse flags, group the replicate directories per scenario, and render."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="replicate directories, scenario-major")
    parser.add_argument("--output", default="docs/images/examples/models/observed_vs_liability.png")
    parser.add_argument("--labels", nargs="+", default=["LTM", "Cure-frailty (lognormal)", "Frailty (Weibull)"])
    parser.add_argument("--reps-per-scenario", nargs="+", type=int, required=True)
    parser.add_argument("--input-h2", type=float, required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    setup_logging(log_file=args.log, tag="examples/observed_vs_liability_h2")

    scenarios = group_reps(args.labels, args.inputs, args.reps_per_scenario)
    compare_observed_vs_liability_h2(
        scenarios,
        Path(args.output),
        trait=1,
        input_h2=args.input_h2,
    )


if __name__ == "__main__":
    main()
