"""The per-rep ``simace validate --checks`` primitive: run on-demand checks on one rep's files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Run on-demand checks (:mod:`.on_demand`) on one rep and write its ``validation.yaml``.

    The checks read the recorded pedigree and regenerate what else they need
    from ``--seed`` and the phenotype flags (:class:`.calibrated.RepInputs`). ``simace validate <scenario>``
    runs this once per rep.
    """
    from simace.cli.cohort_stage import add_phenotype_args, phenotype_kwargs
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging
    from simace.core.publish import publish
    from simace.core.yaml_io import dump_yaml

    from .calibrated import RepInputs
    from .on_demand import ON_DEMAND_CHECKS, run_on_demand

    parser = argparse.ArgumentParser(prog=prog, description="Run on-demand validation checks on one rep")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--pedigree", required=True, help="The rep's recorded pedigree parquet (layout 2)")
    parser.add_argument("--checks", nargs="+", choices=sorted(ON_DEMAND_CHECKS), required=True)
    parser.add_argument("--seed", type=int, required=True, help="The rep seed")
    parser.add_argument("--output", required=True, help="Output validation.yaml")
    parser.add_argument(
        "--source",
        default=None,
        help="The build's git describe, as `simace validate <scenario>` saw it (default: this checkout's)",
    )
    add_phenotype_args(parser.add_argument_group("phenotype"))
    given = sys.argv[1:] if argv is None else argv
    asks_help = "-h" in given or "--help" in given
    if not asks_help and not any(a == "--checks" or a.startswith("--checks=") for a in given):
        parser.error(
            "--checks is required; the run without it was removed (ADR 0028), and `simace analyze` runs those checks"
        )
    args = parser.parse_args(argv)
    init_logging(args)

    inputs = RepInputs(Path(args.pedigree), args.seed, phenotype_kwargs(args))
    from simace.cli.manifest import source_ref

    result = run_on_demand(inputs, args.checks, args.source or source_ref())
    with publish(args.output) as (tmp,):
        dump_yaml(result, tmp)
