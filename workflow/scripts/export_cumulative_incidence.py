"""Cumulative incidence TSV export — Snakemake wrapper with CLI fallback."""

import argparse

import pandas as pd

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.analysis.export_tables import export_cumulative_incidence


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    phenotype_df = pd.read_parquet(snakemake.input.phenotype)
    export_cumulative_incidence(
        phenotype_df,
        censor_age=float(snakemake.params.censor_age),
        out_path=snakemake.output[0],
    )


def _cli():
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Export cumulative incidence TSV")
    parser.add_argument("--phenotype", required=True, help="phenotype.parquet")
    parser.add_argument("--censor-age", type=float, required=True)
    parser.add_argument("--out", required=True, help="output TSV path")
    parser.add_argument("--n-points", type=int, default=200)
    add_logging_args(parser)
    args = parser.parse_args()
    init_logging(args)
    export_cumulative_incidence(
        pd.read_parquet(args.phenotype),
        censor_age=args.censor_age,
        out_path=args.out,
        n_points=args.n_points,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
