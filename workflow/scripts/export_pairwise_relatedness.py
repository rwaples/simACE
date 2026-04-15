"""Pairwise relatedness TSV export — Snakemake wrapper with CLI fallback."""

import argparse

import pandas as pd

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.analysis.export_tables import export_pairwise_relatedness


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    pedigree_df = pd.read_parquet(snakemake.input.pedigree)
    export_pairwise_relatedness(
        pedigree_df,
        out_path=snakemake.output[0],
        min_kinship=float(snakemake.params.min_kinship),
    )


def _cli():
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Export pairwise relatedness TSV")
    parser.add_argument("--pedigree", required=True, help="pedigree.parquet")
    parser.add_argument("--out", required=True, help="output TSV path")
    parser.add_argument(
        "--min-kinship",
        type=float,
        default=0.0625,
        help="drop pairs with kinship below this threshold (default: 0.0625, 1st-cousin)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    init_logging(args)
    export_pairwise_relatedness(
        pd.read_parquet(args.pedigree),
        out_path=args.out,
        min_kinship=args.min_kinship,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
