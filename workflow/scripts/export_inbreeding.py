"""Per-individual inbreeding export — Snakemake wrapper with CLI fallback."""

import argparse

import pandas as pd

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.analysis.export_tables import export_inbreeding

_PEDIGREE_COLS = ["id", "mother", "father", "twin", "sex", "generation"]


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    pedigree_df = pd.read_parquet(snakemake.input.pedigree, columns=_PEDIGREE_COLS)
    export_inbreeding(pedigree_df, out_path=snakemake.output[0])


def _cli():
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Export per-individual inbreeding (F > 0 only)")
    parser.add_argument("--pedigree", required=True, help="pedigree.parquet")
    parser.add_argument("--out", required=True, help="output TSV path")
    add_logging_args(parser)
    args = parser.parse_args()
    init_logging(args)
    export_inbreeding(
        pd.read_parquet(args.pedigree, columns=_PEDIGREE_COLS),
        out_path=args.out,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
