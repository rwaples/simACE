"""Sparse GRM export — Snakemake wrapper with CLI fallback.

Writes the ``ace_sreml`` binary CSC GRM plus its ``.grm.id`` companion,
with family IDs (FIDs) derived from founder-couple connected components
of the pedigree so external tools (GCTA / PLINK / sparseREML) see the
family structure.
"""

import argparse
from pathlib import Path

import pandas as pd

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.analysis.export_tables import export_sparse_grm


def _prefix_from_outputs(bin_path: str) -> Path:
    """Strip the ``.grm.sp.bin`` suffix used by ``export_sparse_grm_binary``."""
    p = Path(bin_path)
    name = p.name
    suffix = ".grm.sp.bin"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected GRM output path (expected *{suffix}): {bin_path}")
    return p.with_name(name[: -len(suffix)])


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    pedigree_df = pd.read_parquet(snakemake.input.pedigree)
    export_sparse_grm(
        pedigree_df,
        prefix=_prefix_from_outputs(snakemake.output.bin),
        threshold=float(snakemake.params.grm_threshold),
    )


def _cli():
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Export sparse GRM (ace_sreml binary)")
    parser.add_argument("--pedigree", required=True, help="pedigree.parquet")
    parser.add_argument(
        "--prefix",
        required=True,
        help="output prefix; writes <prefix>.grm.sp.bin and <prefix>.grm.id",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="drop off-diagonal entries below this GRM-scale threshold (default: 0.05)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    init_logging(args)
    export_sparse_grm(
        pd.read_parquet(args.pedigree),
        prefix=args.prefix,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
