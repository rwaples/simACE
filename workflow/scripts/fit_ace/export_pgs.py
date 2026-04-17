"""Proxy polygenic score export — Snakemake wrapper with CLI fallback."""

import argparse

import numpy as np
import pandas as pd

from sim_ace import _snakemake_tag, setup_logging
from fit_ace.exports.tables import export_pgs

_PEDIGREE_COLS = ["id", "sex", "generation", "A1", "A2"]

# Distinct from any other rule's rep-seed usage so PGS noise is
# independent of simulation / phenotyping / sampling RNG streams.
_PGS_DOMAIN_TAG = 0x504753204558504F  # 'PGS EXPO'


def _derive_sub_seed(rep_seed: int) -> int:
    """Deterministic u64 sub-seed from the rep seed + a fixed domain tag."""
    ss = np.random.SeedSequence([int(rep_seed), _PGS_DOMAIN_TAG])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    pedigree_df = pd.read_parquet(snakemake.input.pedigree, columns=_PEDIGREE_COLS)
    sub_seed = _derive_sub_seed(snakemake.params.seed)
    export_pgs(
        pedigree_df,
        r2=tuple(snakemake.params.pgs_r2),
        rA=float(snakemake.params.rA),
        var_A=(float(snakemake.params.A1), float(snakemake.params.A2)),
        sub_seed=sub_seed,
        out_path=snakemake.output.parquet,
        meta_path=snakemake.output.meta,
    )


def _cli():
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Export proxy polygenic scores")
    parser.add_argument("--pedigree", required=True, help="pedigree.parquet")
    parser.add_argument(
        "--pgs-r2",
        type=float,
        nargs=2,
        required=True,
        metavar=("R2_1", "R2_2"),
        help="expected squared cor(PGS, A) per trait",
    )
    parser.add_argument("--rA", type=float, required=True)
    parser.add_argument("--var-A", type=float, nargs=2, required=True, metavar=("VAR_A1", "VAR_A2"))
    parser.add_argument("--seed", type=int, required=True, help="rep seed (scenario seed + rep - 1)")
    parser.add_argument("--out", required=True, help="output parquet path")
    parser.add_argument("--meta", default=None, help="output JSON sidecar path (default: <out>.meta.json)")
    add_logging_args(parser)
    args = parser.parse_args()
    init_logging(args)
    export_pgs(
        pd.read_parquet(args.pedigree, columns=_PEDIGREE_COLS),
        r2=tuple(args.pgs_r2),
        rA=args.rA,
        var_A=tuple(args.var_A),
        sub_seed=_derive_sub_seed(args.seed),
        out_path=args.out,
        meta_path=args.meta,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
