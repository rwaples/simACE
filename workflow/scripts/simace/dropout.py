"""Pedigree dropout - Snakemake wrapper with CLI fallback."""

import pandas as pd

from simace import _snakemake_tag, setup_logging
from simace.core.parquet import save_parquet
from simace.sampling.dropout import cli as _cli
from simace.sampling.dropout import run_dropout


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    pedigree = pd.read_parquet(snakemake.input.pedigree)

    param_dict = {
        "pedigree_dropout_rate": snakemake.params.dropout_rate,
        "seed": snakemake.params.seed,
    }

    result = run_dropout(pedigree, param_dict)
    save_parquet(result, snakemake.output.pedigree)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
