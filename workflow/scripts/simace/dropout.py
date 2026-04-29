"""Pedigree dropout - Snakemake wrapper with CLI fallback."""

import pandas as pd

from simace import _snakemake_tag, setup_logging
from simace.core.parquet import save_parquet
from simace.core.snakemake_adapter import cli_or_snakemake, run_wrapper
from simace.sampling.dropout import cli as _cli
from simace.sampling.dropout import run_dropout


def _run() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    run_wrapper(
        snakemake,
        run_dropout,
        inputs={"pedigree": pd.read_parquet},
        output="pedigree",
        writer=save_parquet,
    )


cli_or_snakemake(_cli, _run, globals())
