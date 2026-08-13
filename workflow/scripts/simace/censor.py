"""Observation censoring - Snakemake wrapper with CLI fallback."""

from simace import _snakemake_tag, setup_logging
from simace.censoring.censor import cli as _cli
from simace.censoring.censor import run_censor
from simace.core.parquet import load_parquet, save_parquet
from simace.core.snakemake_adapter import cli_or_snakemake, run_wrapper


def _run() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    run_wrapper(
        snakemake,
        run_censor,
        inputs={"phenotype": load_parquet, "pedigree": load_parquet},
        output="phenotype",
        writer=save_parquet,
    )


cli_or_snakemake(_cli, _run, globals())
