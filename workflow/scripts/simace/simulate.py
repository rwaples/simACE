"""ACE pedigree simulation - Snakemake wrapper with CLI fallback."""

from simace import _snakemake_tag, setup_logging
from simace.core.parquet import save_parquet
from simace.core.snakemake_adapter import cli_or_snakemake, run_wrapper
from simace.simulation.simulate import cli as _cli
from simace.simulation.simulate import run_simulation


def _run() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    run_wrapper(
        snakemake,
        run_simulation,
        inputs={},
        output="pedigree",
        writer=save_parquet,
    )


cli_or_snakemake(_cli, _run, globals())
