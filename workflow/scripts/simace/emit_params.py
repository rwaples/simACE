"""Emit params.yaml for a scenario replicate - Snakemake wrapper with CLI fallback."""

from functools import partial

from simace import _snakemake_tag, setup_logging
from simace.core.snakemake_adapter import cli_or_snakemake, run_wrapper
from simace.core.yaml_io import dump_yaml
from simace.simulation.emit_params import cli as _cli
from simace.simulation.emit_params import emit_params


def _run() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    run_wrapper(
        snakemake,
        emit_params,
        inputs={},
        output="params",
        writer=partial(dump_yaml, sort_keys=True),
    )


cli_or_snakemake(_cli, _run, globals())
