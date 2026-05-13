"""Plot Ne atlas - Snakemake wrapper with CLI fallback."""

from pathlib import Path

from simace import _snakemake_tag, setup_logging
from simace.core.snakemake_adapter import cli_or_snakemake
from simace.plotting.plot_effective_size import cli as _cli
from simace.plotting.plot_effective_size import main


def _run() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    main(
        yaml_paths=list(snakemake.input.yamls),
        params_path=snakemake.input.params,
        output_dir=str(Path(snakemake.output.atlas).parent),
        plot_ext=snakemake.params.plot_format,
    )


cli_or_snakemake(_cli, _run, globals())
