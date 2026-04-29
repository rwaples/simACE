"""Convert parquet to TSV - Snakemake wrapper with CLI fallback."""

from simace import _snakemake_tag, setup_logging
from simace.core.parquet_to_tsv import cli as _cli
from simace.core.parquet_to_tsv import convert
from simace.core.snakemake_adapter import cli_or_snakemake


def _run() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    convert(
        snakemake.input[0],
        snakemake.output[0],
        float_precision=snakemake.params.get("float_precision", 4),
        gzip=snakemake.params.get("gzip", True),
    )


cli_or_snakemake(_cli, _run, globals())
