"""Convert parquet to TSV - Snakemake wrapper with CLI fallback."""

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.core.parquet_to_tsv import cli as _cli
from sim_ace.core.parquet_to_tsv import convert


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    precision = snakemake.params.get("float_precision", 4)
    use_gzip = snakemake.params.get("gzip", True)
    convert(snakemake.input[0], snakemake.output[0], float_precision=precision, gzip=use_gzip)


try:
    snakemake
except NameError:
    if __name__ == "__main__":
        _cli()
else:
    _run_snakemake()
