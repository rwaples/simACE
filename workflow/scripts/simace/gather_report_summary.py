"""Gather report summaries - Snakemake wrapper with CLI fallback."""

from simace import _snakemake_tag, setup_logging
from simace.analysis.gather import cli as _cli
from simace.analysis.gather import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    report_files = snakemake.input.reports
    output_path = snakemake.output.tsv

    main(report_files, output_path)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
