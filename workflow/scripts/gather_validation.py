"""Gather validation results - Snakemake wrapper with CLI fallback."""

from sim_ace import setup_logging
from sim_ace.gather import cli as _cli
from sim_ace.gather import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    validation_files = snakemake.input.validations
    output_path = snakemake.output.tsv

    main(validation_files, output_path)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
