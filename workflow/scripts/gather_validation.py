"""Gather validation results - Snakemake wrapper with CLI fallback."""
from sim_ace.gather import main, cli as _cli


def _run_snakemake():
    validation_files = snakemake.input.validations
    output_path = snakemake.output.tsv

    main(validation_files, output_path)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
