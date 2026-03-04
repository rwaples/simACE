"""Plot validation results - Snakemake wrapper with CLI fallback."""
from pathlib import Path

from sim_ace import setup_logging
from sim_ace.plot_validation import main, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    tsv_path = snakemake.input.tsv
    plot_format = snakemake.params.plot_format
    output_dir = Path(snakemake.output[0]).parent

    main(tsv_path, output_dir, plot_ext=plot_format)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
