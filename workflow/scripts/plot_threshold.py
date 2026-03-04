"""Plot threshold phenotype distributions - Snakemake wrapper with CLI fallback."""
from pathlib import Path

from sim_ace import setup_logging
from sim_ace.plot_threshold import main, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    stats_paths = snakemake.input.stats
    sample_paths = snakemake.input.samples
    prevalence1 = snakemake.params.prevalence1
    prevalence2 = snakemake.params.prevalence2
    plot_format = snakemake.params.plot_format
    output_dir = Path(snakemake.output[0]).parent

    main(stats_paths, sample_paths, output_dir, prevalence1, prevalence2,
         plot_ext=plot_format)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
