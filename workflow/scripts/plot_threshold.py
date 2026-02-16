"""Plot threshold phenotype distributions - Snakemake wrapper with CLI fallback."""
from pathlib import Path

from sim_ace.plot_threshold import main, cli as _cli


def _run_snakemake():
    stats_paths = snakemake.input.stats
    sample_paths = snakemake.input.samples
    prevalence1 = snakemake.params.prevalence1
    prevalence2 = snakemake.params.prevalence2
    output_dir = Path(snakemake.output[0]).parent

    main(stats_paths, sample_paths, output_dir, prevalence1, prevalence2)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
