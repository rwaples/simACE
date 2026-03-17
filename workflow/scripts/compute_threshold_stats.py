"""Compute threshold phenotype statistics - Snakemake wrapper with CLI fallback."""

from sim_ace import setup_logging
from sim_ace.threshold_stats import cli as _cli
from sim_ace.threshold_stats import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    phenotype_path = snakemake.input.phenotype
    seed = snakemake.params.seed
    stats_output = snakemake.output.stats
    samples_output = snakemake.output.samples

    extra_tetrachoric = snakemake.params.get("extra_tetrachoric", True)

    main(phenotype_path, stats_output, samples_output, seed=seed, extra_tetrachoric=extra_tetrachoric)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
