"""Compute threshold phenotype statistics - Snakemake wrapper with CLI fallback."""
from sim_ace.threshold_stats import main, cli as _cli


def _run_snakemake():
    phenotype_path = snakemake.input.phenotype
    seed = snakemake.params.seed
    stats_output = snakemake.output.stats
    samples_output = snakemake.output.samples

    main(phenotype_path, stats_output, samples_output, seed=seed)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
