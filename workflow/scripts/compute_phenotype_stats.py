"""Compute phenotype statistics - Snakemake wrapper with CLI fallback."""
from sim_ace import setup_logging
from sim_ace.stats import main, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    phenotype_path = snakemake.input.phenotype
    seed = snakemake.params.seed
    censor_age = snakemake.params.censor_age
    young_gen_censoring = snakemake.params.get("young_gen_censoring", None)
    middle_gen_censoring = snakemake.params.get("middle_gen_censoring", None)
    old_gen_censoring = snakemake.params.get("old_gen_censoring", None)
    stats_output = snakemake.output.stats
    samples_output = snakemake.output.samples

    main(phenotype_path, censor_age, stats_output, samples_output,
         seed=seed, young_gen_censoring=young_gen_censoring,
         middle_gen_censoring=middle_gen_censoring,
         old_gen_censoring=old_gen_censoring)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
