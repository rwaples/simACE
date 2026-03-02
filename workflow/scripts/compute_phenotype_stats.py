"""Compute phenotype statistics - Snakemake wrapper with CLI fallback."""
from sim_ace import setup_logging
from sim_ace.stats import main, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    phenotype_path = snakemake.input.phenotype
    pedigree_path = snakemake.input.pedigree
    seed = snakemake.params.seed
    censor_age = snakemake.params.censor_age
    gen_censoring_raw = snakemake.params.get("gen_censoring", None)
    gen_censoring = {int(k): v for k, v in gen_censoring_raw.items()} if gen_censoring_raw else None
    stats_output = snakemake.output.stats
    samples_output = snakemake.output.samples

    weibull_params = {
        "trait1": {
            "beta": snakemake.params.beta1,
            "scale": snakemake.params.scale1,
            "rho": snakemake.params.rho1,
        },
        "trait2": {
            "beta": snakemake.params.beta2,
            "scale": snakemake.params.scale2,
            "rho": snakemake.params.rho2,
        },
    }

    extra_tetrachoric = snakemake.params.get("extra_tetrachoric", True)

    main(phenotype_path, censor_age, stats_output, samples_output,
         seed=seed, gen_censoring=gen_censoring,
         weibull_params=weibull_params,
         extra_tetrachoric=extra_tetrachoric,
         pedigree_path=pedigree_path)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
