"""Weibull frailty phenotype simulation - Snakemake wrapper with CLI fallback."""
import pandas as pd

from sim_ace.phenotype import run_phenotype, cli as _cli


def _run_snakemake():
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    params = snakemake.params

    param_dict = {
        "G_pheno": params.G_pheno,
        "censor_age": params.censor_age,
        "seed": params.seed,
        "young_gen_censoring": params.young_gen_censoring,
        "middle_gen_censoring": params.middle_gen_censoring,
        "old_gen_censoring": params.old_gen_censoring,
        "death_rate": params.death_rate,
        "death_k": params.death_k,
        "beta1": params.beta1,
        "rate1": params.rate1,
        "k1": params.k1,
        "beta2": params.beta2,
        "rate2": params.rate2,
        "k2": params.k2,
        "standardize": params.standardize,
    }

    phenotype = run_phenotype(pedigree, param_dict)
    phenotype.to_parquet(snakemake.output.phenotype, index=False)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
