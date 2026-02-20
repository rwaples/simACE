"""Weibull frailty phenotype simulation - Snakemake wrapper with CLI fallback."""
import pandas as pd

from sim_ace import setup_logging
from sim_ace.phenotype import run_phenotype, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    params = snakemake.params

    gen_censoring_raw = params.gen_censoring
    gen_censoring = {int(k): v for k, v in gen_censoring_raw.items()}

    param_dict = {
        "G_pheno": params.G_pheno,
        "censor_age": params.censor_age,
        "seed": params.seed,
        "gen_censoring": gen_censoring,
        "death_scale": params.death_scale,
        "death_rho": params.death_rho,
        "beta1": params.beta1,
        "scale1": params.scale1,
        "rho1": params.rho1,
        "beta2": params.beta2,
        "scale2": params.scale2,
        "rho2": params.rho2,
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
