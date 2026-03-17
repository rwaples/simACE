"""Frailty phenotype simulation — Snakemake script with CLI fallback."""

import pandas as pd

from sim_ace import setup_logging
from sim_ace.phenotype import cli as _cli
from sim_ace.phenotype import run_phenotype


def _run_snakemake() -> None:
    setup_logging(log_file=snakemake.log[0])
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    p = snakemake.params

    param_dict = {
        "G_pheno": p.G_pheno,
        "seed": p.seed,
        "standardize": p.standardize,
        "phenotype_model1": p.phenotype_model1,
        "phenotype_model2": p.phenotype_model2,
        "prevalence1": getattr(p, "prevalence1", 0.10),
        "prevalence2": getattr(p, "prevalence2", 0.20),
        "beta1": p.beta1,
        "beta_sex1": p.beta_sex1,
        "phenotype_params1": p.phenotype_params1,
        "beta2": p.beta2,
        "beta_sex2": p.beta_sex2,
        "phenotype_params2": p.phenotype_params2,
    }

    phenotype = run_phenotype(pedigree, param_dict)
    phenotype.to_parquet(snakemake.output.phenotype, index=False)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
