"""Frailty phenotype simulation — Snakemake script with CLI fallback."""
import pandas as pd

from sim_ace import setup_logging
from sim_ace.phenotype import run_phenotype, cli as _cli


def _run_snakemake() -> None:
    setup_logging(log_file=snakemake.log[0])   
    pedigree = pd.read_parquet(snakemake.input.pedigree)   
    p = snakemake.params   

    param_dict = {
        "G_pheno":       p.G_pheno,
        "seed":          p.seed,
        "standardize":   p.standardize,
        "beta1":         p.beta1,
        "beta_sex1":     p.beta_sex1,
        "hazard_model1": p.hazard_model1,
        "hazard_params1": p.hazard_params1,
        "beta2":         p.beta2,
        "beta_sex2":     p.beta_sex2,
        "hazard_model2": p.hazard_model2,
        "hazard_params2": p.hazard_params2,
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