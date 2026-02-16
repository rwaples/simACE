"""Liability threshold phenotype - Snakemake wrapper with CLI fallback."""
import pandas as pd

from sim_ace import setup_logging
from sim_ace.threshold import run_threshold, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    params = snakemake.params

    param_dict = {
        "G_pheno": params.G_pheno,
        "prevalence1": params.prevalence1,
        "prevalence2": params.prevalence2,
    }

    phenotype = run_threshold(pedigree, param_dict)
    phenotype.to_parquet(snakemake.output.phenotype, index=False)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
