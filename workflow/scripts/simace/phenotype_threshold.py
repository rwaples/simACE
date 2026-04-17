"""Liability threshold phenotype - Snakemake wrapper with CLI fallback."""

import pandas as pd

from simace import _snakemake_tag, setup_logging
from simace.phenotyping.threshold import cli as _cli
from simace.phenotyping.threshold import run_threshold


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    params = snakemake.params

    param_dict = {
        "G_pheno": params.G_pheno,
        "prevalence1": params.prevalence1,
        "prevalence2": params.prevalence2,
        "standardize": params.standardize,
    }

    phenotype = run_threshold(pedigree, param_dict)
    phenotype.to_parquet(snakemake.output.phenotype, index=False)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
