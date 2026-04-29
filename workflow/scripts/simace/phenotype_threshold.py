"""Liability threshold phenotype - Snakemake wrapper with CLI fallback."""

import pandas as pd

from simace import _snakemake_tag, setup_logging
from simace.phenotyping.threshold import cli as _cli
from simace.phenotyping.threshold import run_threshold

_DEFAULT_THRESHOLD_PREVALENCE = (0.10, 0.20)


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    params = snakemake.params

    # PR3: prevalence lives inside phenotype_params for adult / cure_frailty.
    # For frailty / first_passage traits, fall back to a documented default
    # so the (model-agnostic) threshold path still has a prevalence to use.
    pp1 = dict(params.phenotype_params1 or {})
    pp2 = dict(params.phenotype_params2 or {})
    param_dict = {
        "G_pheno": params.G_pheno,
        "prevalence1": pp1.get("prevalence", _DEFAULT_THRESHOLD_PREVALENCE[0]),
        "prevalence2": pp2.get("prevalence", _DEFAULT_THRESHOLD_PREVALENCE[1]),
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
