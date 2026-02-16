"""Prepare Survival Kit input files - Snakemake wrapper with CLI fallback."""
import pandas as pd

from sim_ace import setup_logging
from sim_ace.prepare import run_prepare, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    phenotype = pd.read_parquet(snakemake.input.phenotype)

    scenario = snakemake.wildcards.scenario
    rep = snakemake.wildcards.rep
    trait = snakemake.wildcards.trait

    outputs = {
        "data": snakemake.output.data,
        "codelist": snakemake.output.codelist,
        "varlist": snakemake.output.varlist,
        "pedigree_ped": snakemake.output.pedigree_ped,
        "weibull_config": snakemake.output.weibull_config,
    }

    run_prepare(pedigree, phenotype, scenario, rep, trait, outputs)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
