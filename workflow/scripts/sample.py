"""Subsample phenotyped individuals - Snakemake wrapper with CLI fallback."""

import pandas as pd

from sim_ace import setup_logging
from sim_ace.sample import cli as _cli
from sim_ace.sample import run_sample
from sim_ace.utils import save_parquet


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    phenotype = pd.read_parquet(snakemake.input.phenotype)

    param_dict = {
        "N_sample": snakemake.params.N_sample,
        "case_ascertainment_ratio": snakemake.params.case_ascertainment_ratio,
        "seed": snakemake.params.seed,
    }

    result = run_sample(phenotype, param_dict)
    save_parquet(result, snakemake.output.phenotype)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
