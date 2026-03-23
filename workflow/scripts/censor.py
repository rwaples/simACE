"""Observation censoring - Snakemake wrapper with CLI fallback."""

import pandas as pd

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.censor import cli as _cli
from sim_ace.censor import run_censor


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    phenotype = pd.read_parquet(snakemake.input.phenotype)
    params = snakemake.params

    gen_censoring_raw = params.gen_censoring
    gen_censoring = {int(k): v for k, v in gen_censoring_raw.items()}

    param_dict = {
        "censor_age": params.censor_age,
        "seed": params.seed,
        "gen_censoring": gen_censoring,
        "death_scale": params.death_scale,
        "death_rho": params.death_rho,
    }

    result = run_censor(phenotype, param_dict)
    result.to_parquet(snakemake.output.phenotype, index=False)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
