"""Ascertainment - Snakemake wrapper with CLI fallback.

Custom wrapper (not run_wrapper-based) because the rule has two named
outputs and run_wrapper only writes one.
"""

from simace import _snakemake_tag, setup_logging
from simace.ascertainment import cli as _cli
from simace.ascertainment import copy_passthrough_if_possible, run_ascertainment
from simace.core.parquet import load_parquet, save_parquet
from simace.core.snakemake_adapter import cli_or_snakemake


def _run() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    if copy_passthrough_if_possible(
        snakemake.input.pedigree,
        snakemake.input.trait,
        snakemake.output.pedigree,
        snakemake.output.trait,
        dropout_rate=snakemake.params.dropout_rate,
        N_sample=snakemake.params.N_sample,
    ):
        return

    df_ped = load_parquet(snakemake.input.pedigree)
    df_trait = load_parquet(snakemake.input.trait)
    df_ped_out, df_trait_out = run_ascertainment(
        df_ped,
        df_trait,
        dropout_rate=snakemake.params.dropout_rate,
        case_ascertainment_ratio=snakemake.params.case_ascertainment_ratio,
        N_sample=snakemake.params.N_sample,
        seed=snakemake.params.seed,
    )
    save_parquet(df_ped_out, snakemake.output.pedigree)
    save_parquet(df_trait_out, snakemake.output.trait)


cli_or_snakemake(_cli, _run, globals())
