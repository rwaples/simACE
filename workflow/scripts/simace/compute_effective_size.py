"""Compute Ne estimators - Snakemake wrapper with CLI fallback."""

from simace import _snakemake_tag, setup_logging
from simace.analysis.stats.effective_size import cli as _cli
from simace.analysis.stats.effective_size import main
from simace.core.snakemake_adapter import cli_or_snakemake


def _run() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    main(
        pedigree_path=snakemake.input.pedigree,
        phenotype_path=snakemake.input.phenotype,
        params_path=snakemake.input.params,
        output_path=snakemake.output.stats,
    )


cli_or_snakemake(_cli, _run, globals())
