"""Gather EPIMIGHT bias results — Snakemake wrapper with CLI fallback."""

from fit_ace.epimight.epimight_bias_analysis import cli as _cli
from fit_ace.epimight.epimight_bias_analysis import main
from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    main(
        results_dir=snakemake.params.results_dir,
        scenarios=snakemake.params.scenarios,
        output_path=snakemake.output.tsv,
        subdir=snakemake.params.get("subdir", "epimight"),
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
