"""Compute LTM Falconer h² per EPIMIGHT kind — Snakemake wrapper with CLI fallback."""

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.analysis.ltm_falconer import cli as _cli
from sim_ace.analysis.ltm_falconer import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    main(
        simple_ltm_path=snakemake.input.simple_ltm,
        pedigree_path=snakemake.input.pedigree,
        output_path=snakemake.output.json,
        kinds=snakemake.params.get("kinds", None),
        seed=snakemake.params.seed,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
