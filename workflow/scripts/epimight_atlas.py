"""Snakemake wrapper for epimight/plot_epimight.py."""

from pathlib import Path

from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))

    from epimight.plot_epimight import assemble_epimight_atlas

    # The epimight dir is the parent of the tsv/ directory
    epimight_dir = Path(snakemake.output.atlas).parent.parent
    assemble_epimight_atlas(epimight_dir, scenario=snakemake.wildcards.scenario)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        from epimight.plot_epimight import main as _cli

        _cli()
    else:
        _run_snakemake()
