"""Generate EPIMIGHT bias analysis plots — Snakemake wrapper with CLI fallback."""

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.plot_epimight_bias import cli as _cli
from sim_ace.plot_epimight_bias import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    main(
        tsv_path=snakemake.input.tsv,
        output_path=snakemake.output.atlas,
        include_dilution_correction=snakemake.params.get("include_dilution_correction", True),
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
