"""PA-FGRS atlas plotting - Snakemake wrapper."""

from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    import logging
    from pathlib import Path

    from fit_ace.plotting.plot_pafgrs import generate_atlas

    logger = logging.getLogger(__name__)

    base_dir = str(Path(snakemake.output.atlas).parent.parent)
    output_path = snakemake.output.atlas

    logger.info("Generating PA-FGRS atlas: %s", output_path)
    generate_atlas(base_dir, output_path)


try:
    snakemake
    _run_snakemake()
except NameError:
    raise SystemExit("This script must be run via Snakemake") from None
