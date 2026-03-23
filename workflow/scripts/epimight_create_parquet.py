"""Snakemake wrapper for epimight/create_parquet.py."""

import sys
from pathlib import Path

from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))

    # Import here so standalone CLI still works without snakemake
    from epimight.create_parquet import main as _cli

    output_dir = str(Path(snakemake.output.t1).parent)
    sys.argv = [
        "epimight_create_parquet",
        "--phenotype",
        snakemake.input.phenotype,
        "--output-dir",
        output_dir,
    ]
    _cli()


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        from epimight.create_parquet import main as _cli

        _cli()
    else:
        _run_snakemake()
