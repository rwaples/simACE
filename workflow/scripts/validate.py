"""ACE simulation validation - Snakemake wrapper with CLI fallback."""

import yaml

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.utils import to_native
from sim_ace.validate import cli as _cli
from sim_ace.validate import run_validation


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    pedigree_path = snakemake.input.pedigree
    params_path = snakemake.input.params
    output_path = snakemake.output.report

    results = run_validation(pedigree_path, params_path)
    results = to_native(results)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
