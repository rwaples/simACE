"""Compute phenotype statistics - Snakemake wrapper with CLI fallback."""

from simace import _snakemake_tag, setup_logging
from simace.analysis.stats import cli as _cli
from simace.analysis.stats import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    p = snakemake.params

    gen_censoring = p.get("gen_censoring") or None

    main(
        snakemake.input.phenotype,
        p.censor_age,
        snakemake.output.stats,
        snakemake.output.samples,
        seed=p.seed,
        gen_censoring=gen_censoring,
        pedigree_path=snakemake.input.pedigree,
        max_degree=p.get("max_degree", 2),
        case_ascertainment_ratio=p.get("case_ascertainment_ratio", 1.0),
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
