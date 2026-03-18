"""Compute simple LTM phenotype statistics - Snakemake wrapper with CLI fallback."""

from sim_ace import setup_logging
from sim_ace.simple_ltm_stats import cli as _cli
from sim_ace.simple_ltm_stats import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    phenotype_path = snakemake.input.phenotype
    seed = snakemake.params.seed
    stats_output = snakemake.output.stats
    samples_output = snakemake.output.samples

    extra_tetrachoric = snakemake.params.get("extra_tetrachoric", True)

    skip_2nd_cousins = snakemake.params.get("skip_2nd_cousins", True)

    case_ascertainment_ratio = snakemake.params.get("case_ascertainment_ratio", 1.0)

    main(
        phenotype_path,
        stats_output,
        samples_output,
        seed=seed,
        extra_tetrachoric=extra_tetrachoric,
        pedigree_path=snakemake.input.pedigree,
        skip_2nd_cousins=skip_2nd_cousins,
        case_ascertainment_ratio=case_ascertainment_ratio,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
