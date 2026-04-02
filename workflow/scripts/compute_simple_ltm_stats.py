"""Compute simple LTM phenotype statistics - Snakemake wrapper with CLI fallback."""

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.analysis.simple_ltm_stats import cli as _cli
from sim_ace.analysis.simple_ltm_stats import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    phenotype_path = snakemake.input.phenotype
    seed = snakemake.params.seed
    stats_output = snakemake.output.stats
    samples_output = snakemake.output.samples

    max_degree = snakemake.params.get("max_degree", 2)

    case_ascertainment_ratio = snakemake.params.get("case_ascertainment_ratio", 1.0)

    main(
        phenotype_path,
        stats_output,
        samples_output,
        seed=seed,
        pedigree_path=snakemake.input.pedigree,
        max_degree=max_degree,
        case_ascertainment_ratio=case_ascertainment_ratio,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
