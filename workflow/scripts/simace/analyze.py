"""Combined Validate + Stats analysis - Snakemake wrapper with CLI fallback."""

from simace import _snakemake_tag, setup_logging
from simace.analysis.analyze import cli as _cli
from simace.analysis.analyze import run_analysis


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    p = snakemake.params

    gen_censoring = p.get("gen_censoring") or None

    run_analysis(
        pedigree_full_path=snakemake.input.pedigree_full,
        params_path=snakemake.input.params,
        trait_path=snakemake.input.trait,
        pedigree_path=snakemake.input.pedigree,
        report_output=snakemake.output.report,
        samples_output=snakemake.output.samples,
        seed=p.seed,
        censor_age=p.censor_age,
        gen_censoring=gen_censoring,
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
