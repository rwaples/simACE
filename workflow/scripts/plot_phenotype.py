"""Plot phenotype distributions - Snakemake wrapper with CLI fallback."""
from pathlib import Path

from sim_ace import setup_logging
from sim_ace.plot_phenotype import main, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    stats_paths = snakemake.input.stats
    sample_paths = snakemake.input.samples
    censor_age = snakemake.params.censor_age
    young_gen_censoring = snakemake.params.young_gen_censoring
    middle_gen_censoring = snakemake.params.middle_gen_censoring
    old_gen_censoring = snakemake.params.old_gen_censoring
    output_dir = Path(snakemake.output[0]).parent

    main(stats_paths, sample_paths, output_dir, censor_age,
         young_gen_censoring, middle_gen_censoring, old_gen_censoring)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
