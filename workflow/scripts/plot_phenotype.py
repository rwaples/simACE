"""Plot phenotype distributions - Snakemake wrapper with CLI fallback."""
from pathlib import Path

from sim_ace import setup_logging
from sim_ace.plot_phenotype import main, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    stats_paths = snakemake.input.stats
    sample_paths = snakemake.input.samples
    censor_age = snakemake.params.censor_age
    gen_censoring_raw = snakemake.params.gen_censoring
    gen_censoring = {int(k): v for k, v in gen_censoring_raw.items()} if gen_censoring_raw else None
    plot_format = snakemake.params.plot_format
    output_dir = Path(snakemake.output[0]).parent

    validation_paths = list(snakemake.input.validations)

    main(stats_paths, sample_paths, output_dir, censor_age,
         gen_censoring=gen_censoring, plot_ext=plot_format,
         validation_paths=validation_paths)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        _cli()
    else:
        _run_snakemake()
