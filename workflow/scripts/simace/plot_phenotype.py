"""Plot phenotype distributions - Snakemake wrapper with CLI fallback."""

from pathlib import Path

from simace import _snakemake_tag, setup_logging
from simace.plotting.plot_phenotype import cli as _cli
from simace.plotting.plot_phenotype import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    report_paths = snakemake.input.report
    plot_payload_paths = snakemake.input.plot_payload
    sample_paths = snakemake.input.samples
    censor_age = snakemake.params.censor_age
    gen_censoring = snakemake.params.gen_censoring or None
    plot_format = snakemake.params.plot_format
    output_dir = Path(snakemake.output[0]).parent

    max_degree = int(getattr(snakemake.params, "max_degree", 2))

    main(
        report_paths,
        plot_payload_paths,
        sample_paths,
        output_dir,
        censor_age,
        gen_censoring=gen_censoring,
        plot_ext=plot_format,
        max_degree=max_degree,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
