"""Plot simple LTM phenotype distributions - Snakemake wrapper with CLI fallback."""

from pathlib import Path

from sim_ace import _snakemake_tag, setup_logging
from fit_ace.plotting.plot_simple_ltm import cli as _cli
from fit_ace.plotting.plot_simple_ltm import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    stats_paths = snakemake.input.stats
    sample_paths = snakemake.input.samples
    prevalence1 = snakemake.params.prevalence1
    prevalence2 = snakemake.params.prevalence2
    plot_format = snakemake.params.plot_format
    output_dir = Path(snakemake.output[0]).parent

    ace_params = {}
    for key in ("A1", "C1", "A2", "C2"):
        val = getattr(snakemake.params, key, None)
        if val is not None:
            ace_params[key] = val

    main(
        stats_paths,
        sample_paths,
        output_dir,
        prevalence1,
        prevalence2,
        plot_ext=plot_format,
        ace_params=ace_params or None,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
