"""Plot validation results - Snakemake wrapper with CLI fallback."""

from pathlib import Path

from simace import _snakemake_tag, setup_logging
from simace.plotting.plot_validation import assemble_validation_atlas, main
from simace.plotting.plot_validation import cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    plot_format = snakemake.params.plot_format
    outputs = [Path(o) for o in snakemake.output]
    atlas_out = next(o for o in outputs if o.suffix in (".html", ".pdf"))
    output_dir = atlas_out.parent

    if atlas_out.suffix == ".pdf" and len(outputs) == 1:
        # On-demand PDF rule: validation plots already exist as inputs, so only
        # the atlas is (re)assembled — main() would otherwise re-own the plots.
        assemble_validation_atlas(output_dir, atlas_out.name, plot_ext=plot_format)
    else:
        main(snakemake.input.tsv, output_dir, plot_ext=plot_format, atlas_name=atlas_out.name)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
