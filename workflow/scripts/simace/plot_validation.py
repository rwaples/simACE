"""Plot validation results - Snakemake wrapper with CLI fallback."""

from pathlib import Path

from simace import _snakemake_tag, setup_logging
from simace.plotting.plot_validation import assemble_validation_atlas, main
from simace.plotting.plot_validation import cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    plot_format = snakemake.params.plot_format
    atlas_out = next(Path(o) for o in snakemake.output if Path(o).suffix in (".html", ".pdf"))

    if atlas_out.suffix == ".pdf":
        # On-demand PDF rule: the validation plots already exist as rule inputs,
        # so only the atlas is reassembled — main() would re-own the plot files.
        assemble_validation_atlas(atlas_out.parent, atlas_out.name, plot_ext=plot_format)
    else:
        main(snakemake.input.tsv, atlas_out.parent, plot_ext=plot_format, atlas_name=atlas_out.name)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
