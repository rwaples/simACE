"""``simace gather <folder>``: summarize every rep report in a folder and plot it."""

from __future__ import annotations

__all__ = ["cli"]

import argparse
import sys
from pathlib import Path

from simace.cli.layout import Layout, RepArtifact


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Write ``report_summary.tsv`` for a folder, then its validation plots and atlas.

    Only reps with a ``run.yaml`` count: a rep that failed or was interrupted
    has none. ``gather`` reads no config, so it does not know about reps or
    scenarios the config has since dropped; ``simace ls`` shows those.
    """
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(
        prog=prog, description="Summarize a folder's rep reports and render its validation atlas"
    )
    add_logging_args(parser)
    parser.add_argument("folder", help="Folder under the results root")
    parser.add_argument("--format", choices=("html", "pdf"), default="html", help="Atlas format (default: html)")
    parser.add_argument(
        "--plot-format", choices=("png", "pdf"), default="png", help="Validation plot format (default: png)"
    )
    parser.add_argument("--results", type=Path, default=Path("results"), help="Results root (default: results)")
    args = parser.parse_args(argv)
    init_logging(args)

    layout = Layout(root=args.results)
    found = sorted((args.results / args.folder).glob(f"*/rep*/{RepArtifact.REPORT}"))
    reports = [str(p) for p in found if (p.parent / RepArtifact.RUN_MANIFEST).exists()]
    unfinished = [p.parent for p in found if not (p.parent / RepArtifact.RUN_MANIFEST).exists()]
    for rep_dir in unfinished:
        print(
            f"simace gather: skipping {rep_dir} (no {RepArtifact.RUN_MANIFEST}; the rep did not finish)",
            file=sys.stderr,
        )
    if not reports:
        print(f"simace gather: no finished reps under {args.results / args.folder}", file=sys.stderr)
        raise SystemExit(1)

    from simace.analysis.gather import main as gather_reports
    from simace.plotting.plot_validation import main as plot_validation

    summary = layout.folder_summary(args.folder)
    gather_reports(reports, str(summary))
    plot_validation(
        str(summary), layout.folder_plots(args.folder), plot_ext=args.plot_format, atlas_name=f"atlas.{args.format}"
    )
