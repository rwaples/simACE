"""``simace gather <folder>``: summarize every complete rep report in a folder and plot it."""

from __future__ import annotations

__all__ = ["cli"]

import argparse
import sys
from typing import TYPE_CHECKING, Any

from simace.cli.layout import RepArtifact, add_root_args, resolve_roots

if TYPE_CHECKING:
    from pathlib import Path

    from simace.cli.layout import Layout


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Write ``report_summary.tsv`` for a folder, then its validation plots and atlas.

    A rep counts only when it is complete: it has a ``run.yaml`` (a rep that
    failed or was interrupted has none) and, when the config directory
    exists and still lists its scenario, that manifest matches the current
    config and every output exists. Reps of scenarios the config has dropped
    are kept; ``simace ls`` does not show those.
    """
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(
        prog=prog, description="Summarize a folder's complete rep reports and render its validation atlas"
    )
    add_logging_args(parser)
    parser.add_argument("folder", help="Folder under the results root")
    parser.add_argument("--format", choices=("html", "pdf"), default="html", help="Atlas format (default: html)")
    parser.add_argument(
        "--plot-format", choices=("png", "pdf"), default="png", help="Validation plot format (default: png)"
    )
    add_root_args(parser)
    args = parser.parse_args(argv)
    init_logging(args)
    config_dir, layout = resolve_roots(args)

    scenarios: dict[str, dict[str, Any]] = {}
    if (config_dir / "_default.yaml").is_file():
        from simace.cli.run import resolve_all

        scenarios = resolve_all(config_dir)
    elif args.config_dir is not None:
        print(f"simace gather: no config directory at {config_dir} (no _default.yaml)", file=sys.stderr)
        raise SystemExit(2)

    reports: list[str] = []
    for report in sorted((layout.root / args.folder).glob(f"*/rep*/{RepArtifact.REPORT}")):
        if not report.parent.name.removeprefix("rep").isdigit():
            print(f"simace gather: skipping {report.parent} (not a rep directory)", file=sys.stderr)
            continue
        why = _not_complete(report.parent, scenarios, layout)
        if why is None:
            reports.append(str(report))
        else:
            print(f"simace gather: skipping {report.parent} ({why})", file=sys.stderr)
    if not reports:
        print(f"simace gather: no complete reps under {layout.root / args.folder}", file=sys.stderr)
        raise SystemExit(1)

    from simace.analysis.gather import main as gather_reports
    from simace.plotting.plot_validation import main as plot_validation

    summary = layout.folder_summary(args.folder)
    gather_reports(reports, str(summary))
    plot_validation(
        str(summary), layout.folder_plots(args.folder), plot_ext=args.plot_format, atlas_name=f"atlas.{args.format}"
    )


def _not_complete(rep_dir: Path, scenarios: dict[str, dict[str, Any]], layout: Layout) -> str | None:
    """Return why ``rep_dir`` is left out of the summary, or None when it counts."""
    if not (rep_dir / RepArtifact.RUN_MANIFEST).exists():
        return f"no {RepArtifact.RUN_MANIFEST}; the rep did not finish"
    scenario, rep = rep_dir.parent.name, int(rep_dir.name.removeprefix("rep"))
    params = scenarios.get(scenario)
    if params is None or params["folder"] != rep_dir.parent.parent.name:
        return None

    from simace.cli.manifest import RepState
    from simace.cli.run import status_on_disk
    from simace.cli.stages import ResolvedRep

    status = status_on_disk(ResolvedRep(params["folder"], scenario, rep, params), layout)
    return None if status.state is RepState.COMPLETE else f"{status.state}: {status.describe()}"
