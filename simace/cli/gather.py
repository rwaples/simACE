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

    By default the folder is what ``config/{folder}.yaml`` lists: every rep
    of every runnable scenario there, counting the complete ones and naming
    each other rep and its state, so the summary is of today's configured
    folder. ``--all`` gathers what is on disk instead: every rep directory
    with a ``run.yaml`` and a ``report.yaml``, checked against the config
    only when the config still lists its scenario. That is for archived
    folders the config no longer describes.
    """
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(
        prog=prog, description="Summarize a folder's complete rep reports and render its validation atlas"
    )
    add_logging_args(parser)
    parser.add_argument("folder", help="Folder under the results root")
    parser.add_argument(
        "--format",
        choices=("html", "pdf"),
        default="html",
        help="pdf also writes plots/atlas.pdf beside the always-built atlas.html (default: html)",
    )
    parser.add_argument(
        "--plot-format", choices=("png", "pdf"), default="png", help="Validation plot format (default: png)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Gather every finished rep on disk, not just the configured folder (for archived folders)",
    )
    add_root_args(parser)
    args = parser.parse_args(argv)
    init_logging(args)
    config_dir, layout = resolve_roots(args)

    scenarios: dict[str, dict[str, Any]] = {}
    if (config_dir / "_default.yaml").is_file():
        from simace.cli.status import resolve_all

        scenarios = resolve_all(config_dir)
    elif args.config_dir is not None or not args.all:
        print(
            f"simace gather: no config directory at {config_dir} (no _default.yaml); --all gathers by disk",
            file=sys.stderr,
        )
        raise SystemExit(2)

    reports = (
        _reports_on_disk(args.folder, scenarios, layout)
        if args.all
        else _configured_reports(args.folder, scenarios, layout)
    )
    if not reports:
        print(f"simace gather: no complete reps under {layout.root / args.folder}", file=sys.stderr)
        raise SystemExit(1)

    from simace.analysis.gather import main as gather_reports
    from simace.plotting.plot_validation import atlas_names
    from simace.plotting.plot_validation import main as plot_validation

    summary = layout.folder_summary(args.folder)
    gather_reports(reports, str(summary))
    plot_validation(
        str(summary), layout.folder_plots(args.folder), plot_ext=args.plot_format, atlas_names=atlas_names(args.format)
    )


def _configured_reports(folder: str, scenarios: dict[str, dict[str, Any]], layout: Layout) -> list[str]:
    """Return the report paths of the complete reps of ``folder``'s configured scenarios, naming every other rep."""
    from simace.cli.manifest import RepState
    from simace.cli.stages import ResolvedRep
    from simace.cli.status import ScenarioError, check_runnable, status_on_disk

    configured = {name: params for name, params in scenarios.items() if params["folder"] == folder}
    if not configured:
        print(
            f"simace gather: no configured scenario has folder {folder!r}; --all gathers what is on disk",
            file=sys.stderr,
        )
        raise SystemExit(2)
    reports: list[str] = []
    for name in sorted(configured):
        params = configured[name]
        try:
            check_runnable(name, params)
        except ScenarioError:
            print(f"simace gather: skipping {folder}/{name} (gene drop; not run by simace run)", file=sys.stderr)
            continue
        reps = [ResolvedRep(folder, name, r, params) for r in range(1, int(params["replicates"]) + 1)]
        included = 0
        for rep in reps:
            status = status_on_disk(rep, layout)
            if status.state is RepState.COMPLETE:
                reports.append(str(rep.path(layout, RepArtifact.REPORT)))
                included += 1
            else:
                why = f"{status.state}: {status.describe()}" if status.reasons else str(status.state)
                print(f"simace gather: skipping {rep.dir(layout)} ({why})", file=sys.stderr)
        print(f"simace gather: {folder}/{name}: {included} of {len(reps)} reps", file=sys.stderr)
        for extra in sorted(layout.scenario_dir(folder, name).glob("rep*")):
            if extra.is_dir() and extra.name.removeprefix("rep").isdigit() and int(extra.name[3:]) > len(reps):
                print(
                    f"simace gather: skipping {extra} (beyond replicates: {len(reps)}; --all includes it)",
                    file=sys.stderr,
                )
    folder_dir = layout.root / folder
    if folder_dir.is_dir():
        for entry in sorted(folder_dir.iterdir()):
            if entry.is_dir() and entry.name not in configured and entry != layout.folder_plots(folder):
                print(
                    f"simace gather: skipping {entry} (not in config/{folder}.yaml; --all includes it)", file=sys.stderr
                )
    return reports


def _reports_on_disk(folder: str, scenarios: dict[str, dict[str, Any]], layout: Layout) -> list[str]:
    """Return the report paths of every finished rep directory under ``folder``, whatever the config lists."""
    reports: list[str] = []
    for report in sorted((layout.root / folder).glob(f"*/rep*/{RepArtifact.REPORT}")):
        if not report.parent.name.removeprefix("rep").isdigit():
            print(f"simace gather: skipping {report.parent} (not a rep directory)", file=sys.stderr)
            continue
        why = _not_complete(report.parent, scenarios, layout)
        if why is None:
            reports.append(str(report))
        else:
            print(f"simace gather: skipping {report.parent} ({why})", file=sys.stderr)
    return reports


def _not_complete(rep_dir: Path, scenarios: dict[str, dict[str, Any]], layout: Layout) -> str | None:
    """Return why ``rep_dir`` is left out of the summary, or None when it counts."""
    if not (rep_dir / RepArtifact.RUN_MANIFEST).exists():
        return f"no {RepArtifact.RUN_MANIFEST}; the rep did not finish"
    scenario, rep = rep_dir.parent.name, int(rep_dir.name.removeprefix("rep"))
    params = scenarios.get(scenario)
    if params is None or params["folder"] != rep_dir.parent.parent.name:
        return None

    from simace.cli.manifest import RepState
    from simace.cli.stages import ResolvedRep
    from simace.cli.status import status_on_disk

    status = status_on_disk(ResolvedRep(params["folder"], scenario, rep, params), layout)
    return None if status.state is RepState.COMPLETE else f"{status.state}: {status.describe()}"
