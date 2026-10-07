"""``simace show`` and ``simace ls``: read-only views of config and results."""

from __future__ import annotations

__all__ = ["ls_cli", "show_cli"]

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import yaml

import simace
from simace.cli.layout import RepArtifact, add_root_args, require_config, resolve_roots
from simace.cli.manifest import RepState, source_ref
from simace.cli.stages import ResolvedRep
from simace.cli.status import (
    ScenarioError,
    check_runnable,
    load_scenario,
    rep_ranges,
    resolve_all,
    scenario_plots_status,
    status_on_disk,
)
from simace.core.yaml_io import to_native

if TYPE_CHECKING:
    from simace.cli.layout import Layout
    from simace.cli.manifest import RepStatus


_PEAK_COLUMNS = ("max_rss_mb", "tree_peak_mb")


def _reps(params: dict, scenario: str) -> list[ResolvedRep]:
    return [ResolvedRep(params["folder"], scenario, r, params) for r in range(1, int(params["replicates"]) + 1)]


def _note(status: RepStatus) -> str:
    """Why a rep is in its state: the changed keys with values, the missing outputs, or a different builder."""
    notes = [status.describe()] if status.reasons else []
    if status.simace_version not in (None, simace.__version__):
        notes.append(f"built by simace {status.simace_version}")
    if status.source is not None and status.source != source_ref():
        notes.append(f"built at {status.source}")
    return "; ".join(notes)


def _states(layout: Layout, reps: list[ResolvedRep]) -> str:
    """Summarize a scenario's reps as ``N reps: n complete, n stale (reps 1-3: N: 999 -> 300)``.

    Complete reps built by the running code are counted only; every other
    rep is listed, grouped by its note.
    """
    by_state: dict[RepState, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rep in reps:
        status = status_on_disk(rep, layout)
        by_state[status.state][_note(status)].append(rep.rep)
    parts = []
    for state in RepState:
        groups = by_state.get(state)
        if not groups:
            continue
        count = sum(len(members) for members in groups.values())
        detail = [
            f"{rep_ranges(members)}: {note}" if note else rep_ranges(members)
            for note, members in groups.items()
            if note or state is not RepState.COMPLETE
        ]
        parts.append(f"{count} {state}" + (f" ({'; '.join(detail)})" if detail else ""))
    return f"{len(reps)} reps: " + ", ".join(parts)


def _timing(layout: Layout, reps: list[ResolvedRep]) -> dict[str, Any]:
    """Per stage over the complete reps: median wall time, and each recorded memory peak with the rep that set it.

    ``tree_peak_mb`` is reported for a stage only when some rep recorded it;
    reps built before it existed, or without a delegated cgroup, have none.
    """
    walls: dict[str, list[float]] = defaultdict(list)
    peaks: dict[str, dict[str, tuple[float, int]]] = defaultdict(dict)
    used = 0
    for rep in reps:
        if status_on_disk(rep, layout).state is not RepState.COMPLETE:
            continue
        used += 1
        with open(rep.path(layout, RepArtifact.TIMING), encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                stage = row["stage"]
                walls[stage].append(float(row["wall_s"]))
                for column in _PEAK_COLUMNS:
                    if row.get(column):
                        value = float(row[column])
                        if column not in peaks[stage] or value > peaks[stage][column][0]:
                            peaks[stage][column] = (value, rep.rep)
    stages: dict[str, dict[str, Any]] = {}
    for stage, stage_walls in walls.items():
        stages[stage] = {"wall_s_median": round(statistics.median(stage_walls), 1)}
        for column, (peak, peak_rep) in peaks[stage].items():
            stages[stage][column] = round(peak)
            stages[stage][f"{column.removesuffix('_mb')}_rep"] = peak_rep
    return {"from_complete_reps": used, **stages}


def show_cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Print a scenario's resolved parameters, per-rep seeds, output paths, and past timing."""
    parser = argparse.ArgumentParser(prog=prog, description="Show a scenario's resolved parameters, paths, and timing")
    parser.add_argument("scenario")
    add_root_args(parser)
    args = parser.parse_args(argv)
    config_dir, layout = resolve_roots(args)
    require_config(config_dir, "show")
    try:
        params = load_scenario(config_dir, args.scenario, require_runnable=False)
    except ScenarioError as exc:
        print(f"simace show: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    reps = _reps(params, args.scenario)
    body = {
        "scenario": args.scenario,
        "params": to_native(params),
        "reps": {f"rep{rep.rep}": {"seed": rep.seed, "dir": str(rep.dir(layout))} for rep in reps},
        "plots": {
            "dir": str(layout.scenario_plots(params["folder"], args.scenario)),
            "state": scenario_plots_status(reps, layout).describe(),
        },
        "timing": _timing(layout, reps),
    }
    print(yaml.safe_dump(body, sort_keys=False), end="")


def ls_cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """List scenarios by folder with the state of their replicates.

    A rep built by a different simace version or commit is flagged but not
    stale: ``simace run`` skips it, and ``--force`` recomputes it.
    """
    parser = argparse.ArgumentParser(prog=prog, description="List scenarios and the state of each replicate")
    parser.add_argument("folder", nargs="?", default=None, help="Only this folder")
    add_root_args(parser)
    args = parser.parse_args(argv)
    config_dir, layout = resolve_roots(args)
    require_config(config_dir, "ls")

    scenarios = resolve_all(config_dir)
    for name in sorted(scenarios, key=lambda s: (scenarios[s]["folder"], s)):
        params = scenarios[name]
        if args.folder is not None and params["folder"] != args.folder:
            continue
        try:
            check_runnable(name, params)
        except ScenarioError:
            print(f"{params['folder']}/{name}  gene drop (not run by simace run)")
            continue
        reps = _reps(params, name)
        print(
            f"{params['folder']}/{name}  {_states(layout, reps)}; plots {scenario_plots_status(reps, layout).describe()}"
        )
