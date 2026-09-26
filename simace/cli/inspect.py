"""``simace show`` and ``simace ls``: read-only views of config and results."""

from __future__ import annotations

__all__ = ["ls_cli", "show_cli"]

import argparse
import sys
from pathlib import Path

import yaml

import simace
from simace.cli.layout import Layout
from simace.cli.run import ScenarioError, check_runnable, load_scenario, resolve_all, status_on_disk
from simace.cli.stages import ResolvedRep
from simace.core.yaml_io import to_native


def _roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", type=Path, default=Path("config"), help="Config directory (default: config)")
    parser.add_argument("--results", type=Path, default=Path("results"), help="Results root (default: results)")


def _reps(params: dict, scenario: str) -> list[ResolvedRep]:
    return [ResolvedRep(params["folder"], scenario, r, params) for r in range(1, int(params["replicates"]) + 1)]


def _state(layout: Layout, rep: ResolvedRep) -> str:
    status = status_on_disk(rep, layout)
    notes = list(status.reasons)
    if status.simace_version not in (None, simace.__version__):
        notes.append(f"built by simace {status.simace_version}")
    return f"{status.state} ({', '.join(notes)})" if notes else str(status.state)


def show_cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Print a scenario's resolved parameters, per-rep seeds, and output paths."""
    parser = argparse.ArgumentParser(prog=prog, description="Show a scenario's resolved parameters and paths")
    parser.add_argument("scenario")
    _roots(parser)
    args = parser.parse_args(argv)
    try:
        params = load_scenario(args.config_dir, args.scenario)
    except ScenarioError as exc:
        print(f"simace show: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    layout = Layout(root=args.results)
    reps = {
        f"rep{rep.rep}": {"seed": rep.seed, "dir": str(layout.rep_dir(rep.folder, rep.scenario, rep.rep))}
        for rep in _reps(params, args.scenario)
    }
    body = {
        "scenario": args.scenario,
        "params": to_native(params),
        "reps": reps,
        "plots": str(layout.scenario_plots(params["folder"], args.scenario)),
    }
    print(yaml.safe_dump(body, sort_keys=False), end="")


def ls_cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """List scenarios by folder with each replicate's state.

    A rep built by a different simace version is flagged but not stale:
    ``simace run`` skips it, and ``--force`` recomputes it.
    """
    parser = argparse.ArgumentParser(prog=prog, description="List scenarios and the state of each replicate")
    parser.add_argument("folder", nargs="?", default=None, help="Only this folder")
    _roots(parser)
    args = parser.parse_args(argv)

    layout = Layout(root=args.results)
    scenarios = resolve_all(args.config_dir)
    for name in sorted(scenarios, key=lambda s: (scenarios[s]["folder"], s)):
        params = scenarios[name]
        if args.folder is not None and params["folder"] != args.folder:
            continue
        try:
            check_runnable(name, params)
        except ScenarioError:
            print(f"{params['folder']}/{name}  gene drop (not run by simace run)")
            continue
        states = "  ".join(f"rep{rep.rep} {_state(layout, rep)}" for rep in _reps(params, name))
        print(f"{params['folder']}/{name}  {states}")
