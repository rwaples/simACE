"""``simace validate <scenario|folder>``: on-demand checks over a scenario's complete reps.

``simace run`` never runs these checks (:mod:`simace.analysis.validate.on_demand`):
they cost too much for every rep or need outcomes a rep does not store. This
command runs them on finished reps, one ``simace validate --checks ...``
subprocess per rep built from the scenario's config, and each writes the
rep's ``validation.yaml``. Only complete reps are validated; the rest are
skipped with their state. Like ``simace run`` it takes each scenario's lock,
so it never reads a rep a run is rewriting, and a later recompute of a rep
removes its ``validation.yaml``.

``simace validate --pedigree ... --checks ...`` is the per-rep primitive: see
:func:`simace.analysis.validate.runner.cli`.
"""

from __future__ import annotations

__all__ = ["cli"]

import argparse
import shlex
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import TYPE_CHECKING

from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS
from simace.cli.launch import Launcher, child_env, command
from simace.cli.layout import RepArtifact, add_root_args, require_config, resolve_roots
from simace.cli.manifest import RepState
from simace.cli.run import ScenarioBusy, lock_scenario, plan_runs, rep_spec
from simace.cli.stages import validate_argv
from simace.cli.status import status_on_disk
from simace.core.yaml_io import load_yaml

if TYPE_CHECKING:
    from simace.cli.layout import Layout
    from simace.cli.run import ScenarioRun
    from simace.cli.stages import ResolvedRep


def _parse(argv: list[str] | None, prog: str | None) -> argparse.Namespace:
    from simace.core.cli_base import add_version_arg

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run on-demand validation checks on every complete replicate of each scenario or folder",
    )
    add_version_arg(parser, "simace")
    parser.add_argument("targets", nargs="+", metavar="TARGET", help="A scenario name, or a folder name")
    parser.add_argument(
        "--rep", type=rep_spec, nargs="+", default=None, metavar="REP", help="Replicates (default: all; one scenario)"
    )
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Reps validated concurrently (default: 1)")
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=sorted(ON_DEMAND_CHECKS),
        default=sorted(ON_DEMAND_CHECKS),
        help="On-demand check families to run (default: all): "
        + "; ".join(f"{c.name}: {c.description}" for c in ON_DEMAND_CHECKS.values()),
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print the per-rep commands; write nothing")
    add_root_args(parser, logs=True)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.rep is not None:
        args.rep = [r for spec in args.rep for r in spec]
    return args


def _validate_rep(rep: ResolvedRep, layout: Layout, launcher: Launcher, checks: list[str]) -> str:
    """Validate one rep; return its one-line outcome."""
    log_path = rep.log(layout, "validate")
    result = launcher.run(command("validate", validate_argv(rep, layout, checks)), log_path)
    if result.exit_code != 0:
        return f"ERROR ({result.failure}); log: {log_path}"
    summary = load_yaml(rep.path(layout, RepArtifact.VALIDATION))["summary"]
    verdict = "passed" if summary["passed"] else f"FAILED {summary['n_failed']} check(s)"
    return f"{verdict} in {result.wall_s:.1f}s"


def _complete_reps(runs: list[ScenarioRun], layout: Layout) -> list[ResolvedRep]:
    """The requested reps that are complete under the current config; print why each other one is skipped."""
    todo = []
    for run in runs:
        for rep in run.requested:
            status = status_on_disk(rep, layout)
            if status.state is RepState.COMPLETE:
                todo.append(rep)
            else:
                why = status.describe()
                print(f"[{rep.scenario}/rep{rep.rep}] skipped ({status.state.value}{': ' + why if why else ''})")
    return todo


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line entry point for ``simace validate``."""
    if argv and any(a == "--pedigree" or a.startswith("--pedigree=") for a in argv):
        from simace.analysis.validate.runner import cli as rep_cli

        rep_cli(argv, prog)
        return
    args = _parse(argv, prog)
    config_dir, layout = resolve_roots(args)
    require_config(config_dir, "validate")
    runs = plan_runs("validate", config_dir, args.targets, args.rep)
    if args.dry_run:
        for rep in _complete_reps(runs, layout):
            print(shlex.join(command("validate", validate_argv(rep, layout, args.checks))))
        return

    lock = threading.Lock()
    outcomes: dict[str, list[str]] = {}

    def one(rep: ResolvedRep) -> None:
        line = _validate_rep(rep, layout, launcher, args.checks)
        with lock:
            print(f"[{rep.scenario}/rep{rep.rep}] {line}", flush=True)
            outcomes.setdefault(rep.scenario, []).append(line)

    try:
        with ExitStack() as stack:
            # Lock first, then classify, as `simace run` does: a run cannot rewrite a rep in between.
            lock_fds = tuple(lock_scenario(stack, run, layout) for run in runs)
            todo = _complete_reps(runs, layout)
            launcher = Launcher(child_env(args.jobs), None, lock_fds)
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(one, todo))
    except ScenarioBusy as exc:
        print(f"simace validate: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    errors = 0
    for scenario, lines in outcomes.items():
        failed = sum(line.startswith("FAILED") for line in lines)
        errored = sum(line.startswith("ERROR") for line in lines)
        errors += errored
        print(f"[{scenario}] validated {len(lines)} rep(s): {failed} with failing checks, {errored} errors")
    if errors:
        raise SystemExit(1)
