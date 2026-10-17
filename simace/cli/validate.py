"""``simace validate <scenario|folder>``: on-demand checks over a scenario's complete reps, and their verdicts.

``simace run`` never runs these checks (:mod:`simace.analysis.validate.on_demand`).
This command runs them on finished reps, one ``simace validate --checks ...``
subprocess per rep built from the scenario's config, and each writes the
rep's ``validation.yaml``. Only complete reps are validated; the rest are
skipped with their state. A rep whose ``validation.yaml`` holds the requested
checks and was written by this build (the simace version and the ``git
describe`` of its checkout, as in ``run.yaml``) is current and is not
validated again; ``--force`` validates it anyway. Builds compare without
``git describe``'s ``-dirty`` mark, so editing a config or a doc keeps reps
current, and after changing code without committing, pass ``--force``. The
build is read once per invocation and passed to every rep.

Then each scenario gets its verdict, ``validation_summary.yaml``, over every
complete rep and every registered calibrated test, and a folder target gets
the folder's verdict over its scenarios
(:mod:`simace.analysis.validate.verdict`). A verdict needs every complete rep
validated with every test by one build; otherwise none is written, an old one
is removed, and the command says why; so does a rep that fails validation.
Validating a scenario without its folder removes the folder's verdict unless
it still records that scenario's reps and build.

Like ``simace run`` it takes each scenario's lock, so it never reads a rep a
run is rewriting; a later recompute of a rep removes its ``validation.yaml``
and both verdicts.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeGuard

import simace
from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS, SCHEMA, load_validation
from simace.analysis.validate.schema_file import one_warning_for_refused
from simace.analysis.validate.verdict import (
    decide,
    folder_family,
    load_verdict,
    overlapping_seed_ranges,
    scenario_family,
    write_verdict,
)
from simace.cli.launch import Launcher, child_env, command
from simace.cli.layout import RepArtifact, add_root_args, require_config, resolve_roots
from simace.cli.manifest import RepState, source_ref
from simace.cli.run import ScenarioBusy, lock_scenario, plan_runs, rep_spec
from simace.cli.stages import validate_argv
from simace.cli.status import rep_ranges, rep_spans, status_on_disk

if TYPE_CHECKING:
    from pathlib import Path

    from simace.cli.layout import Layout
    from simace.cli.run import ScenarioRun
    from simace.cli.stages import ResolvedRep


def _parse(argv: list[str] | None, prog: str | None) -> argparse.Namespace:
    from simace.core.cli_base import add_version_arg

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run on-demand validation checks on every complete replicate of each scenario or folder, "
        "then decide each scenario's and folder's verdict",
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
    parser.add_argument(
        "--force", action="store_true", help="Validate reps whose validation.yaml is already current, too"
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print the per-rep commands; write nothing")
    add_root_args(parser, logs=True)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.rep is not None:
        args.rep = [r for spec in args.rep for r in spec]
    return args


def _validate_rep(rep: ResolvedRep, layout: Layout, launcher: Launcher, checks: list[str], source: str | None) -> str:
    """Validate one rep; return its one-line outcome."""
    log_path = rep.log(layout, "validate")
    result = launcher.run(command("validate", validate_argv(rep, layout, checks, source)), log_path)
    if result.exit_code != 0:
        return f"ERROR ({result.failure}); log: {log_path}"
    data = load_validation(rep.path(layout, RepArtifact.VALIDATION))
    if data is None:
        return f"ERROR (no validation.yaml with schema {SCHEMA}); log: {log_path}"
    sections = "; ".join(_describe(name, section["summary"]) for name, section in data["checks"].items())
    return f"validated in {result.wall_s:.1f}s: {sections}"


def _describe(name: str, summary: dict) -> str:
    """One check's per-rep line: what it scored, never a pass (the verdict is across reps; ADR 0028)."""
    line = f"{name} {summary['n_scored']} cell(s) scored"
    if summary.get("max_abs_z") is not None:
        line += f", max |z| {summary['max_abs_z']:.2f} (per-rep bound {summary['z_bound']:.2f})"
    return line


def _complete_reps(runs: list[ScenarioRun], layout: Layout) -> list[ResolvedRep]:
    """The requested reps that are complete under the current config; print why each other one is skipped."""
    complete = []
    for run in runs:
        for rep in run.requested:
            status = status_on_disk(rep, layout)
            if status.state is RepState.COMPLETE:
                complete.append(rep)
            else:
                why = status.describe()
                print(f"[{rep.scenario}/rep{rep.rep}] skipped ({status.state.value}{': ' + why if why else ''})")
    return complete


def _stale(
    reps: list[ResolvedRep], layout: Layout, checks: list[str], source: str | None, force: bool
) -> list[ResolvedRep]:
    """The reps to validate: all of them with ``force``, else those whose ``validation.yaml`` is not current."""
    if force:
        return reps
    running = _built_by({"simace_version": simace.__version__, "source": source})

    def current(rep: ResolvedRep) -> bool:
        data = load_validation(rep.path(layout, RepArtifact.VALIDATION))
        return _holds(data, checks) and _built_by(data) == running

    with one_warning_for_refused():
        return [rep for rep in reps if not current(rep)]


def _holds(data: dict[str, Any] | None, checks: list[str]) -> TypeGuard[dict[str, Any]]:
    """Whether a rep's ``validation.yaml`` is schema-valid and holds every named check."""
    return data is not None and set(checks) <= set(data["checks"])


def _built_by(data: dict[str, Any]) -> str:
    """The build that wrote a file: its simace version and ``git describe`` without the ``-dirty`` mark."""
    source = data.get("source")
    return f"{data['simace_version']} at {source.removesuffix('-dirty') if source else None}"


@dataclass(frozen=True)
class _Validated:
    """A scenario's complete reps, all validated with every registered test by one build: what its verdicts read."""

    reps: list[int]
    #: ``simace_version`` and ``source`` of the build that validated them.
    built_by: dict[str, Any]
    #: Per test, each rep's section of ``validation.yaml``, in rep order.
    sections: dict[str, list[dict[str, Any]]]


def _validated(run: ScenarioRun, layout: Layout) -> _Validated | str:
    """Every complete rep's sections of ``run``'s scenario for every registered test, or why there is no verdict."""
    checks = list(ON_DEMAND_CHECKS)
    complete = [rep for rep in run.all_reps if status_on_disk(rep, layout).state is RepState.COMPLETE]
    if not complete:
        return "no complete reps"
    with one_warning_for_refused():
        files = {rep.rep: load_validation(rep.path(layout, RepArtifact.VALIDATION)) for rep in complete}
    held = {r: data for r, data in files.items() if _holds(data, checks)}
    if missing := sorted(set(files) - set(held)):
        return f"{rep_ranges(missing)} not validated with {', '.join(checks)}"
    by_build: dict[str, list[int]] = {}
    for r, data in held.items():
        by_build.setdefault(_built_by(data), []).append(r)
    if len(by_build) > 1:
        builds = "; ".join(f"{b} ({rep_ranges(reps)})" for b, reps in sorted(by_build.items()))
        return f"reps validated by more than one build: {builds}"
    reps = sorted(held)
    first = held[reps[0]]
    sections = {name: [held[r]["checks"][name] for r in reps] for name in checks}
    return _Validated(reps, {"simace_version": first["simace_version"], "source": first.get("source")}, sections)


def _seed_range(run: ScenarioRun) -> tuple[int, int]:
    """The scenario's rep seeds, ``[seed, seed + replicates − 1]``."""
    return run.all_reps[0].seed, run.all_reps[-1].seed


def _verdict_line(decided: dict[str, Any], n_reps: int, path: Path) -> str:
    if decided["n_tests"]:
        tests = f"{decided['n_tests']} test(s) at alpha {decided['alpha_each']:.2e} each"
    else:
        tests = "no test decided"
    return f"verdict {decided['verdict']}: {tests} over {n_reps} rep(s); {path}"


def _scenario_verdict(run: ScenarioRun, layout: Layout, failed: list[int]) -> tuple[_Validated | None, str]:
    """Write the scenario's ``validation_summary.yaml``, or remove it when there is no verdict; return the line.

    ``failed`` are the reps whose validation failed in this invocation: their
    ``validation.yaml``, if any, is from an earlier one.
    """
    path = layout.scenario_verdict(run.folder, run.scenario)
    validated = f"{rep_ranges(failed)} failed validation" if failed else _validated(run, layout)
    if isinstance(validated, str):
        path.unlink(missing_ok=True)
        return None, f"no verdict: {validated}"
    decided = decide(scenario_family(validated.sections, ON_DEMAND_CHECKS))
    header = {
        **validated.built_by,
        "family": "scenario",
        "scenario": run.scenario,
        "seed": _seed_range(run)[0],
        "reps": rep_spans(validated.reps),
    }
    write_verdict(path, header, decided)
    return validated, _verdict_line(decided, len(validated.reps), path)


def _folder_verdict(
    folder: str, runs: list[ScenarioRun], validated: dict[str, _Validated | None], layout: Layout
) -> str:
    """Write the folder's ``validation_summary.yaml``, or remove it when there is no verdict; return the line."""
    path = layout.folder_verdict(folder)
    members = [run for run in runs if run.folder == folder]
    ready = {run.scenario: v for run in members if (v := validated[run.scenario]) is not None}
    builds = sorted({_built_by(v.built_by) for v in ready.values()})
    if not members:
        reason = "no scenario to validate"
    elif missing := [run.scenario for run in members if run.scenario not in ready]:
        reason = f"no verdict for {', '.join(missing)}"
    elif len(builds) > 1:
        reason = f"scenarios validated by more than one build: {'; '.join(builds)}"
    else:
        reason = None
    if reason is not None:
        path.unlink(missing_ok=True)
        return f"no verdict: {reason}"
    overlaps = overlapping_seed_ranges({run.scenario: _seed_range(run) for run in members})
    refused = "rep seed ranges overlap: " + ", ".join(f"{a} and {b}" for a, b in overlaps) if overlaps else None
    decided = decide(folder_family({s: v.sections for s, v in ready.items()}, ON_DEMAND_CHECKS, refused))
    header = {
        **next(iter(ready.values())).built_by,
        "family": "folder",
        "folder": folder,
        "reps_by_scenario": {run.scenario: _covers(run, ready[run.scenario]) for run in members},
    }
    write_verdict(path, header, decided)
    line = _verdict_line(decided, sum(len(v.reps) for v in ready.values()), path)
    return line + (f"; pooling refused, {refused}" if refused else "")


def _covers(run: ScenarioRun, validated: _Validated) -> dict[str, Any]:
    """What a folder verdict records of one scenario: its seed and the reps it read."""
    return {"seed": _seed_range(run)[0], "reps": rep_spans(validated.reps)}


def _drop_stale_folder_verdict(
    folder: str, runs: list[ScenarioRun], validated: dict[str, _Validated | None], layout: Layout
) -> str | None:
    """Remove a folder verdict that no longer records its validated scenarios' reps and build; return the line."""
    path = layout.folder_verdict(folder)
    data = load_verdict(path)
    if data is not None and all(
        (v := validated[run.scenario]) is not None
        and data["reps_by_scenario"].get(run.scenario) == _covers(run, v)
        and _built_by(data) == _built_by(v.built_by)
        for run in runs
        if run.folder == folder
    ):
        return None
    if not path.exists():
        return None
    path.unlink(missing_ok=True)
    return f"folder verdict removed; rerun `simace validate {folder}`"


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
    folders = list(dict.fromkeys(t for t in args.targets if t not in {run.scenario for run in runs}))
    source = source_ref()
    if args.dry_run:
        for rep in _stale(_complete_reps(runs, layout), layout, args.checks, source, args.force):
            print(shlex.join(command("validate", validate_argv(rep, layout, args.checks, source))))
        return

    lock = threading.Lock()
    outcomes: dict[str, dict[int, str]] = {run.scenario: {} for run in runs}

    def one(rep: ResolvedRep) -> None:
        line = _validate_rep(rep, layout, launcher, args.checks, source)
        with lock:
            print(f"[{rep.scenario}/rep{rep.rep}] {line}", flush=True)
            outcomes[rep.scenario][rep.rep] = line

    try:
        with ExitStack() as stack:
            # Lock first, then classify, as `simace run` does: a run cannot rewrite a rep in between.
            lock_fds = tuple(lock_scenario(stack, run, layout) for run in runs)
            complete = _complete_reps(runs, layout)
            todo = _stale(complete, layout, args.checks, source, args.force)
            launcher = Launcher(child_env(args.jobs), None, lock_fds)
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(one, todo))

            errors = 0
            validated: dict[str, _Validated | None] = {}
            for run in runs:
                lines = outcomes[run.scenario]
                failed = sorted(r for r, line in lines.items() if line.startswith("ERROR"))
                errors += len(failed)
                current = sum(rep.scenario == run.scenario for rep in complete) - len(lines)
                print(
                    f"[{run.scenario}] validated {len(lines) - len(failed)} rep(s), {current} already current, "
                    f"{len(failed)} errors"
                )
                validated[run.scenario], line = _scenario_verdict(run, layout, failed)
                print(f"[{run.scenario}] {line}")
            for folder in folders:
                print(f"[{folder}] folder {_folder_verdict(folder, runs, validated, layout)}")
            for folder in dict.fromkeys(run.folder for run in runs if run.folder not in folders):
                if line := _drop_stale_folder_verdict(folder, runs, validated, layout):
                    print(f"[{folder}] {line}")
    except ScenarioBusy as exc:
        print(f"simace validate: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if errors:
        raise SystemExit(1)
