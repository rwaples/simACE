"""``simace run <scenario>``: compute every rep of a scenario, then its plots.

Resume granularity is the rep, or with ``--until`` the stage. A rep whose
``run.yaml`` matches the current parameters is skipped; a rep with no
``run.yaml`` is recomputed from scratch; a rep whose ``run.yaml`` differs is
refused unless ``--force``. A rep built through fewer stages than asked for
resumes at its first missing stage.
Every stage runs as its own ``simace <stage>`` subprocess
(:mod:`simace.cli.launch`), so its wall time and memory peaks land in the
rep's ``timing.tsv``; what is already on disk comes from
:mod:`simace.cli.status`. Plots and the atlas are rebuilt on every run.
"""

from __future__ import annotations

__all__ = [
    "ScenarioRun",
    "cli",
    "expand_targets",
    "rep_spec",
]

import argparse
import fcntl
import os
import shlex
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from simace.cli.cgroups import CgroupRoot, CgroupUnavailable
from simace.cli.launch import Launcher, append_timing, child_env, command, parse_size, start_timing
from simace.cli.layout import RepArtifact, add_root_args, require_config, resolve_roots
from simace.cli.manifest import RepState, write_manifest, write_plots_manifest
from simace.cli.stages import (
    PARAMS_YAML_KEYS,
    REP_OUTPUTS,
    RETIRED_OUTPUTS,
    STAGES,
    ResolvedRep,
    atlas_argv,
    plot_argv,
)
from simace.cli.status import (
    ScenarioError,
    built_stages,
    check_runnable,
    expected_manifest,
    rep_outputs,
    rep_ranges,
    resolve_all,
    scenario_plots_status,
    stage_names,
    status_on_disk,
)
from simace.core.publish import TMP_SUFFIX, publish

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from simace.cli.layout import Layout


class ScenarioBusy(Exception):
    """Another ``simace run`` holds the scenario's lock; the message is its pid."""


@contextmanager
def _scenario_lock(path: Path) -> Iterator[int]:
    """Hold an exclusive ``flock`` on ``path`` for the block, recording our pid in it.

    Stage children inherit the descriptor, so the lock stays held if the
    orchestrator is killed while a stage is still writing outputs.

    Raises:
        ScenarioBusy: another process holds the lock; the message is its pid.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            raise ScenarioBusy(fh.read().strip() or "unknown") from None
        fh.truncate(0)
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        yield fh.fileno()


def _write_params(rep: ResolvedRep, layout: Layout) -> None:
    from simace.core.yaml_io import dump_yaml
    from simace.simulation.emit_params import emit_params

    params = emit_params(seed=rep.seed, rep=rep.rep, **{key: rep.params[key] for key in PARAMS_YAML_KEYS})
    with publish(rep.path(layout, RepArtifact.PARAMS)) as (tmp,):
        dump_yaml(params, tmp, sort_keys=True)


class _Console:
    """Serializes progress lines from concurrent reps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def say(self, tag: str, message: str) -> None:
        with self._lock:
            print(f"[{tag}] {message}", flush=True)


def _recompute(
    rep: ResolvedRep,
    layout: Layout,
    launcher: Launcher,
    console: _Console,
    first: int = 0,
    until: int = len(STAGES),
) -> str | None:
    """Compute stages ``first`` up to ``until`` of one rep. Return None when they all exit 0, else why not.

    From the first stage, every file a previous run of this rep wrote is
    removed first, manifest first, so a failure partway leaves no output
    from the old parameters beside outputs from the new ones. Files only an
    earlier results layout wrote go too. A resume (``first > 0``) keeps the
    earlier stages' outputs, which the caller has checked against the rep's
    ``run.yaml``, removes that manifest and the later stages' outputs, and
    rewrites ``params.yaml`` from the current parameters.
    """
    tag = f"{rep.scenario}/rep{rep.rep}"
    rep_dir = rep.dir(layout)
    if first == 0:
        for artifact in REP_OUTPUTS:
            rep.path(layout, artifact).unlink(missing_ok=True)
        for retired in RETIRED_OUTPUTS:
            (rep_dir / retired).unlink(missing_ok=True)
    else:
        rep.path(layout, RepArtifact.RUN_MANIFEST).unlink()
        for stage in STAGES[first:]:
            for artifact in stage.outputs:
                rep.path(layout, artifact).unlink(missing_ok=True)
    if rep_dir.exists():
        for stale in rep_dir.glob(f"*{TMP_SUFFIX}"):
            stale.unlink()

    _write_params(rep, layout)
    timing = rep.path(layout, RepArtifact.TIMING)
    if first == 0:
        start_timing(timing)
    for stage in STAGES[first:until]:
        console.say(tag, f"{stage.name} started")
        log_path = rep.log(layout, stage.name)
        result = launcher.run(command(stage.name, stage.argv(rep, layout)), log_path)
        append_timing(timing, stage.name, result)
        if result.exit_code != 0:
            failure = f"{stage.name} FAILED ({result.failure}); log: {log_path}"
            console.say(tag, failure)
            return failure
        console.say(tag, f"{stage.name} finished in {result.wall_s:.1f}s, {result.memory}")

    write_manifest(
        rep.path(layout, RepArtifact.RUN_MANIFEST),
        expected_manifest(rep, STAGES[:until]),
        rep_outputs(rep, layout, STAGES[:until]),
    )
    return None


@dataclass(frozen=True)
class _ScenarioStage:
    """One scenario-level stage: its label in ``timing.tsv`` and the log name, the subcommand, and its argv."""

    label: str
    subcommand: str
    argv: list[str]
    log_path: Path

    @property
    def command(self) -> list[str]:
        return command(self.subcommand, self.argv)


def _scenario_stages(reps: list[ResolvedRep], layout: Layout, atlas_format: str) -> list[_ScenarioStage]:
    """The plot and HTML atlas stages, plus the PDF atlas when asked for (ADR 0010: HTML is always built)."""
    folder, scenario = reps[0].folder, reps[0].scenario
    stages = [
        _ScenarioStage("plot", "plot", plot_argv(reps, layout), layout.scenario_log(folder, scenario, "plot")),
        _ScenarioStage(
            "atlas", "atlas", atlas_argv(reps, layout, "html"), layout.scenario_log(folder, scenario, "atlas")
        ),
    ]
    if atlas_format == "pdf":
        stages.append(
            _ScenarioStage(
                "atlas-pdf",
                "atlas",
                atlas_argv(reps, layout, "pdf"),
                layout.scenario_log(folder, scenario, "atlas-pdf"),
            )
        )
    return stages


def rep_spec(text: str) -> list[int]:
    """Parse one ``--rep`` value: ``7`` or an inclusive range ``3-10``."""
    lo, sep, hi = text.partition("-")
    try:
        first = int(lo)
        last = int(hi) if sep else first
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a rep or range: {text!r} (use e.g. 7 or 3-10)") from None
    if last < first:
        raise argparse.ArgumentTypeError(f"empty range: {text!r}")
    return list(range(first, last + 1))


def _parse(argv: list[str] | None, prog: str | None) -> argparse.Namespace:
    from simace.core.cli_base import add_version_arg

    parser = argparse.ArgumentParser(
        prog=prog, description="Run every replicate of each scenario or folder, then their plots"
    )
    add_version_arg(parser, "simace")
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="TARGET",
        help="A scenario name, or a folder name for every scenario in config/{folder}.yaml",
    )
    parser.add_argument(
        "--rep",
        type=rep_spec,
        nargs="+",
        default=None,
        metavar="REP",
        help="Replicates to compute, as numbers or ranges like 3-10 (default: all; one scenario only)",
    )
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Reps computed concurrently (default: 1)")
    parser.add_argument("--force", action="store_true", help="Recompute requested reps even when complete or stale")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print what would run; write nothing")
    parser.add_argument("--fail-fast", action="store_true", help="Cancel pending reps after the first failure")
    parser.add_argument("--no-plots", action="store_true", help="Skip the scenario plots and atlas")
    parser.add_argument(
        "--until",
        choices=stage_names(),
        default=STAGES[-1].name,
        help="Last stage to compute (default: %(default)s). A rep built through an earlier stage is partial, "
        "a later run resumes it, and plots are skipped",
    )
    parser.add_argument(
        "--max-memory",
        type=parse_size,
        default=None,
        metavar="SIZE",
        help="Kill any stage whose memory goes over SIZE (e.g. 8G); applies to each stage, not the whole run. "
        "In a delegated cgroup this is the kernel limit memory.max on the stage's process tree, with no swap; "
        "otherwise the tree's resident memory is polled from /proc. Linux only",
    )
    parser.add_argument(
        "--format",
        choices=("html", "pdf"),
        default="html",
        help="pdf also writes plots/atlas.pdf beside the always-built atlas.html (default: html)",
    )
    add_root_args(parser, logs=True)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.max_memory is not None and sys.platform != "linux":
        parser.error("--max-memory needs Linux: it is enforced by a cgroup or by polling /proc")
    if args.rep is not None:
        args.rep = [r for spec in args.rep for r in spec]
    args.until = stage_names().index(args.until) + 1
    return args


@dataclass(frozen=True)
class ScenarioRun:
    """One scenario of a run: every rep it has, and the reps this invocation asked for."""

    scenario: str
    folder: str
    all_reps: list[ResolvedRep]
    requested: list[ResolvedRep]


def expand_targets(targets: list[str], scenarios: dict[str, dict[str, Any]]) -> list[str]:
    """Return the scenario names ``targets`` name, in order, once each.

    A target is a scenario name first, else a folder name (every runnable
    scenario whose ``folder`` it is; gene-drop scenarios are left out).

    Raises:
        ScenarioError: a target is neither a scenario nor a folder, or names
            a gene-drop scenario directly.
    """
    names: list[str] = []
    for target in targets:
        if target in scenarios:
            check_runnable(target, scenarios[target])
            names.append(target)
            continue
        in_folder = [name for name, params in scenarios.items() if params["folder"] == target]
        if not in_folder:
            known = "\n  ".join(sorted(scenarios))
            folders = ", ".join(sorted({params["folder"] for params in scenarios.values()}))
            raise ScenarioError(f"unknown target {target!r}; folders: {folders}\nscenarios:\n  {known}")
        names += [name for name in in_folder if _runnable(name, scenarios[name])]
    return list(dict.fromkeys(names))


def _runnable(name: str, params: dict[str, Any]) -> bool:
    try:
        check_runnable(name, params)
    except ScenarioError:
        return False
    return True


def _plan(name: str, params: dict[str, Any], reps: list[int] | None) -> ScenarioRun:
    n_reps = int(params["replicates"])
    if reps is not None:
        bad = sorted({r for r in reps if not 1 <= r <= n_reps} | {r for r in reps if reps.count(r) > 1})
        if bad:
            raise ScenarioError(f"--rep values must be distinct and in 1..{n_reps}; got {bad}")
    all_reps = [ResolvedRep(params["folder"], name, r, params) for r in range(1, n_reps + 1)]
    return ScenarioRun(name, params["folder"], all_reps, all_reps if reps is None else [all_reps[r - 1] for r in reps])


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line entry point for ``simace run``."""
    args = _parse(argv, prog)
    config_dir, layout = resolve_roots(args)
    require_config(config_dir, "run")
    try:
        scenarios = resolve_all(config_dir)
        names = expand_targets(args.targets, scenarios)
        if args.rep is not None and len(names) != 1:
            raise ScenarioError(f"--rep needs exactly one scenario; the targets name {len(names)}")
        runs = [_plan(name, scenarios[name], args.rep) for name in names]
    except ScenarioError as exc:
        print(f"simace run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if not runs:
        print(f"simace run: nothing to run; every scenario in {args.targets} uses gene drop", file=sys.stderr)
        raise SystemExit(2)

    try:
        with ExitStack() as stack:
            lock_fds: tuple[int, ...] = ()
            cgroups = None
            if not args.dry_run:
                lock_fds = tuple(_lock_scenario(stack, run, layout) for run in runs)
                cgroups = _open_cgroups(stack, args.max_memory)
            _run_reps(args, layout, runs, lock_fds, cgroups)
    except ScenarioBusy as exc:
        print(f"simace run: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _lock_scenario(locks: ExitStack, run: ScenarioRun, layout: Layout) -> int:
    lock_path = layout.scenario_lock(run.folder, run.scenario)
    try:
        return locks.enter_context(_scenario_lock(lock_path))
    except ScenarioBusy as exc:
        raise ScenarioBusy(
            f"{run.scenario} is locked ({lock_path}) by simace run pid {exc}, or by a stage it started"
        ) from None


def _open_cgroups(stack: ExitStack, max_memory: int | None) -> CgroupRoot | None:
    try:
        return stack.enter_context(CgroupRoot.open())
    except CgroupUnavailable as exc:
        fallback = "; --max-memory is enforced by polling /proc" if max_memory is not None else ""
        print(f"simace run: no delegated cgroup ({exc}); tree_peak_mb not recorded{fallback}", file=sys.stderr)
        return None


@dataclass
class _Outcome:
    """What happened to one scenario's requested reps."""

    #: Each rep to compute, with the index of its first stage to run (0 unless resuming a partial rep).
    to_compute: list[tuple[ResolvedRep, int]] = field(default_factory=list)
    skipped: int = 0
    refused: dict[int, str] = field(default_factory=dict)
    failed: dict[int, str] = field(default_factory=dict)


def _classify(run: ScenarioRun, layout: Layout, console: _Console, *, force: bool, until: int) -> _Outcome:
    outcome = _Outcome()
    for rep in run.requested:
        status = status_on_disk(rep, layout, until)
        tag = f"{run.scenario}/rep{rep.rep}"
        if force or status.state is RepState.ABSENT:
            outcome.to_compute.append((rep, 0))
        elif status.state is RepState.INCOMPLETE:
            console.say(tag, f"recompute: {', '.join(status.reasons)}")
            outcome.to_compute.append((rep, 0))
        elif status.state is RepState.PARTIAL:
            built = built_stages(rep, layout)
            console.say(tag, f"resume: {status.describe()}")
            outcome.to_compute.append((rep, len(built)))
        elif status.state is RepState.COMPLETE:
            console.say(tag, "skip (run.yaml matches)")
            outcome.skipped += 1
        else:
            outcome.refused[rep.rep] = f"run.yaml differs in {status.describe()}"
            console.say(tag, f"refused: {outcome.refused[rep.rep]}; --force recomputes it")
    return outcome


def _run_reps(
    args: argparse.Namespace,
    layout: Layout,
    runs: list[ScenarioRun],
    lock_fds: tuple[int, ...],
    cgroups: CgroupRoot | None,
) -> None:
    console = _Console()
    outcomes = {run.scenario: _classify(run, layout, console, force=args.force, until=args.until) for run in runs}
    full = args.until == len(STAGES)
    refused = any(outcome.refused for outcome in outcomes.values())

    if args.dry_run:
        for run in runs:
            outcome = outcomes[run.scenario]
            selected = {rep.rep for rep in run.requested}
            others_complete = all(
                status_on_disk(rep, layout).state is RepState.COMPLETE
                for rep in run.all_reps
                if rep.rep not in selected
            )
            plots = full and not args.no_plots and not outcome.refused and others_complete
            _print_plan(outcome.to_compute, run.all_reps, layout, args.format, args.until, include_plots=plots)
        raise SystemExit(1 if refused else 0)

    failed = _compute(
        [item for run in runs for item in outcomes[run.scenario].to_compute],
        layout,
        Launcher(child_env(args.jobs), args.max_memory, lock_fds, cgroups),
        console,
        jobs=args.jobs,
        fail_fast=args.fail_fast,
        until=args.until,
    )
    for (scenario, rep), why in failed.items():
        outcomes[scenario].failed[rep] = why
    for run in runs:
        _summarize(run.scenario, console, outcomes[run.scenario])
    if len(runs) > 1:
        console.say("total", f"{len(runs)} scenarios: {_counts(outcomes.values())}")

    plotted = True
    if not full and not args.no_plots:
        for run in runs:
            console.say(run.scenario, f"plots skipped: --until {STAGES[args.until - 1].name}")
    elif not args.no_plots:
        launcher = Launcher(child_env(1), args.max_memory, lock_fds, cgroups)
        for run in runs:
            incomplete = [rep.rep for rep in run.all_reps if status_on_disk(rep, layout).state is not RepState.COMPLETE]
            if incomplete:
                console.say(run.scenario, f"plots skipped: {rep_ranges(incomplete)} not complete")
            elif not _build_plots(run.all_reps, layout, args.format, launcher, console):
                plotted = False
    for run in runs:
        console.say(run.scenario, f"plots: {scenario_plots_status(run.all_reps, layout).describe()}")
    if refused or failed or not plotted:
        raise SystemExit(1)


def _compute(
    reps: list[tuple[ResolvedRep, int]],
    layout: Layout,
    launcher: Launcher,
    console: _Console,
    *,
    jobs: int,
    fail_fast: bool,
    until: int,
) -> dict[tuple[str, int], str]:
    """Compute each ``(rep, first stage)`` up to ``until``; return why each rep that failed or was cancelled did not finish."""
    cancelled = threading.Event()

    def one(item: tuple[ResolvedRep, int]) -> str | None:
        rep, first = item
        if cancelled.is_set():
            console.say(f"{rep.scenario}/rep{rep.rep}", "cancelled")
            return "cancelled by --fail-fast"
        failure = _recompute(rep, layout, launcher, console, first, until)
        if failure and fail_fast:
            cancelled.set()
        return failure

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        outcomes = list(pool.map(one, reps))
    return {(rep.scenario, rep.rep): failure for (rep, _), failure in zip(reps, outcomes, strict=True) if failure}


def _counts(outcomes: Iterable[_Outcome]) -> str:
    skipped = computed = failed = refused = 0
    for outcome in outcomes:
        skipped += outcome.skipped
        computed += len(outcome.to_compute) - len(outcome.failed)
        failed += len(outcome.failed)
        refused += len(outcome.refused)
    counts = [(skipped, "skipped"), (computed, "computed"), (failed, "failed"), (refused, "refused")]
    return ", ".join(f"{n} {what}" for n, what in counts if n)


def _summarize(scenario: str, console: _Console, outcome: _Outcome) -> None:
    """Print one line of counts, then one line per rep that failed or was refused."""
    console.say(scenario, f"summary: {_counts([outcome])}")
    for rep, why in sorted(outcome.failed.items()):
        console.say(scenario, f"  rep{rep} failed: {why}")
    for rep, why in sorted(outcome.refused.items()):
        console.say(scenario, f"  rep{rep} refused: {why}")


def _build_plots(
    reps: list[ResolvedRep], layout: Layout, atlas_format: str, launcher: Launcher, console: _Console
) -> bool:
    folder, scenario = reps[0].folder, reps[0].scenario
    layout.scenario_plots_manifest(folder, scenario).unlink(missing_ok=True)
    timing = layout.scenario_plots(folder, scenario) / RepArtifact.TIMING
    start_timing(timing)
    for stage in _scenario_stages(reps, layout, atlas_format):
        console.say(scenario, f"{stage.label} started")
        result = launcher.run(stage.command, stage.log_path)
        append_timing(timing, stage.label, result)
        if result.exit_code != 0:
            console.say(scenario, f"{stage.label} FAILED ({result.failure}); log: {stage.log_path}")
            return False
        console.say(scenario, f"{stage.label} finished in {result.wall_s:.1f}s")
    plots = layout.scenario_plots(folder, scenario)
    write_plots_manifest(
        layout.scenario_plots_manifest(folder, scenario),
        {f"rep{rep.rep}": rep.path(layout, RepArtifact.RUN_MANIFEST) for rep in reps},
        [plots / "atlas.html", *([plots / "atlas.pdf"] if atlas_format == "pdf" else [])],
    )
    return True


def _print_plan(
    to_compute: list[tuple[ResolvedRep, int]],
    all_reps: list[ResolvedRep],
    layout: Layout,
    atlas_format: str,
    until: int,
    *,
    include_plots: bool,
) -> None:
    for rep, first in to_compute:
        print(f"# rep {rep.rep}: write {rep.path(layout, RepArtifact.PARAMS)}")
        for stage in STAGES[first:until]:
            print(shlex.join(command(stage.name, stage.argv(rep, layout))))
    if include_plots:
        for stage in _scenario_stages(all_reps, layout, atlas_format):
            print(shlex.join(stage.command), flush=True)
