"""``simace run <scenario>``: compute every rep of a scenario, then its plots.

Resume granularity is the rep. A rep whose ``run.yaml`` matches the current
parameters is skipped; a rep with no ``run.yaml`` is recomputed from
scratch; a rep whose ``run.yaml`` differs is refused unless ``--force``.
Every stage runs as its own ``simace <stage>`` subprocess so its wall time
and peak RSS land in the rep's ``timing.tsv``. Plots and the atlas are
rebuilt on every run.
"""

from __future__ import annotations

__all__ = [
    "ScenarioError",
    "ScenarioRun",
    "check_runnable",
    "cli",
    "expand_targets",
    "expected_manifest",
    "load_scenario",
    "rep_outputs",
    "rep_ranges",
    "rep_spec",
    "resolve_all",
    "scenario_plots_status",
    "status_on_disk",
]

import argparse
import fcntl
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from simace.cli.layout import RepArtifact, add_root_args, resolve_roots
from simace.cli.manifest import (
    Manifest,
    PlotsState,
    PlotsStatus,
    RepState,
    manifest_params,
    plots_status,
    rep_status,
    write_manifest,
    write_plots_manifest,
)
from simace.cli.stages import (
    PARAMS_YAML_KEYS,
    REP_OUTPUTS,
    REP_PARAM_KEYS,
    STAGES,
    ResolvedRep,
    atlas_argv,
    plot_argv,
)
from simace.core.publish import TMP_SUFFIX, publish

if TYPE_CHECKING:
    import resource
    from collections.abc import Iterable, Iterator
    from pathlib import Path
    from typing import TextIO

    from simace.cli.layout import Layout
    from simace.cli.manifest import RepStatus

# OpenMP/BLAS pools are pinned to one thread in every stage, as Snakemake's
# `threads: 1` rules (and `--cores 1`) did. Measured at baseline100K, one thread
# is as fast as four or five for simulate and analyze, and ~6% faster for plot.
_BLAS_THREADS = (
    "OMP_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# Pinned only when reps run concurrently; with one rep at a time the numba
# parallel kernels, polars, and pedigree-graph's Rust pool get every core.
_KERNEL_THREADS = ("NUMBA_NUM_THREADS", "POLARS_MAX_THREADS", "PEDIGREE_GRAPH_THREADS")
_TIMING_HEADER = "stage\twall_s\tmax_rss_mb\texit_code\n"


class ScenarioError(Exception):
    """A scenario that cannot be run: unknown, or outside what ``run`` supports."""


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


def resolve_all(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Return every scenario's resolved flat parameters (defaults merged in)."""
    from simace.config import resolve_defaults, resolve_scenarios

    defaults = resolve_defaults(config_dir)
    return {name: {**defaults, **params} for name, params in resolve_scenarios(config_dir, defaults).items()}


def check_runnable(name: str, params: dict[str, Any]) -> None:
    """Raise :class:`ScenarioError` if ``simace run`` cannot run this scenario."""
    if params.get("use_gene_drop") or params.get("drop_from") is not None:
        raise ScenarioError(
            f"scenario {name!r} uses gene drop (use_gene_drop / drop_from); "
            "`simace run` does not run it. Use the scripts in scripts/gene_drop/."
        )


def load_scenario(config_dir: Path, name: str, *, require_runnable: bool = True) -> dict[str, Any]:
    """Return the resolved flat parameters of one scenario.

    Raises:
        ScenarioError: the scenario is unknown, or uses gene drop when
            ``require_runnable`` is true.
    """
    scenarios = resolve_all(config_dir)
    if name not in scenarios:
        known = "\n  ".join(sorted(scenarios))
        raise ScenarioError(f"unknown scenario {name!r}; known scenarios:\n  {known}")
    if require_runnable:
        check_runnable(name, scenarios[name])
    return scenarios[name]


def _stage_names() -> list[str]:
    return [stage.name for stage in STAGES]


def expected_manifest(rep: ResolvedRep) -> Manifest:
    """Return the manifest a complete rep would carry under the current parameters."""
    return Manifest(
        scenario=rep.scenario,
        rep=rep.rep,
        seed=rep.seed,
        resolved=manifest_params(rep.params, REP_PARAM_KEYS),
        stages=_stage_names(),
    )


def status_on_disk(rep: ResolvedRep, layout: Layout) -> RepStatus:
    """Return one rep's state from its ``run.yaml`` and the outputs it declares, under the current parameters."""
    manifest = layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST)
    return rep_status(manifest, expected_manifest(rep), rep_outputs(rep, layout))


def scenario_plots_status(reps: list[ResolvedRep], layout: Layout) -> PlotsStatus:
    """Return whether a scenario's plots and atlas were built from its reps as they stand now."""
    if not reps:
        return PlotsStatus(PlotsState.ABSENT)
    folder, scenario = reps[0].folder, reps[0].scenario
    manifests = {f"rep{rep.rep}": layout.rep(folder, scenario, rep.rep, RepArtifact.RUN_MANIFEST) for rep in reps}
    not_complete = [f"rep{rep.rep}" for rep in reps if status_on_disk(rep, layout).state is not RepState.COMPLETE]
    return plots_status(layout.scenario_plots_manifest(folder, scenario), manifests, not_complete)


def rep_outputs(rep: ResolvedRep, layout: Layout) -> list[Path]:
    """Return every output a complete rep must have, fingerprinted in its ``run.yaml``."""
    return [layout.rep(rep.folder, rep.scenario, rep.rep, a) for a in REP_OUTPUTS if a is not RepArtifact.RUN_MANIFEST]


def _command(stage: str, argv: list[str]) -> list[str]:
    return [sys.executable, "-m", "simace", stage, *argv]


@dataclass(frozen=True)
class StageResult:
    """One finished stage subprocess."""

    wall_s: float
    max_rss_mb: float
    exit_code: int
    over_memory: bool = False

    @property
    def failure(self) -> str:
        """Why the stage failed, for the progress line."""
        if self.over_memory:
            return f"exit {self.exit_code}, killed for going over --max-memory"
        return f"exit {self.exit_code}"


_MEMORY_POLL_S = 0.1
_SIZE_UNITS = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def parse_size(text: str) -> int:
    """Parse ``512M``, ``8G``, or a plain byte count into bytes (binary units)."""
    number = text.strip().upper().rstrip("B")
    unit = number[-1:] if number[-1:] in _SIZE_UNITS else ""
    try:
        size = float(number.removesuffix(unit)) * _SIZE_UNITS[unit]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (use e.g. 512M or 8G)") from None
    if size <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive: {text!r}")
    return int(size)


def _rss_bytes(pid: int) -> int:
    """Return the resident memory of a live child from ``/proc``; 0 once it has exited."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


@dataclass(frozen=True)
class _Launcher:
    """How stage subprocesses start: their environment and an optional resident-memory cap.

    With a cap, the child's RSS is polled every ``_MEMORY_POLL_S`` and the
    child is killed once it goes over. Polling runs before the child is
    reaped, so its pid cannot have been reused when it is killed.
    """

    env: dict[str, str]
    max_rss: int | None = None
    lock_fds: tuple[int, ...] = ()

    def run(self, cmd: list[str], log_path: Path) -> StageResult:
        """Run one stage with output to ``log_path``; measure it with ``wait4``."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env, pass_fds=self.lock_fds)
            status, rusage, over = self._wait(proc, log)
        proc.returncode = os.waitstatus_to_exitcode(status)
        rss_bytes = rusage.ru_maxrss if sys.platform == "darwin" else rusage.ru_maxrss * 1024
        return StageResult(time.perf_counter() - start, rss_bytes / 2**20, proc.returncode, over)

    def _wait(self, proc: subprocess.Popen[bytes], log: TextIO) -> tuple[int, resource.struct_rusage, bool]:
        """Reap ``proc``; return its wait status, rusage, and whether it was killed for its memory."""
        if self.max_rss is None:
            _, status, rusage = os.wait4(proc.pid, 0)
            return status, rusage, False
        over = False
        while True:
            pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                return status, rusage, over
            if not over and _rss_bytes(proc.pid) > self.max_rss:
                over = True
                proc.kill()
                log.write(
                    f"\nsimace run: killed; resident memory went over --max-memory ({self.max_rss / 2**20:.0f} MB)\n"
                )
            time.sleep(_MEMORY_POLL_S)


def _append_timing(path: Path, stage: str, result: StageResult) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{stage}\t{result.wall_s:.3f}\t{result.max_rss_mb:.1f}\t{result.exit_code}\n")


def _start_timing(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TIMING_HEADER, encoding="utf-8")


def _write_params(rep: ResolvedRep, layout: Layout) -> None:
    from simace.core.yaml_io import dump_yaml
    from simace.simulation.emit_params import emit_params

    params = emit_params(seed=rep.seed, rep=rep.rep, **{key: rep.params[key] for key in PARAMS_YAML_KEYS})
    with publish(layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.PARAMS)) as (tmp,):
        dump_yaml(params, tmp, sort_keys=True)


class _Console:
    """Serializes progress lines from concurrent reps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def say(self, tag: str, message: str) -> None:
        with self._lock:
            print(f"[{tag}] {message}", flush=True)


def _recompute(rep: ResolvedRep, layout: Layout, launcher: _Launcher, console: _Console) -> str | None:
    """Compute every stage of one rep from scratch. Return None when all stages exit 0, else why not.

    Every file a previous run of this rep wrote is removed first, manifest
    first, so a failure partway leaves no output from the old parameters
    beside outputs from the new ones.
    """
    tag = f"{rep.scenario}/rep{rep.rep}"
    for artifact in REP_OUTPUTS:
        layout.rep(rep.folder, rep.scenario, rep.rep, artifact).unlink(missing_ok=True)
    rep_dir = layout.rep_dir(rep.folder, rep.scenario, rep.rep)
    if rep_dir.exists():
        for stale in rep_dir.glob(f"*{TMP_SUFFIX}"):
            stale.unlink()

    _write_params(rep, layout)
    timing = layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.TIMING)
    _start_timing(timing)
    for stage in STAGES:
        console.say(tag, f"{stage.name} started")
        log_path = layout.log(rep.folder, rep.scenario, rep.rep, stage.name)
        result = launcher.run(_command(stage.name, stage.argv(rep, layout)), log_path)
        _append_timing(timing, stage.name, result)
        if result.exit_code != 0:
            failure = f"{stage.name} FAILED ({result.failure}); log: {log_path}"
            console.say(tag, failure)
            return failure
        console.say(tag, f"{stage.name} finished in {result.wall_s:.1f}s, peak {result.max_rss_mb:.0f} MB")

    write_manifest(
        layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST),
        expected_manifest(rep),
        rep_outputs(rep, layout),
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
        return _command(self.subcommand, self.argv)


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


def _child_env(jobs: int) -> dict[str, str]:
    env = {**os.environ, **dict.fromkeys(_BLAS_THREADS, "1")}
    if jobs > 1:
        env.update(dict.fromkeys(_KERNEL_THREADS, "1"))
    else:
        # numba and polars default to every core; pedigree-graph defaults to one.
        env.setdefault("PEDIGREE_GRAPH_THREADS", str(len(os.sched_getaffinity(0))))
    return env


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
        "--max-memory",
        type=parse_size,
        default=None,
        metavar="SIZE",
        help="Kill any stage whose resident memory goes over SIZE (e.g. 8G); applies to each stage, not the whole run",
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
    if args.rep is not None:
        args.rep = [r for spec in args.rep for r in spec]
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
        with ExitStack() as locks:
            lock_fds: tuple[int, ...] = ()
            if not args.dry_run:
                lock_fds = tuple(_lock_scenario(locks, run, layout) for run in runs)
            _run_reps(args, layout, runs, lock_fds)
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


@dataclass
class _Outcome:
    """What happened to one scenario's requested reps."""

    to_compute: list[ResolvedRep] = field(default_factory=list)
    skipped: int = 0
    refused: dict[int, str] = field(default_factory=dict)
    failed: dict[int, str] = field(default_factory=dict)


def _classify(run: ScenarioRun, layout: Layout, console: _Console, *, force: bool) -> _Outcome:
    outcome = _Outcome()
    for rep in run.requested:
        status = status_on_disk(rep, layout)
        tag = f"{run.scenario}/rep{rep.rep}"
        if force or status.state is RepState.ABSENT:
            outcome.to_compute.append(rep)
        elif status.state is RepState.INCOMPLETE:
            console.say(tag, f"recompute: {', '.join(status.reasons)}")
            outcome.to_compute.append(rep)
        elif status.state is RepState.COMPLETE:
            console.say(tag, "skip (run.yaml matches)")
            outcome.skipped += 1
        else:
            outcome.refused[rep.rep] = f"run.yaml differs in {status.describe()}"
            console.say(tag, f"refused: {outcome.refused[rep.rep]}; --force recomputes it")
    return outcome


def _run_reps(args: argparse.Namespace, layout: Layout, runs: list[ScenarioRun], lock_fds: tuple[int, ...]) -> None:
    console = _Console()
    outcomes = {run.scenario: _classify(run, layout, console, force=args.force) for run in runs}
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
            plots = not args.no_plots and not outcome.refused and others_complete
            _print_plan(outcome.to_compute, run.all_reps, layout, args.format, include_plots=plots)
        raise SystemExit(1 if refused else 0)

    failed = _compute(
        [rep for run in runs for rep in outcomes[run.scenario].to_compute],
        layout,
        _Launcher(_child_env(args.jobs), args.max_memory, lock_fds),
        console,
        jobs=args.jobs,
        fail_fast=args.fail_fast,
    )
    for (scenario, rep), why in failed.items():
        outcomes[scenario].failed[rep] = why
    for run in runs:
        _summarize(run.scenario, console, outcomes[run.scenario])
    if len(runs) > 1:
        console.say("total", f"{len(runs)} scenarios: {_counts(outcomes.values())}")

    plotted = True
    if not args.no_plots:
        launcher = _Launcher(_child_env(1), args.max_memory, lock_fds)
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
    reps: list[ResolvedRep],
    layout: Layout,
    launcher: _Launcher,
    console: _Console,
    *,
    jobs: int,
    fail_fast: bool,
) -> dict[tuple[str, int], str]:
    """Recompute ``reps``; return why each rep that failed or was cancelled did not finish, by (scenario, rep)."""
    cancelled = threading.Event()

    def one(rep: ResolvedRep) -> str | None:
        if cancelled.is_set():
            console.say(f"{rep.scenario}/rep{rep.rep}", "cancelled")
            return "cancelled by --fail-fast"
        failure = _recompute(rep, layout, launcher, console)
        if failure and fail_fast:
            cancelled.set()
        return failure

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        outcomes = list(pool.map(one, reps))
    return {(rep.scenario, rep.rep): failure for rep, failure in zip(reps, outcomes, strict=True) if failure}


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


def rep_ranges(reps: list[int]) -> str:
    """Render sorted rep numbers as ``rep2`` or ``reps 1-3, 7``."""
    runs: list[list[int]] = []
    for rep in reps:
        if runs and rep == runs[-1][-1] + 1:
            runs[-1].append(rep)
        else:
            runs.append([rep])
    text = ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)
    return f"rep{text}" if len(reps) == 1 else f"reps {text}"


def _build_plots(
    reps: list[ResolvedRep], layout: Layout, atlas_format: str, launcher: _Launcher, console: _Console
) -> bool:
    folder, scenario = reps[0].folder, reps[0].scenario
    layout.scenario_plots_manifest(folder, scenario).unlink(missing_ok=True)
    timing = layout.scenario_plots(folder, scenario) / RepArtifact.TIMING
    _start_timing(timing)
    for stage in _scenario_stages(reps, layout, atlas_format):
        console.say(scenario, f"{stage.label} started")
        result = launcher.run(stage.command, stage.log_path)
        _append_timing(timing, stage.label, result)
        if result.exit_code != 0:
            console.say(scenario, f"{stage.label} FAILED ({result.failure}); log: {stage.log_path}")
            return False
        console.say(scenario, f"{stage.label} finished in {result.wall_s:.1f}s")
    plots = layout.scenario_plots(folder, scenario)
    write_plots_manifest(
        layout.scenario_plots_manifest(folder, scenario),
        {f"rep{rep.rep}": layout.rep(folder, scenario, rep.rep, RepArtifact.RUN_MANIFEST) for rep in reps},
        [plots / "atlas.html", *([plots / "atlas.pdf"] if atlas_format == "pdf" else [])],
    )
    return True


def _print_plan(
    to_compute: list[ResolvedRep],
    all_reps: list[ResolvedRep],
    layout: Layout,
    atlas_format: str,
    *,
    include_plots: bool,
) -> None:
    for rep in to_compute:
        print(f"# rep {rep.rep}: write {layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.PARAMS)}")
        for stage in STAGES:
            print(shlex.join(_command(stage.name, stage.argv(rep, layout))))
    if include_plots:
        for stage in _scenario_stages(all_reps, layout, atlas_format):
            print(shlex.join(stage.command), flush=True)
