"""``simace run <scenario>``: compute every rep of a scenario, then its plots.

Resume granularity is the rep. A rep whose ``run.yaml`` matches the current
parameters is skipped; a rep with no ``run.yaml`` is recomputed from
scratch; a rep whose ``run.yaml`` differs is refused unless ``--force``.
Every stage runs as its own ``simace <stage>`` subprocess so its wall time
and peak RSS land in the rep's ``timing.tsv``. Plots and the atlas are
rebuilt on every run.
"""

from __future__ import annotations

__all__ = ["ScenarioError", "check_runnable", "cli", "expected_manifest", "load_scenario", "resolve_all"]

import argparse
import fcntl
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simace.cli.layout import Layout, RepArtifact
from simace.cli.manifest import Manifest, RepState, manifest_params, rep_status, write_manifest
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
    from collections.abc import Iterator
    from typing import TextIO

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
    """Another ``simace run`` holds the scenario's lock."""


@contextmanager
def _scenario_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` for the block, recording our pid in it.

    The kernel drops the lock when the process exits, however it exits, so a
    killed run never leaves the scenario locked.

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
        yield


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


def load_scenario(config_dir: Path, name: str) -> dict[str, Any]:
    """Return the resolved flat parameters of one runnable scenario.

    Raises:
        ScenarioError: the scenario is unknown, or uses gene drop.
    """
    scenarios = resolve_all(config_dir)
    if name not in scenarios:
        known = "\n  ".join(sorted(scenarios))
        raise ScenarioError(f"unknown scenario {name!r}; known scenarios:\n  {known}")
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

    def run(self, cmd: list[str], log_path: Path) -> StageResult:
        """Run one stage with output to ``log_path``; measure it with ``wait4``."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env)
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


def _recompute(rep: ResolvedRep, layout: Layout, launcher: _Launcher, console: _Console) -> bool:
    """Compute every stage of one rep from scratch. Return True when all stages exit 0.

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
            console.say(tag, f"{stage.name} FAILED ({result.failure}); log: {log_path}")
            return False
        console.say(tag, f"{stage.name} finished in {result.wall_s:.1f}s, peak {result.max_rss_mb:.0f} MB")

    write_manifest(
        layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST),
        expected_manifest(rep),
    )
    return True


def _scenario_stages(reps: list[ResolvedRep], layout: Layout, atlas_format: str) -> list[tuple[str, list[str], Path]]:
    folder, scenario = reps[0].folder, reps[0].scenario
    return [
        ("plot", plot_argv(reps, layout), layout.scenario_log(folder, scenario, "plot")),
        ("atlas", atlas_argv(reps, layout, atlas_format), layout.scenario_log(folder, scenario, "atlas")),
    ]


def _child_env(jobs: int) -> dict[str, str]:
    env = {**os.environ, **dict.fromkeys(_BLAS_THREADS, "1")}
    if jobs > 1:
        env.update(dict.fromkeys(_KERNEL_THREADS, "1"))
    else:
        # numba and polars default to every core; pedigree-graph defaults to one.
        env.setdefault("PEDIGREE_GRAPH_THREADS", str(len(os.sched_getaffinity(0))))
    return env


def _parse(argv: list[str] | None, prog: str | None) -> argparse.Namespace:
    from simace.core.cli_base import add_version_arg

    parser = argparse.ArgumentParser(prog=prog, description="Run every replicate of a scenario, then its plots")
    add_version_arg(parser, "simace")
    parser.add_argument("scenario", help="Scenario name from config/{folder}.yaml")
    parser.add_argument(
        "--rep", type=int, nargs="+", default=None, help="Replicates to compute (default: 1..replicates)"
    )
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Reps computed concurrently (default: 1)")
    parser.add_argument("--force", action="store_true", help="Recompute requested reps even when complete or stale")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print what would run; write nothing")
    parser.add_argument("--fail-fast", action="store_true", help="Cancel pending reps after the first failure")
    parser.add_argument(
        "--max-memory",
        type=parse_size,
        default=None,
        metavar="SIZE",
        help="Kill any stage whose resident memory goes over SIZE (e.g. 8G); applies to each stage, not the whole run",
    )
    parser.add_argument(
        "--format", choices=("html", "pdf"), default="html", help="Scenario atlas format (default: html)"
    )
    parser.add_argument("--config-dir", type=Path, default=Path("config"), help="Config directory (default: config)")
    parser.add_argument("--results", type=Path, default=Path("results"), help="Results root (default: results)")
    parser.add_argument("--logs", type=Path, default=Path("logs"), help="Log root (default: logs)")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line entry point for ``simace run``."""
    args = _parse(argv, prog)
    try:
        params = load_scenario(args.config_dir, args.scenario)
    except ScenarioError as exc:
        print(f"simace run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    layout = Layout(root=args.results, logs=args.logs)
    folder, n_reps = params["folder"], int(params["replicates"])
    if args.rep is not None:
        bad = sorted({r for r in args.rep if not 1 <= r <= n_reps} | {r for r in args.rep if args.rep.count(r) > 1})
        if bad:
            print(f"simace run: --rep values must be distinct and in 1..{n_reps}; got {bad}", file=sys.stderr)
            raise SystemExit(2)
    all_reps = [ResolvedRep(folder, args.scenario, r, params) for r in range(1, n_reps + 1)]
    requested = all_reps if args.rep is None else [all_reps[r - 1] for r in args.rep]

    lock_path = layout.scenario_lock(folder, args.scenario)
    try:
        with nullcontext() if args.dry_run else _scenario_lock(lock_path):
            _run_reps(args, layout, all_reps, requested)
    except ScenarioBusy as exc:
        print(
            f"simace run: another simace run (pid {exc}) is running {args.scenario}; lock: {lock_path}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _run_reps(
    args: argparse.Namespace, layout: Layout, all_reps: list[ResolvedRep], requested: list[ResolvedRep]
) -> None:
    folder = all_reps[0].folder
    console = _Console()
    to_compute: list[ResolvedRep] = []
    refused: list[int] = []
    for rep in requested:
        manifest_path = layout.rep(folder, args.scenario, rep.rep, RepArtifact.RUN_MANIFEST)
        status = rep_status(manifest_path, expected_manifest(rep))
        tag = f"{args.scenario}/rep{rep.rep}"
        if args.force or status.state is RepState.ABSENT:
            to_compute.append(rep)
        elif status.state is RepState.COMPLETE:
            console.say(tag, "skip (run.yaml matches)")
        else:
            console.say(tag, f"refused: run.yaml differs in {', '.join(status.differing)}; --force recomputes it")
            refused.append(rep.rep)

    if args.dry_run:
        _print_plan(to_compute, all_reps, layout, args.format)
        raise SystemExit(1 if refused else 0)

    failed = _compute(
        to_compute,
        layout,
        _Launcher(_child_env(args.jobs), args.max_memory),
        console,
        jobs=args.jobs,
        fail_fast=args.fail_fast,
    )
    if refused or failed:
        raise SystemExit(1)

    incomplete = [
        rep.rep
        for rep in all_reps
        if rep_status(
            layout.rep(folder, args.scenario, rep.rep, RepArtifact.RUN_MANIFEST), expected_manifest(rep)
        ).state
        is not RepState.COMPLETE
    ]
    if incomplete:
        console.say(args.scenario, f"plots skipped: reps {incomplete} are not complete")
        return
    if not _build_plots(all_reps, layout, args.format, _Launcher(_child_env(1), args.max_memory), console):
        raise SystemExit(1)


def _compute(
    reps: list[ResolvedRep],
    layout: Layout,
    launcher: _Launcher,
    console: _Console,
    *,
    jobs: int,
    fail_fast: bool,
) -> list[int]:
    """Recompute ``reps``; return the rep numbers that failed or were cancelled."""
    cancelled = threading.Event()

    def one(rep: ResolvedRep) -> bool:
        if cancelled.is_set():
            console.say(f"{rep.scenario}/rep{rep.rep}", "cancelled")
            return False
        ok = _recompute(rep, layout, launcher, console)
        if not ok and fail_fast:
            cancelled.set()
        return ok

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        outcomes = list(pool.map(one, reps))
    return [rep.rep for rep, ok in zip(reps, outcomes, strict=True) if not ok]


def _build_plots(
    reps: list[ResolvedRep], layout: Layout, atlas_format: str, launcher: _Launcher, console: _Console
) -> bool:
    folder, scenario = reps[0].folder, reps[0].scenario
    timing = layout.scenario_plots(folder, scenario) / RepArtifact.TIMING
    _start_timing(timing)
    for name, argv, log_path in _scenario_stages(reps, layout, atlas_format):
        console.say(scenario, f"{name} started")
        result = launcher.run(_command(name, argv), log_path)
        _append_timing(timing, name, result)
        if result.exit_code != 0:
            console.say(scenario, f"{name} FAILED ({result.failure}); log: {log_path}")
            return False
        console.say(scenario, f"{name} finished in {result.wall_s:.1f}s")
    return True


def _print_plan(to_compute: list[ResolvedRep], all_reps: list[ResolvedRep], layout: Layout, atlas_format: str) -> None:
    for rep in to_compute:
        print(f"# rep {rep.rep}: write {layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.PARAMS)}")
        for stage in STAGES:
            print(shlex.join(_command(stage.name, stage.argv(rep, layout))))
    for name, argv, _ in _scenario_stages(all_reps, layout, atlas_format):
        print(shlex.join(_command(name, argv)), flush=True)
