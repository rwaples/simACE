"""``simace run <scenario>``: compute every rep of a scenario, then its plots.

Resume granularity is the rep, or with ``--until`` the stage. A rep whose
``run.yaml`` matches the current parameters is skipped; a rep with no
``run.yaml`` is recomputed from scratch; a rep whose ``run.yaml`` differs is
refused unless ``--force``. A rep built through fewer stages than asked for
resumes at its first missing stage.
Every stage runs as its own ``simace <stage>`` subprocess, in its own cgroup
when one is delegated, so its wall time and memory peaks land in the rep's
``timing.tsv``. Plots and the atlas are rebuilt on every run.
"""

from __future__ import annotations

__all__ = [
    "ScenarioError",
    "ScenarioRun",
    "built_stages",
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
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from simace.cli.cgroups import CgroupRoot, CgroupUnavailable
from simace.cli.layout import RepArtifact, add_root_args, resolve_roots
from simace.cli.manifest import (
    Manifest,
    PlotsState,
    PlotsStatus,
    RepState,
    RepStatus,
    manifest_params,
    plots_status,
    read_manifest,
    recorded_stages,
    rep_status,
    write_manifest,
    write_plots_manifest,
)
from simace.cli.stages import (
    PARAMS_YAML_KEYS,
    REP_LAYOUT,
    REP_OUTPUTS,
    RETIRED_OUTPUTS,
    STAGES,
    ResolvedRep,
    atlas_argv,
    plot_argv,
    rep_param_keys,
)
from simace.core.publish import TMP_SUFFIX, publish

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path
    from typing import TextIO

    from simace.cli.layout import Layout
    from simace.cli.stages import Stage

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
# parallel kernels and polars get every core. pedigree-graph's Rust pool gets an
# equal share of the cores per concurrent rep (see _child_env).
_KERNEL_THREADS = ("NUMBA_NUM_THREADS", "POLARS_MAX_THREADS")
_TIMING_HEADER = "stage\twall_s\tmax_rss_mb\ttree_peak_mb\texit_code\n"


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


def _stage_names(stages: Sequence[Stage] = STAGES) -> list[str]:
    return [stage.name for stage in stages]


def expected_manifest(rep: ResolvedRep, stages: Sequence[Stage] = STAGES) -> Manifest:
    """Return the manifest a rep built through ``stages`` would carry under the current parameters."""
    return Manifest(
        scenario=rep.scenario,
        rep=rep.rep,
        seed=rep.seed,
        resolved=manifest_params(rep.params, rep_param_keys(stages)),
        stages=_stage_names(stages),
        layout=REP_LAYOUT,
    )


def built_stages(rep: ResolvedRep, layout: Layout) -> tuple[Stage, ...]:
    """Return the stages ``rep``'s ``run.yaml`` records when they begin the chain, else the whole chain."""
    return _built(read_manifest(layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST)))


def _built(recorded: Any) -> tuple[Stage, ...]:
    """Return the stages a loaded ``run.yaml`` records when they begin the chain, else the whole chain.

    Returning the whole chain for any other list lets :func:`rep_status`
    report the difference in ``stages``, which makes the rep stale.
    """
    names = recorded_stages(recorded)
    if names and names == _stage_names()[: len(names)]:
        return STAGES[: len(names)]
    return STAGES


def status_on_disk(rep: ResolvedRep, layout: Layout, until: int = len(STAGES)) -> RepStatus:
    """Return one rep's state from its ``run.yaml`` and the outputs it declares, under the current parameters.

    The rep is checked against the stages its ``run.yaml`` records. One
    complete through fewer than the first ``until`` stages is partial.
    """
    recorded = read_manifest(layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST))
    if recorded is None:
        return RepStatus(RepState.ABSENT)
    built = _built(recorded)
    status = rep_status(recorded, expected_manifest(rep, built), rep_outputs(rep, layout, built))
    if status.state is RepState.COMPLETE and len(built) < until:
        return replace(status, state=RepState.PARTIAL, reasons=(f"built through {built[-1].name}",))
    return status


def scenario_plots_status(reps: list[ResolvedRep], layout: Layout) -> PlotsStatus:
    """Return whether a scenario's plots and atlas were built from its reps as they stand now."""
    if not reps:
        return PlotsStatus(PlotsState.ABSENT)
    folder, scenario = reps[0].folder, reps[0].scenario
    manifests = {f"rep{rep.rep}": layout.rep(folder, scenario, rep.rep, RepArtifact.RUN_MANIFEST) for rep in reps}
    not_complete = [f"rep{rep.rep}" for rep in reps if status_on_disk(rep, layout).state is not RepState.COMPLETE]
    return plots_status(layout.scenario_plots_manifest(folder, scenario), manifests, not_complete)


def rep_outputs(rep: ResolvedRep, layout: Layout, stages: Sequence[Stage] = STAGES) -> list[Path]:
    """Return every output a rep built through ``stages`` must have, fingerprinted in its ``run.yaml``."""
    artifacts = (RepArtifact.PARAMS, RepArtifact.TIMING, *(out for stage in stages for out in stage.outputs))
    return [layout.rep(rep.folder, rep.scenario, rep.rep, a) for a in artifacts]


def _command(stage: str, argv: list[str]) -> list[str]:
    return [sys.executable, "-m", "simace", stage, *argv]


@dataclass(frozen=True)
class StageResult:
    """One finished stage subprocess.

    ``max_rss_mb`` is ``wait4``'s ``ru_maxrss``, the largest single-process
    peak among the stage and the descendants it reaped; ``tree_peak_mb`` is
    its cgroup's ``memory.peak`` (every descendant, shared pages once, page
    cache included), or None without a delegated cgroup.
    """

    wall_s: float
    max_rss_mb: float
    tree_peak_mb: float | None
    exit_code: int
    over_memory: bool = False

    @property
    def memory(self) -> str:
        """The memory peaks, for the progress line."""
        tree = "" if self.tree_peak_mb is None else f", tree peak {self.tree_peak_mb:.0f} MB"
        return f"max RSS {self.max_rss_mb:.0f} MB{tree}"

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
    """Return the resident memory of a live process from ``/proc``; 0 once it has exited."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def _descendants(pid: int) -> list[int]:
    """Return the pids of every live descendant of ``pid``, from each thread's ``/proc`` children list."""
    found: list[int] = []
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            tids = os.listdir(f"/proc/{parent}/task")
        except FileNotFoundError:
            continue
        for tid in tids:
            try:
                with open(f"/proc/{parent}/task/{tid}/children", encoding="ascii") as fh:
                    children = [int(child) for child in fh.read().split()]
            except FileNotFoundError:
                continue
            found.extend(children)
            pending.extend(children)
    return found


class _TreeWatch(threading.Thread):
    """Kills a stage and its descendants once their summed resident memory goes over a cap.

    The fallback for ``--max-memory`` without a delegated cgroup. ``simace
    plot`` renders in worker processes, so the stage process alone
    understates the stage. Shared pages count once per process, and a spike
    shorter than ``_MEMORY_POLL_S`` is missed. The watcher is stopped before
    the stage is reaped, so the stage's pid cannot have been reused when it
    is killed.
    """

    def __init__(self, pid: int, max_rss: int, log: TextIO) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.max_rss = max_rss
        self.log = log
        self.over = False
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(_MEMORY_POLL_S):
            tree = [self.pid, *_descendants(self.pid)]
            if sum(map(_rss_bytes, tree)) > self.max_rss:
                self.over = True
                for pid in tree:
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                self.log.write(
                    f"\nsimace run: killed; resident memory went over --max-memory ({self.max_rss / 2**20:.0f} MB)\n"
                )
                return

    def finish(self) -> None:
        self._stop.set()
        self.join()


@dataclass(frozen=True)
class _Launcher:
    """How stage subprocesses start: their environment, an optional memory cap, and where they run.

    With a cgroup root each stage runs in its own leaf, which records the
    exact peak of the stage's process tree and, with a cap, has the kernel
    kill the tree once it goes over. Without one a :class:`_TreeWatch`
    enforces the cap by polling ``/proc``, and no tree peak is recorded.
    """

    env: dict[str, str]
    max_rss: int | None = None
    lock_fds: tuple[int, ...] = ()
    cgroups: CgroupRoot | None = None

    def run(self, cmd: list[str], log_path: Path) -> StageResult:
        """Run one stage with output to ``log_path``; record its ``wait4`` peak and, in a cgroup, its tree's."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            if self.cgroups is None:
                exit_code, max_rss_mb, over = self._polled(cmd, log)
                tree_peak_mb = None
            else:
                with self.cgroups.leaf(self.max_rss) as leaf:
                    exit_code, max_rss_mb = _reap(self._start(leaf.wrap(cmd), log))
                    tree_peak_mb = leaf.peak_bytes() / 2**20
                    over = self.max_rss is not None and leaf.oom_killed()
                if over:
                    log.write(f"\nsimace run: killed; memory went over --max-memory ({self.max_rss / 2**20:.0f} MB)\n")
        return StageResult(time.perf_counter() - start, max_rss_mb, tree_peak_mb, exit_code, over)

    def _start(self, cmd: list[str], log: TextIO) -> subprocess.Popen[bytes]:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env, pass_fds=self.lock_fds)

    def _polled(self, cmd: list[str], log: TextIO) -> tuple[int, float, bool]:
        proc = self._start(cmd, log)
        if self.max_rss is None:
            return (*_reap(proc), False)
        watch = _TreeWatch(proc.pid, self.max_rss, log)
        watch.start()
        os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
        watch.finish()
        return (*_reap(proc), watch.over)


def _reap(proc: subprocess.Popen[bytes]) -> tuple[int, float]:
    """Wait for ``proc``; return its exit code and ``wait4``'s ``ru_maxrss`` in MB."""
    _, status, rusage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    rss_bytes = rusage.ru_maxrss if sys.platform == "darwin" else rusage.ru_maxrss * 1024
    return proc.returncode, rss_bytes / 2**20


def _append_timing(path: Path, stage: str, result: StageResult) -> None:
    tree = "" if result.tree_peak_mb is None else f"{result.tree_peak_mb:.1f}"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{stage}\t{result.wall_s:.3f}\t{result.max_rss_mb:.1f}\t{tree}\t{result.exit_code}\n")


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


def _recompute(
    rep: ResolvedRep,
    layout: Layout,
    launcher: _Launcher,
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
    rep_dir = layout.rep_dir(rep.folder, rep.scenario, rep.rep)
    if first == 0:
        for artifact in REP_OUTPUTS:
            layout.rep(rep.folder, rep.scenario, rep.rep, artifact).unlink(missing_ok=True)
        for retired in RETIRED_OUTPUTS:
            (rep_dir / retired).unlink(missing_ok=True)
    else:
        layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST).unlink()
        for stage in STAGES[first:]:
            for artifact in stage.outputs:
                layout.rep(rep.folder, rep.scenario, rep.rep, artifact).unlink(missing_ok=True)
    if rep_dir.exists():
        for stale in rep_dir.glob(f"*{TMP_SUFFIX}"):
            stale.unlink()

    _write_params(rep, layout)
    timing = layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.TIMING)
    if first == 0:
        _start_timing(timing)
    for stage in STAGES[first:until]:
        console.say(tag, f"{stage.name} started")
        log_path = layout.log(rep.folder, rep.scenario, rep.rep, stage.name)
        result = launcher.run(_command(stage.name, stage.argv(rep, layout)), log_path)
        _append_timing(timing, stage.name, result)
        if result.exit_code != 0:
            failure = f"{stage.name} FAILED ({result.failure}); log: {log_path}"
            console.say(tag, failure)
            return failure
        console.say(tag, f"{stage.name} finished in {result.wall_s:.1f}s, {result.memory}")

    write_manifest(
        layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST),
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
    # pedigree-graph defaults to one thread, so without this its relationship-pair
    # extraction in analyze runs serially.
    env.setdefault("PEDIGREE_GRAPH_THREADS", str(max(1, len(os.sched_getaffinity(0)) // jobs)))
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
        "--until",
        choices=_stage_names(),
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
        "otherwise the tree's resident memory is polled from /proc",
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
    args.until = _stage_names().index(args.until) + 1
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
        with ExitStack() as stack:
            lock_fds: tuple[int, ...] = ()
            cgroups = None
            if not args.dry_run:
                lock_fds = tuple(_lock_scenario(stack, run, layout) for run in runs)
                cgroups = _open_cgroups(stack)
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


def _open_cgroups(stack: ExitStack) -> CgroupRoot | None:
    try:
        return stack.enter_context(CgroupRoot.open())
    except CgroupUnavailable as exc:
        print(
            f"simace run: no delegated cgroup ({exc}); tree_peak_mb not recorded, --max-memory polls /proc",
            file=sys.stderr,
        )
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
        _Launcher(_child_env(args.jobs), args.max_memory, lock_fds, cgroups),
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
        launcher = _Launcher(_child_env(1), args.max_memory, lock_fds, cgroups)
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
    launcher: _Launcher,
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
    to_compute: list[tuple[ResolvedRep, int]],
    all_reps: list[ResolvedRep],
    layout: Layout,
    atlas_format: str,
    until: int,
    *,
    include_plots: bool,
) -> None:
    for rep, first in to_compute:
        print(f"# rep {rep.rep}: write {layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.PARAMS)}")
        for stage in STAGES[first:until]:
            print(shlex.join(_command(stage.name, stage.argv(rep, layout))))
    if include_plots:
        for stage in _scenario_stages(all_reps, layout, atlas_format):
            print(shlex.join(stage.command), flush=True)
