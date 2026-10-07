"""Stage subprocesses for ``simace run``: their environment, memory metering and cap, and ``timing.tsv``.

Every stage runs as its own ``simace <stage>`` subprocess, in its own cgroup
leaf when one is delegated (:mod:`simace.cli.cgroups`), so its wall time and
memory peaks are exact.
"""

from __future__ import annotations

__all__ = [
    "Launcher",
    "StageResult",
    "append_timing",
    "child_env",
    "command",
    "parse_size",
    "start_timing",
]

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO

    from simace.cli.cgroups import CgroupRoot

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
# equal share of the cores per concurrent rep (see child_env).
_KERNEL_THREADS = ("NUMBA_NUM_THREADS", "POLARS_MAX_THREADS")
_TIMING_HEADER = "stage\twall_s\tmax_rss_mb\ttree_peak_mb\texit_code\n"


def command(stage: str, argv: list[str]) -> list[str]:
    """Return the argv that runs ``simace <stage>`` under this interpreter."""
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
class Launcher:
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


def append_timing(path: Path, stage: str, result: StageResult) -> None:
    """Append one stage's row to a ``timing.tsv``."""
    tree = "" if result.tree_peak_mb is None else f"{result.tree_peak_mb:.1f}"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{stage}\t{result.wall_s:.3f}\t{result.max_rss_mb:.1f}\t{tree}\t{result.exit_code}\n")


def start_timing(path: Path) -> None:
    """Create a ``timing.tsv`` holding only its header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TIMING_HEADER, encoding="utf-8")


def child_env(jobs: int) -> dict[str, str]:
    """Return the environment for stage subprocesses when ``jobs`` reps run at once."""
    env = {**os.environ, **dict.fromkeys(_BLAS_THREADS, "1")}
    if jobs > 1:
        env.update(dict.fromkeys(_KERNEL_THREADS, "1"))
    # pedigree-graph defaults to one thread, so without this its relationship-pair
    # extraction in analyze runs serially.
    env.setdefault("PEDIGREE_GRAPH_THREADS", str(max(1, (os.process_cpu_count() or 1) // jobs)))
    return env
