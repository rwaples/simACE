"""Exact per-stage memory accounting and caps through a delegated cgroup v2 subtree.

Each stage runs in its own child cgroup (a :class:`Leaf`) of a
:class:`CgroupRoot`, so the kernel keeps the exact high-water mark of the
stage and every process it starts (``memory.peak``) and enforces a cap
(``memory.max``) without sampling. ``memory.peak`` counts a shared page once,
and counts page cache and kernel memory charged to the cgroup as well as
anonymous memory.

The root is a transient systemd scope started with ``Delegate=yes``, or the
directory named by ``SIMACE_CGROUP_ROOT``, which is how ``tools/benchmark``
puts a run's stage cgroups under its own root.
"""

from __future__ import annotations

__all__ = ["ROOT_ENV", "CgroupRoot", "CgroupUnavailable", "Leaf"]

import itertools
import logging
import os
import select
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

ROOT_ENV = "SIMACE_CGROUP_ROOT"
_ANCHOR_TIMEOUT_S = 10.0
_DRAIN_TIMEOUT_S = 2.0
# A cgroup that holds processes cannot enable controllers for its children, so
# the anchor moves into a child of the scope before enabling memory. It then
# holds the scope open until its stdin closes, which also happens when the
# process that started it dies.
_ANCHOR = (
    "cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)"
    ' && mkdir "$cg/anchor" && echo 0 > "$cg/anchor/cgroup.procs"'
    ' && echo +memory > "$cg/cgroup.subtree_control" && echo "$cg" && exec cat > /dev/null'
)
# Memory is charged to the cgroup a process is in when it allocates and never
# moves with the process, so the stage joins its leaf before it execs.
_JOIN = 'echo 0 > "$0/cgroup.procs" && exec "$@"'


class CgroupUnavailable(Exception):
    """No delegated cgroup with the memory controller is available; the message says why."""


def _events(path: Path) -> dict[str, int]:
    return {key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())}


def _peak_bytes(cgroup: Path) -> int:
    return int((cgroup / "memory.peak").read_text())


@dataclass(frozen=True)
class Leaf:
    """One stage's cgroup."""

    path: Path

    def wrap(self, cmd: list[str]) -> list[str]:
        """Return ``cmd`` behind a shim that moves it into this cgroup before it runs; its pid is kept."""
        return ["sh", "-c", _JOIN, str(self.path), *cmd]

    def peak_bytes(self) -> int:
        """Return the most memory charged to this cgroup at once since it was created."""
        return _peak_bytes(self.path)

    def oom_killed(self) -> bool:
        """Return whether the OOM killer killed a process in this cgroup."""
        return _events(self.path / "memory.events")["oom_kill"] > 0


class CgroupRoot:
    """A cgroup this process may create children in, with the memory controller enabled for them."""

    def __init__(self, path: Path, anchor: subprocess.Popen[str] | None = None) -> None:
        self.path = path
        self._anchor = anchor
        self._names = itertools.count()

    @classmethod
    def open(cls) -> CgroupRoot:
        """Adopt the cgroup ``SIMACE_CGROUP_ROOT`` names, else start a delegated systemd scope.

        Raises:
            CgroupUnavailable: neither is possible here, or the kernel lacks
                ``memory.peak`` (Linux 5.19) or ``cgroup.kill`` (5.14).
        """
        adopted = os.environ.get(ROOT_ENV)
        if adopted:
            root = cls(Path(adopted))
            try:
                enabled = (root.path / "cgroup.subtree_control").read_text().split()
            except OSError as exc:
                raise CgroupUnavailable(f"{ROOT_ENV}={adopted}: {exc.strerror}") from None
            if "memory" not in enabled:
                raise CgroupUnavailable(f"{ROOT_ENV}={adopted} does not enable memory for its children")
        else:
            root = cls._start_scope()
        missing = [name for name in ("memory.peak", "cgroup.kill") if not (root.path / name).exists()]
        if missing:
            root.close()
            raise CgroupUnavailable(f"the kernel has no {' or '.join(missing)} (needs Linux 5.19 or later)")
        return root

    @classmethod
    def _start_scope(cls) -> CgroupRoot:
        if sys.platform != "linux":
            raise CgroupUnavailable("not Linux")
        if shutil.which("systemd-run") is None:
            raise CgroupUnavailable("systemd-run not found")
        anchor = subprocess.Popen(
            ["systemd-run", "--user", "--scope", "--quiet", "-p", "Delegate=yes", "--", "sh", "-c", _ANCHOR],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert anchor.stdout is not None
        ready, _, _ = select.select([anchor.stdout], [], [], _ANCHOR_TIMEOUT_S)
        line = anchor.stdout.readline().strip() if ready else ""
        if line:
            return cls(Path(line), anchor)
        anchor.kill()
        _, stderr = anchor.communicate()
        if not ready:
            raise CgroupUnavailable(f"systemd-run started no scope within {_ANCHOR_TIMEOUT_S:.0f} s")
        lines = stderr.strip().splitlines()
        raise CgroupUnavailable(lines[-1] if lines else f"systemd-run exited {anchor.returncode}")

    @contextmanager
    def leaf(self, max_bytes: int | None = None) -> Iterator[Leaf]:
        """Yield a new child cgroup; on exit kill whatever is left in it and remove it.

        With ``max_bytes`` the child is capped there with no swap, and the OOM
        killer kills every process in it together once it goes over.
        """
        path = self.path / f"stage-{os.getpid()}-{next(self._names)}"
        path.mkdir()
        try:
            if max_bytes is not None:
                (path / "memory.max").write_text(str(max_bytes))
                # Without this a capped stage swaps instead of being killed.
                if (path / "memory.swap.max").exists():
                    (path / "memory.swap.max").write_text("0")
                (path / "memory.oom.group").write_text("1")
            yield Leaf(path)
        finally:
            _remove(path)

    def peak_bytes(self) -> int:
        """Return the most memory charged to this root and its children at once."""
        return _peak_bytes(self.path)

    def close(self) -> None:
        """Release a scope this root started; an adopted root is left alone."""
        if self._anchor is not None:
            self._anchor.communicate()
            self._anchor = None

    def __enter__(self) -> CgroupRoot:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _remove(path: Path) -> None:
    """Kill every process in ``path`` and remove it once empty; warn if it does not empty in time."""
    (path / "cgroup.kill").write_text("1")
    deadline = time.monotonic() + _DRAIN_TIMEOUT_S
    while _events(path / "cgroup.events")["populated"] and time.monotonic() < deadline:
        time.sleep(0.005)
    try:
        path.rmdir()
    except OSError as exc:
        logger.warning("could not remove cgroup %s (%s); it goes when its scope does", path, exc.strerror)
