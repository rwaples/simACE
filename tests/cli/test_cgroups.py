"""cgroup v2 leaves: exact tree peaks, kernel memory caps, and cleanup."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from simace.cli.cgroups import ROOT_ENV, CgroupRoot, CgroupUnavailable

if TYPE_CHECKING:
    from collections.abc import Iterator

MiB = 2**20


def _hold(mb: int, seconds: float) -> str:
    return f"import time; b = b'x' * ({mb} * 2**20); time.sleep({seconds})"


def _run(root: CgroupRoot, code: str, max_bytes: int | None = None) -> tuple[int, int, bool]:
    """Run ``code`` in a fresh leaf; return its exit code, the leaf's peak, and whether the OOM killer fired."""
    with root.leaf(max_bytes) as leaf:
        exit_code = subprocess.run(leaf.wrap([sys.executable, "-c", code]), check=False).returncode
        return exit_code, leaf.peak_bytes(), leaf.oom_killed()


@pytest.fixture
def root(monkeypatch) -> Iterator[CgroupRoot]:
    monkeypatch.delenv(ROOT_ENV, raising=False)
    try:
        opened = CgroupRoot.open()
    except CgroupUnavailable as exc:
        pytest.skip(f"no delegated cgroup: {exc}")
    with opened:
        yield opened


def _gone(path: Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return not path.exists()


def test_leaf_records_the_exact_peak_of_a_short_spike(root) -> None:
    code, peak, oom = _run(root, _hold(400, 0.03))
    assert (code, oom) == (0, False)
    assert 400 * MiB <= peak < 480 * MiB


def test_leaf_sums_concurrent_children(root) -> None:
    fork = (
        "import os, time\n"
        "pids = []\n"
        "for _ in range(2):\n"
        "    pid = os.fork()\n"
        "    if pid == 0:\n"
        f"        {_hold(300, 0.3)}; os._exit(0)\n"
        "    pids.append(pid)\n"
        "for pid in pids:\n"
        "    os.waitpid(pid, 0)\n"
    )
    code, peak, _ = _run(root, fork)
    assert code == 0
    assert peak >= 600 * MiB


def test_cap_kills_a_leaf_that_goes_over_it(root) -> None:
    code, peak, oom = _run(root, _hold(600, 1), max_bytes=400 * MiB)
    assert (code, oom) == (-9, True)
    assert peak <= 400 * MiB


def test_leaf_under_its_cap_runs_to_completion(root) -> None:
    code, _, oom = _run(root, _hold(100, 0), max_bytes=400 * MiB)
    assert (code, oom) == (0, False)


def _running(pid: int) -> bool:
    """Whether ``pid`` is alive and not a zombie waiting for its new parent to reap it."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat[stat.rfind(")") + 2] not in "ZX"


def test_leaf_exit_kills_orphans_and_removes_the_leaf(root, tmp_path) -> None:
    pid_file = tmp_path / "orphan.pid"
    orphan = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
    )
    with root.leaf() as leaf:
        assert subprocess.run(leaf.wrap([sys.executable, "-c", orphan]), check=False).returncode == 0
        pid = int(pid_file.read_text())
        assert _running(pid)
    assert not leaf.path.exists()
    assert not _running(pid)


def test_close_removes_the_scope(monkeypatch) -> None:
    monkeypatch.delenv(ROOT_ENV, raising=False)
    try:
        root = CgroupRoot.open()
    except CgroupUnavailable as exc:
        pytest.skip(f"no delegated cgroup: {exc}")
    assert root.path.is_dir()
    root.close()
    assert _gone(root.path)


def test_adopted_root_nests_leaves_and_its_owner_sees_their_peak(root, monkeypatch) -> None:
    monkeypatch.setenv(ROOT_ENV, str(root.path))
    with CgroupRoot.open() as adopted:
        assert adopted.path == root.path
        assert _run(adopted, _hold(300, 0))[0] == 0
    assert root.path.is_dir()
    assert root.peak_bytes() >= 300 * MiB


def test_adopting_a_cgroup_without_memory_is_refused(tmp_path, monkeypatch) -> None:
    (tmp_path / "cgroup.subtree_control").write_text("cpu pids\n")
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    with pytest.raises(CgroupUnavailable, match="does not enable memory"):
        CgroupRoot.open()
