"""``tools/githooks``: the shared family git hooks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PRE_PUSH = Path(__file__).resolve().parents[2] / "tools/githooks/pre-push"


def test_pre_push_runs_mkdocs_with_a_frozen_lock(tmp_path: Path) -> None:
    """A push never re-solves the env, so a stale pixi.lock is not rewritten."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "mkdocs.yml").write_text("site_name: x\n")
    argv_log = tmp_path / "argv"
    fake_pixi = tmp_path / "pixi"
    fake_pixi.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {argv_log}\n')
    fake_pixi.chmod(0o755)

    sha = "1" * 40
    subprocess.run(
        ["sh", str(PRE_PUSH)],
        cwd=repo,
        input=f"refs/heads/dev {sha} refs/heads/dev {'0' * 40}\n",
        text=True,
        check=True,
        env={**os.environ, "PIXI": str(fake_pixi)},
    )

    argv = argv_log.read_text().split("\n")
    assert argv[:3] == ["run", "--frozen", "mkdocs"]
