"""``tools/family_worktree.py`` on a throwaway family: simACE, fitACE, fitACE_epimight."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from family_worktree import main


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def make_repo(path: Path, branch: str, files: dict[str, str], ignore: str = "") -> None:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", branch)
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")
    (path / ".gitignore").write_text(ignore)
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "init")


@pytest.fixture
def family(tmp_path: Path) -> Path:
    root = tmp_path / "simACE"
    make_repo(root, "dev", {"README.md": "simACE\n"}, ignore="/fitACE/\nexternal/\n.claude/\n")
    fitace = root / "fitACE"
    make_repo(fitace, "main", {"pixi.toml": ""}, ignore="fitACE_epimight/\nbuild-fp*/\nldak6.2.simace\n")
    (fitace / "results").symlink_to("../results")
    git(fitace, "add", "results")
    git(fitace, "commit", "-q", "-m", "results link")
    (root / "results").mkdir()
    (fitace / "fitACE_iter_reml/ace_iter_reml/build-fp32").mkdir(parents=True)
    (fitace / "fitACE_iter_reml/ace_iter_reml/build-fp32/ace_iter_reml").write_text("binary")
    (fitace / "tetraher_simace").mkdir()
    (fitace / "tetraher_simace/ldak6.2.simace").write_text("binary")
    epimight = fitace / "fitACE_epimight"
    make_repo(epimight, "main", {"pkg.py": ""})
    (epimight / "external").mkdir()
    (epimight / "external/epimight-2.1").symlink_to("../../../external/epimight-2.1")
    git(epimight, "add", "external")
    git(epimight, "commit", "-q", "-m", "link")
    (root / "external/epimight-2.1").mkdir(parents=True)
    (root / "external/epimight-2.1/VERSION").write_text("2.1\n")
    return root


def test_add_branches_named_repos_and_restores_ignored_inputs(family: Path) -> None:
    assert main(["add", "wt1", "fitACE"], root=family) == 0
    wt = family / ".claude/worktrees/wt1"
    assert git(wt, "branch", "--show-current") == ""
    assert git(wt / "fitACE", "branch", "--show-current") == "wt1"
    assert git(wt / "fitACE/fitACE_epimight", "branch", "--show-current") == ""
    assert (wt / "fitACE/fitACE_iter_reml/ace_iter_reml/build-fp32/ace_iter_reml").read_text() == "binary"
    assert (wt / "fitACE/tetraher_simace/ldak6.2.simace").read_text() == "binary"
    assert (wt / "fitACE/fitACE_epimight/external/epimight-2.1/VERSION").read_text() == "2.1\n"
    assert not (wt / "results").exists(), "outputs must not be shared with the main checkout"
    assert git(wt / "fitACE", "status", "--porcelain") == ""


def test_add_refuses_unknown_repo_and_existing_name(family: Path) -> None:
    assert main(["add", "wt1", "nope"], root=family) == 2
    assert main(["add", "wt1"], root=family) == 0
    assert main(["add", "wt1"], root=family) == 2


def test_remove_refuses_dirty_and_unmerged_then_keeps_unmerged_branch(family: Path) -> None:
    main(["add", "wt1", "fitACE"], root=family)
    wt_fitace = family / ".claude/worktrees/wt1/fitACE"
    (wt_fitace / "new.py").write_text("x = 1\n")
    assert main(["remove", "wt1"], root=family) == 1
    git(wt_fitace, "add", "new.py")
    git(wt_fitace, "commit", "-q", "-m", "work")
    assert main(["remove", "wt1"], root=family) == 1
    assert main(["remove", "wt1", "--force"], root=family) == 0
    assert not (family / ".claude/worktrees/wt1").exists()
    assert git(family / "fitACE", "branch", "--list", "wt1")
    assert "wt1" not in git(family / "fitACE", "worktree", "list")


def test_remove_deletes_merged_branch(family: Path) -> None:
    main(["add", "wt1", "fitACE"], root=family)
    assert main(["remove", "wt1"], root=family) == 0
    assert git(family / "fitACE", "branch", "--list", "wt1") == ""
    assert git(family, "worktree", "list").count("\n") == 0


def test_remove_deletes_merged_branch_while_main_checkout_is_elsewhere(family: Path) -> None:
    fitace = family / "fitACE"
    git(fitace, "switch", "-q", "-c", "feature")
    (fitace / "feature.py").write_text("")
    git(fitace, "add", "feature.py")
    git(fitace, "commit", "-q", "-m", "feature")
    main(["add", "wt1", "fitACE"], root=family)
    wt_fitace = family / ".claude/worktrees/wt1/fitACE"
    (wt_fitace / "work.py").write_text("")
    git(wt_fitace, "add", "work.py")
    git(wt_fitace, "commit", "-q", "-m", "work")
    git(fitace, "update-ref", "refs/heads/main", "wt1")
    assert main(["remove", "wt1"], root=family) == 0
    assert git(fitace, "branch", "--list", "wt1") == ""


def test_remove_leaves_same_named_branch_of_a_detached_checkout(family: Path) -> None:
    epimight = family / "fitACE/fitACE_epimight"
    git(epimight, "switch", "-q", "-c", "wt1")
    (epimight / "side.py").write_text("")
    git(epimight, "add", "side.py")
    git(epimight, "commit", "-q", "-m", "side")
    git(epimight, "switch", "-q", "main")
    main(["add", "wt1", "fitACE"], root=family)
    assert main(["remove", "wt1"], root=family) == 0
    assert git(epimight, "branch", "--list", "wt1")


def test_add_rolls_back_when_a_checkout_fails(family: Path) -> None:
    git(family / "fitACE/fitACE_epimight", "branch", "-m", "main", "trunk")
    assert main(["add", "wt1", "fitACE", "fitACE_epimight"], root=family) == 1
    assert not (family / ".claude/worktrees/wt1").exists()
    assert git(family / "fitACE", "branch", "--list", "wt1") == ""
    assert git(family / "fitACE", "worktree", "list").count("\n") == 0
    assert git(family, "worktree", "list").count("\n") == 0
