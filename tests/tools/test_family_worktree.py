"""``tools/family_worktree.py`` on a throwaway family: simACE, fitACE, fitACE_epimight."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import family_worktree as fw
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


def commit_in(path: Path, rel: str) -> str:
    (path / rel).write_text("")
    git(path, "add", rel)
    git(path, "commit", "-q", "-m", rel)
    return git(path, "rev-parse", "HEAD")


def registered(repo: Path) -> str:
    return git(repo, "worktree", "list")


def test_remove_refuses_unmerged_detached_head_and_rescues_it_under_force(family: Path, capsys) -> None:
    assert main(["add", "wt1"], root=family) == 0
    wt = family / ".claude/worktrees/wt1"
    sha = commit_in(wt, "detached.py")
    assert main(["remove", "wt1"], root=family) == 1
    assert f"detached HEAD {sha[:12]}" in capsys.readouterr().err
    assert "wt1" in registered(family)
    assert main(["remove", "wt1", "--force"], root=family) == 0
    assert not wt.exists()
    assert git(family, "rev-parse", f"worktree-rescue-wt1-{sha[:12]}") == sha
    assert f"unreachable commit {sha}" not in git(family, "fsck", "--no-reflogs", "--unreachable")


def test_remove_reuses_matching_rescue_ref_and_refuses_a_conflicting_one(family: Path) -> None:
    main(["add", "wt1"], root=family)
    wt = family / ".claude/worktrees/wt1"
    sha = commit_in(wt, "detached.py")
    ref = f"worktree-rescue-wt1-{sha[:12]}"
    git(family, "branch", ref, "dev")
    assert main(["remove", "wt1", "--force"], root=family) == 1
    assert "wt1" in registered(family), "nothing removed when the rescue ref points elsewhere"
    git(family, "branch", "-f", ref, sha)
    assert main(["remove", "wt1", "--force"], root=family) == 0
    assert git(family, "rev-parse", ref) == sha


def test_remove_refuses_whole_family_when_any_checkout_is_unmerged(family: Path, capsys) -> None:
    main(["add", "wt1", "fitACE"], root=family)
    wt = family / ".claude/worktrees/wt1"
    commit_in(wt / "fitACE/fitACE_epimight", "inner.py")
    assert main(["remove", "wt1"], root=family) == 1
    assert "fitACE_epimight: detached HEAD" in capsys.readouterr().err
    assert "wt1" in registered(family / "fitACE")
    assert "wt1" in registered(family / "fitACE/fitACE_epimight")
    assert git(family / "fitACE", "branch", "--list", "wt1")


def test_remove_checks_a_checkout_switched_to_another_branch(family: Path, capsys) -> None:
    main(["add", "wt1", "fitACE"], root=family)
    wt_fitace = family / ".claude/worktrees/wt1/fitACE"
    git(wt_fitace, "switch", "-q", "-c", "other")
    commit_in(wt_fitace, "other.py")
    assert main(["remove", "wt1"], root=family) == 1
    assert "branch 'other' is not merged into main" in capsys.readouterr().err
    assert main(["remove", "wt1", "--force"], root=family) == 0
    assert git(family / "fitACE", "branch", "--list", "other")
    assert git(family / "fitACE", "branch", "--list", "wt1"), "the unused family branch is not deleted"


def test_remove_treats_a_git_error_as_failure(family: Path) -> None:
    main(["add", "wt1"], root=family)
    git(family / "fitACE/fitACE_epimight", "branch", "-m", "main", "trunk")
    with pytest.raises(subprocess.CalledProcessError):
        main(["remove", "wt1"], root=family)
    assert "wt1" in registered(family)
    assert "wt1" in registered(family / "fitACE")
    assert "wt1" in registered(family / "fitACE/fitACE_epimight")


def test_remove_retries_after_a_partial_removal(family: Path, monkeypatch) -> None:
    main(["add", "wt1"], root=family)
    real_git = fw.git

    def failing_git(cwd: Path, *args: str) -> str:
        if args[:2] == ("worktree", "remove") and cwd == family / "fitACE":
            raise subprocess.CalledProcessError(1, args, stderr="injected")
        return real_git(cwd, *args)

    monkeypatch.setattr(fw, "git", failing_git)
    with pytest.raises(subprocess.CalledProcessError):
        main(["remove", "wt1"], root=family)
    assert "wt1" not in registered(family / "fitACE/fitACE_epimight")
    assert "wt1" in registered(family / "fitACE")
    monkeypatch.undo()
    assert main(["remove", "wt1"], root=family) == 0
    assert not (family / ".claude/worktrees/wt1").exists()


INVALID_NAMES = ["feat/x", "../../tmp/x", "/tmp/x", "-x", "--force", ".hidden", "a..b", "x.lock", "x.", "", "a b"]


@pytest.mark.parametrize("name", INVALID_NAMES)
def test_add_and_remove_reject_invalid_names(family: Path, name: str) -> None:
    assert fw.add(family, name, {"fitACE"}, install=False) == 2
    assert fw.remove(family, name, force=False) == 2
    assert not (family / ".claude/worktrees").exists()
    assert git(family / "fitACE", "branch", "--list") == "* main"
    assert registered(family / "fitACE").count("\n") == 0


def test_add_and_remove_refuse_a_symlink_escaping_the_worktree_root(family: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (family / ".claude/worktrees").mkdir(parents=True)
    (family / ".claude/worktrees/esc").symlink_to(elsewhere)
    assert main(["add", "esc"], root=family) == 2
    assert main(["remove", "esc", "--force"], root=family) == 2
    assert not any(elsewhere.iterdir())
    assert registered(family).count("\n") == 0


def test_valid_slug_round_trips_through_add_list_remove(family: Path, capsys) -> None:
    assert main(["add", "r-relatives", "fitACE"], root=family) == 0
    capsys.readouterr()
    assert main(["list"], root=family) == 0
    out = capsys.readouterr().out
    assert out.startswith("r-relatives\n")
    assert "fitACE           r-relatives" in out
    assert main(["remove", "r-relatives"], root=family) == 0
    assert git(family / "fitACE", "branch", "--list", "r-relatives") == ""


def test_add_refuses_a_reserved_family_dir(family: Path) -> None:
    (family / ".claude/worktrees/wt1").mkdir(parents=True)
    assert main(["add", "wt1", "fitACE"], root=family) == 2
    assert registered(family).count("\n") == 0
    assert git(family / "fitACE", "branch", "--list", "wt1") == ""


@pytest.mark.parametrize("phase", ["copytree", "link_external_targets", "interrupt"])
def test_add_rolls_back_failures_after_worktree_creation(family: Path, monkeypatch, phase: str) -> None:
    if phase == "copytree":
        monkeypatch.setattr(fw.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("injected")))
    elif phase == "link_external_targets":
        monkeypatch.setattr(fw, "link_external_targets", lambda *a: (_ for _ in ()).throw(OSError("injected")))
    else:
        monkeypatch.setattr(fw, "link_external_targets", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert main(["add", "wt1", "fitACE"], root=family) == 1
    assert not (family / ".claude/worktrees/wt1").exists()
    for repo in (family, family / "fitACE", family / "fitACE/fitACE_epimight"):
        assert registered(repo).count("\n") == 0
    assert git(family / "fitACE", "branch", "--list", "wt1") == ""
    monkeypatch.undo()
    assert main(["add", "wt1", "fitACE"], root=family) == 0, "a clean rollback frees the name"


def test_add_reports_rollback_failures_and_keeps_survivors(family: Path, monkeypatch, capsys) -> None:
    real_git = fw.git

    def failing_git(cwd: Path, *args: str) -> str:
        if args[:2] == ("worktree", "add") and cwd == family / "fitACE/fitACE_epimight":
            raise subprocess.CalledProcessError(1, args, stderr="injected add failure")
        if args[:2] == ("worktree", "remove") and cwd == family / "fitACE":
            raise subprocess.CalledProcessError(1, args, stderr="injected remove failure")
        return real_git(cwd, *args)

    monkeypatch.setattr(fw, "git", failing_git)
    assert main(["add", "wt1", "simACE", "fitACE"], root=family) == 1
    err = capsys.readouterr().err
    assert "rollback incomplete" in err
    assert "fitACE: injected remove failure" in err
    assert "simACE: kept" in err, "the parent of a surviving worktree is not removed"
    wt = family / ".claude/worktrees/wt1"
    assert (wt / "fitACE/pixi.toml").exists()
    assert "wt1" in registered(family / "fitACE")
    assert "wt1" in registered(family)
    assert git(family / "fitACE", "branch", "--list", "wt1")
    assert git(family, "branch", "--list", "wt1")
    assert git(family / "fitACE/fitACE_epimight", "branch", "--list") == "* main"
    monkeypatch.undo()
    assert main(["add", "wt1", "simACE", "fitACE"], root=family) == 2, "an incomplete rollback keeps the name taken"


@pytest.mark.parametrize("interrupted", [False, True])
def test_add_install_failure_keeps_the_complete_worktree(family: Path, monkeypatch, capsys, interrupted) -> None:
    real_run = subprocess.run

    def failing_run(cmd, *args, **kwargs):
        if cmd[:2] == ["pixi", "install"]:
            if interrupted:
                raise KeyboardInterrupt
            return subprocess.CompletedProcess(cmd, 1)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(fw.subprocess, "run", failing_run)
    assert main(["add", "wt1", "fitACE", "--install"], root=family) == 1
    wt_fitace = family / ".claude/worktrees/wt1/fitACE"
    assert f"cd {wt_fitace} && pixi install --frozen" in capsys.readouterr().err
    assert "wt1" in registered(family / "fitACE")
    assert "wt1" in registered(family / "fitACE/fitACE_epimight")
    assert git(wt_fitace, "status", "--porcelain") == ""
    assert (wt_fitace / "fitACE_epimight/external/epimight-2.1/VERSION").exists()


def test_add_restores_tracked_symlinks_with_unquotable_names(family: Path) -> None:
    epimight = family / "fitACE/fitACE_epimight"
    names = ["é-link", "with space", "tab\tname", "new\nline", 'q"uote', "back\\slash", "trailing "]
    for n in names:
        (epimight / "external" / n).symlink_to("../../../external/epimight-2.1")
    git(epimight, "add", "external")
    git(epimight, "commit", "-q", "-m", "odd links")
    assert main(["add", "wt1"], root=family) == 0
    wt_external = family / ".claude/worktrees/wt1/fitACE/fitACE_epimight/external"
    for n in names:
        assert (wt_external / n / "VERSION").read_text() == "2.1\n", repr(n)
