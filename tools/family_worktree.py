#!/usr/bin/env python
"""Create, list, and remove a family worktree: one git worktree per family checkout.

A family worktree lives at ``<main>/.claude/worktrees/<name>/`` and mirrors the
umbrella layout, so cross-repo paths (``fitACE/pixi.toml``'s ``simace = {path =
".."}``, pedsum's ty ``extra-paths = ../pedigree-graph``) resolve inside it.
The checkouts named on the command line get a ``<name>`` branch off their base
branch (``dev`` for simACE, ``main`` elsewhere); every other checkout gets a
detached worktree at its base, so nothing in the tree is shared with another
session.

``add`` also restores what git does not carry:

- fitACE's gitignored compiled binaries (``BUILD_OUTPUTS``), copied from the
  main checkout so the ace_iter_reml, pcgc, stan, and tetraher units run;
- the targets of tracked symlinks into ``external/`` (fitACE_epimight's
  ``external/epimight*``), linked back to the main checkout's copies, which are
  read-only references. Other dangling targets stay dangling: fitACE's
  ``results -> ../results`` must reach the worktree's own simACE results, not
  the main checkout's.

``remove`` checks every checkout before touching any: it refuses when a
worktree has uncommitted or untracked files or its ``<name>`` branch is not
merged into the base, unless ``--force``. It deletes merged branches only;
an unmerged branch is kept (``--force`` removes the worktree, never the branch).

Examples:
    python tools/family_worktree.py add issue-40 pedigree-graph pedsum
    python tools/family_worktree.py add bench-base               # all detached
    python tools/family_worktree.py add py315 --all --install
    python tools/family_worktree.py list
    python tools/family_worktree.py remove issue-40
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from family_repos import ROOT, checkout_repos

if TYPE_CHECKING:
    from family_repos import Repo

#: Base branch per checkout label; anything not listed uses ``main``.
BASE_BRANCH = {"simACE": "dev"}

#: Gitignored build outputs copied into a new fitACE worktree, relative to fitACE.
BUILD_OUTPUTS = (
    "fitACE_iter_reml/ace_iter_reml/build-fp32",
    "fitACE_iter_reml/ace_iter_reml/build-fp64",
    "fitACE_pcgc/ace_pcgc/build",
    "fitACE_stan/fitace_stan/ace_dii",
    "fitACE_stan/fitace_stan/fit_pedigree_ace",
    "fitACE_stan/fitace_stan/fit_pedigree_ace_reml",
    "tetraher_simace/ldak6.2.simace",
)


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return stripped stdout; raise on failure."""
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def main_root(start: Path = ROOT) -> Path:
    """The main simACE checkout, even when this script runs from a worktree."""
    return Path(git(start, "rev-parse", "--path-format=absolute", "--git-common-dir")).parent


def present(umbrella: Path) -> list[Repo]:
    """Family checkouts present under ``umbrella``, outermost first."""
    return [r for r in checkout_repos() if (umbrella / r.path / ".git").exists()]


def worktree_dir(umbrella: Path, name: str) -> Path:
    """Where family worktree ``name`` lives."""
    return umbrella / ".claude" / "worktrees" / name


def base(repo: Repo) -> str:
    """The branch new worktrees of ``repo`` start from."""
    return BASE_BRANCH.get(repo.label, "main")


def add(umbrella: Path, name: str, branched: set[str], install: bool) -> int:
    """Create family worktree ``name``, branching the checkouts in ``branched``."""
    repos = present(umbrella)
    unknown = branched - {r.label for r in repos}
    if unknown:
        print(f"unknown or missing checkouts: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    wt = worktree_dir(umbrella, name)
    if wt.exists():
        print(f"{wt} already exists", file=sys.stderr)
        return 2
    for r in repos:
        if r.label in branched and git(umbrella / r.path, "branch", "--list", name):
            print(f"{r.label}: branch {name!r} already exists", file=sys.stderr)
            return 2

    created: list[Repo] = []
    try:
        for r in repos:
            dest = wt / r.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if r.label in branched:
                git(umbrella / r.path, "worktree", "add", "-q", "-b", name, str(dest), base(r))
            else:
                git(umbrella / r.path, "worktree", "add", "-q", "--detach", str(dest), base(r))
            created.append(r)
    except subprocess.CalledProcessError as exc:
        print(f"{r.label}: {exc.stderr.strip()}; rolling back", file=sys.stderr)
        for done in reversed(created):
            git(umbrella / done.path, "worktree", "remove", "--force", str(wt / done.path))
            if done.label in branched:
                git(umbrella / done.path, "branch", "-D", name)
        git(umbrella / r.path, "worktree", "prune")
        shutil.rmtree(wt, ignore_errors=True)
        return 1

    fitace = next((r for r in repos if r.label == "fitACE"), None)
    if fitace is not None:
        for rel in BUILD_OUTPUTS:
            src = umbrella / fitace.path / rel
            dst = wt / fitace.path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, symlinks=True)
            elif src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"note: {fitace.path}/{rel} is not built in the main checkout; not copied")

    for r in repos:
        link_external_targets(umbrella, wt, wt / r.path)

    for r in repos:
        state = name if r.label in branched else "detached"
        print(f"{r.label:16} {state:24} {git(wt / r.path, 'log', '--oneline', '-1')}")

    if install:
        for r in repos:
            if r.label in branched and (wt / r.path / "pixi.toml").exists():
                print(f"pixi install --frozen in {r.path}", flush=True)
                subprocess.run(["pixi", "install", "--frozen"], cwd=wt / r.path, check=True)
    print(f"\nworktree: {wt}")
    return 0


def link_external_targets(umbrella: Path, wt: Path, checkout: Path) -> None:
    """Link the dangling targets of tracked symlinks into ``wt/external/`` back to ``umbrella``'s copies."""
    for line in git(checkout, "ls-files", "-s").splitlines():
        mode, _, _, path = line.split(maxsplit=3)
        if mode != "120000":
            continue
        link = checkout / path
        target = Path(os.path.normpath(link.parent / os.readlink(link)))
        if target.exists() or not target.is_relative_to(wt / "external"):
            continue
        source = umbrella / target.relative_to(wt)
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)


def family_entries(umbrella: Path, name: str) -> list[tuple[Repo, Path, bool]]:
    """(repo, worktree path, on the ``name`` branch) for each checkout in family worktree ``name``."""
    wt = worktree_dir(umbrella, name)
    entries = []
    for r in present(umbrella):
        for block in git(umbrella / r.path, "worktree", "list", "--porcelain").split("\n\n"):
            lines = block.splitlines()
            if f"worktree {wt / r.path}" in lines:
                entries.append((r, wt / r.path, f"branch refs/heads/{name}" in lines))
    return entries


def remove(umbrella: Path, name: str, force: bool) -> int:
    """Remove family worktree ``name`` after checking every checkout first."""
    entries = family_entries(umbrella, name)
    if not entries:
        print(f"no family worktree named {name!r}", file=sys.stderr)
        return 2

    problems, merged = [], set()
    for r, path, on_branch in entries:
        dirty = git(path, "status", "--porcelain")
        if dirty:
            problems.append(f"{r.label}: {len(dirty.splitlines())} uncommitted or untracked file(s)")
        if on_branch:
            ok = subprocess.run(
                ["git", "-C", str(umbrella / r.path), "merge-base", "--is-ancestor", name, base(r)], check=False
            )
            if ok.returncode == 0:
                merged.add(r.label)
            else:
                problems.append(f"{r.label}: branch {name!r} is not merged into {base(r)}")
    if problems and not force:
        print("refusing to remove (pass --force to remove the worktrees anyway):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    for r, path, on_branch in reversed(entries):
        git(umbrella / r.path, "worktree", "remove", "--force", str(path))
        if r.label in merged:
            # -D: merged into the base was checked above; -d would test the main checkout's HEAD.
            git(umbrella / r.path, "branch", "-D", name)
            print(f"{r.label}: removed worktree, deleted merged branch {name!r}")
        elif on_branch:
            print(f"{r.label}: removed worktree, kept unmerged branch {name!r}")
        else:
            print(f"{r.label}: removed worktree")
    shutil.rmtree(worktree_dir(umbrella, name), ignore_errors=True)
    if "pedigree-graph" in merged:
        print("pedigree-graph: run `pixi run build-dev` in the main checkout to rebuild the editable extension")
    return 0


def list_worktrees(umbrella: Path) -> int:
    """Print each family worktree with per-checkout branch, ahead count, and changes."""
    root = umbrella / ".claude" / "worktrees"
    names = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
    for name in names:
        print(name)
        for r, path, _ in family_entries(umbrella, name):
            branch = git(path, "branch", "--show-current") or "detached"
            ahead = git(path, "rev-list", "--count", f"{base(r)}..HEAD")
            dirty = len(git(path, "status", "--porcelain").splitlines())
            print(f"  {r.label:16} {branch:24} {ahead} ahead of {base(r)}, {dirty} changed")
    return 0


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    """CLI entry point; ``root`` is any path inside the umbrella checkout (tests pass a fake one)."""
    parser = argparse.ArgumentParser(description="Family worktrees under .claude/worktrees/<name>/.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add", help="create a family worktree")
    p_add.add_argument("name")
    p_add.add_argument("repos", nargs="*", help="checkouts that get a <name> branch (the rest are detached)")
    p_add.add_argument("--all", action="store_true", help="branch every checkout")
    p_add.add_argument("--install", action="store_true", help="pixi install --frozen in each branched checkout")
    p_rm = sub.add_parser("remove", help="remove a family worktree")
    p_rm.add_argument("name")
    p_rm.add_argument("--force", action="store_true", help="remove despite uncommitted or unmerged work")
    sub.add_parser("list", help="list family worktrees")
    args = parser.parse_args(argv)

    main_dir = main_root(root)
    if args.cmd == "add":
        branched = {r.label for r in present(main_dir)} if args.all else set(args.repos)
        return add(main_dir, args.name, branched, args.install)
    if args.cmd == "remove":
        return remove(main_dir, args.name, args.force)
    return list_worktrees(main_dir)


if __name__ == "__main__":
    raise SystemExit(main())
