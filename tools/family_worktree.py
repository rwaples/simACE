#!/usr/bin/env python
"""Create, list, and remove a family worktree: one git worktree per family checkout.

A family worktree lives at ``<main>/.claude/worktrees/<name>/`` and mirrors the
umbrella layout, so cross-repo paths (``fitACE/pixi.toml``'s ``simace = {path =
".."}``, pedsum's ty ``extra-paths = ../pedigree-graph``) resolve inside it.
The checkouts named on the command line get a ``<name>`` branch off their base
branch (``dev`` for simACE, ``main`` elsewhere); every other checkout gets a
detached worktree at its base, so nothing in the tree is shared with another
session.

A name is one path component matching ``[A-Za-z0-9][A-Za-z0-9._-]*`` that git
accepts as a branch name (so no ``feat/x``, ``a..b``, ``x.lock``); the family
directory must resolve inside ``.claude/worktrees/`` even through symlinks.

``add`` reserves the family directory first (two concurrent ``add``s of one name
cannot both proceed), creates the worktrees, then restores what git does not
carry:

- fitACE's gitignored compiled binaries (``BUILD_OUTPUTS``), copied from the
  main checkout so the ace_iter_reml, pcgc, stan, and tetraher units run;
- the targets of tracked symlinks into ``external/`` (fitACE_epimight's
  ``external/epimight*``), linked back to the main checkout's copies, which are
  read-only references. Other dangling targets stay dangling: fitACE's
  ``results -> ../results`` must reach the worktree's own simACE results, not
  the main checkout's.

If any of that fails or is interrupted, ``add`` removes what it created, in
reverse order, and deletes the family directory only once nothing registered
survives; otherwise it names every survivor and how to finish by hand, and the
name stays taken. ``--install`` runs after creation is complete: a failed
``pixi install --frozen`` keeps the finished worktree and prints the command to
rerun (exit 1).

``remove`` checks every checkout before touching any: it refuses when a
worktree has uncommitted or untracked files or its HEAD is not merged into the
base, whatever branch it is on, unless ``--force``. A git error during a check
is a failure, not a pass. It deletes a merged ``<name>`` branch only; an
unmerged or differently named branch is kept, and an unmerged detached HEAD is
saved under ``worktree-rescue-<name>-<sha12>`` in the main checkout before
anything is removed (``--force`` discards uncommitted files, never commits).

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
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
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

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class Entry:
    """One registered checkout of a family worktree."""

    repo: Repo
    path: Path
    head: str
    branch: str | None
    """Checked-out branch name, ``None`` when detached."""


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return stripped stdout; raise on failure."""
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def git_z(cwd: Path, *args: str) -> list[bytes]:
    """Run a ``-z`` git command in ``cwd`` and return its NUL-terminated records, verbatim."""
    out = subprocess.run(["git", "-C", str(cwd), *args, "-z"], check=True, capture_output=True).stdout
    return out.split(b"\0")[:-1]


def main_root(start: Path = ROOT) -> Path:
    """The main simACE checkout, even when this script runs from a worktree."""
    return Path(git(start, "rev-parse", "--path-format=absolute", "--git-common-dir")).parent


def present(umbrella: Path) -> list[Repo]:
    """Family checkouts present under ``umbrella``, outermost first."""
    return [r for r in checkout_repos() if (umbrella / r.path / ".git").exists()]


def worktree_dir(umbrella: Path, name: str) -> Path:
    """Where family worktree ``name`` lives; ``ValueError`` for a bad name or a path outside the root."""
    root = umbrella / ".claude" / "worktrees"
    check = ["git", "check-ref-format", "--branch", name]
    if not NAME_RE.fullmatch(name) or subprocess.run(check, check=False, capture_output=True).returncode:
        raise ValueError(f"invalid worktree name {name!r}: want [A-Za-z0-9][A-Za-z0-9._-]* and a valid branch name")
    wt = root / name
    if not wt.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{wt} resolves outside {root}")
    return wt


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
    try:
        wt = worktree_dir(umbrella, name)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    for r in repos:
        if r.label in branched and git(umbrella / r.path, "branch", "--list", name):
            print(f"{r.label}: branch {name!r} already exists", file=sys.stderr)
            return 2
    wt.parent.mkdir(parents=True, exist_ok=True)
    try:
        wt.mkdir()
    except FileExistsError:
        print(f"{wt} already exists", file=sys.stderr)
        return 2

    created: list[Repo] = []
    try:
        create(umbrella, wt, name, repos, branched, created)
    except (Exception, KeyboardInterrupt) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else repr(exc)
        print(f"{detail}; rolling back", file=sys.stderr)
        failures = rollback(umbrella, wt, name, created, branched)
        if failures:
            print("rollback incomplete; finish it by hand before reusing the name:", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
        return 1

    for r in repos:
        state = name if r.label in branched else "detached"
        print(f"{r.label:16} {state:24} {git(wt / r.path, 'log', '--oneline', '-1')}")
    print(f"\nworktree: {wt}", flush=True)

    if install:
        for r in repos:
            if r.label in branched and (wt / r.path / "pixi.toml").exists():
                print(f"pixi install --frozen in {r.path}", flush=True)
                try:
                    failed = (
                        subprocess.run(["pixi", "install", "--frozen"], cwd=wt / r.path, check=False).returncode != 0
                    )
                except KeyboardInterrupt:
                    failed = True
                if failed:
                    print(
                        f"{r.label}: pixi install failed; the worktree is complete, rerun:\n"
                        f"  cd {wt / r.path} && pixi install --frozen",
                        file=sys.stderr,
                    )
                    return 1
    return 0


def create(umbrella: Path, wt: Path, name: str, repos: list[Repo], branched: set[str], created: list[Repo]) -> None:
    """Register every checkout's worktree under ``wt`` (appending to ``created``), then restore ignored inputs."""
    for r in repos:
        dest = wt / r.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if r.label in branched:
            git(umbrella / r.path, "worktree", "add", "-q", "-b", name, str(dest), base(r))
        else:
            git(umbrella / r.path, "worktree", "add", "-q", "--detach", str(dest), base(r))
        created.append(r)

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


def rollback(umbrella: Path, wt: Path, name: str, created: list[Repo], branched: set[str]) -> list[str]:
    """Undo a failed ``create``: innermost first, skipping a parent whose child survives. Returns the failures."""
    failures: list[str] = []
    survivors: list[Path] = []
    for r in reversed(created):
        path = wt / r.path
        if any(s.is_relative_to(path) for s in survivors):
            failures.append(f"{r.label}: kept {path}, a surviving worktree is nested in it")
            survivors.append(path)
            continue
        try:
            git(umbrella / r.path, "worktree", "remove", "--force", str(path))
        except subprocess.CalledProcessError as exc:
            failures.append(f"{r.label}: {exc.stderr.strip()}; worktree {path} and its branch are kept")
            survivors.append(path)
            continue
        if r.label in branched:
            try:
                git(umbrella / r.path, "branch", "-D", name)
            except subprocess.CalledProcessError as exc:
                failures.append(f"{r.label}: {exc.stderr.strip()}; worktree removed, branch {name!r} kept")
    for r in set(present(umbrella)) - set(created):
        # A failed `worktree add` can leave a half-registered entry or the new branch behind. The branch
        # did not exist before this run; delete it only while it still holds nothing beyond the base.
        main = umbrella / r.path
        try:
            git(main, "worktree", "prune")
            if r.label in branched and git(main, "branch", "--list", name):
                if git(main, "rev-parse", name) != git(main, "rev-parse", base(r)):
                    failures.append(f"{r.label}: branch {name!r} has moved off {base(r)}; kept")
                else:
                    git(main, "branch", "-D", name)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{r.label}: {exc.stderr.strip()}")
    if not failures:
        shutil.rmtree(wt, ignore_errors=True)
    return failures


def link_external_targets(umbrella: Path, wt: Path, checkout: Path) -> None:
    """Link the dangling targets of tracked symlinks into ``wt/external/`` back to ``umbrella``'s copies."""
    for record in git_z(checkout, "ls-files", "--stage"):
        meta, path = record.split(b"\t", 1)
        if not meta.startswith(b"120000 "):
            continue
        link = checkout / os.fsdecode(path)
        target = Path(os.path.normpath(link.parent / os.readlink(link)))
        if target.exists() or not target.is_relative_to(wt / "external"):
            continue
        source = umbrella / target.relative_to(wt)
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)


def family_entries(umbrella: Path, name: str) -> list[Entry]:
    """The registered checkouts of family worktree ``name``, outermost first."""
    wt = worktree_dir(umbrella, name)
    entries = []
    for r in present(umbrella):
        block: dict[bytes, bytes] = {}
        for record in [*git_z(umbrella / r.path, "worktree", "list", "--porcelain"), b""]:
            if record:
                key, _, value = record.partition(b" ")
                block[key] = value
                continue
            if block.get(b"worktree") == os.fsencode(wt / r.path):
                branch = block.get(b"branch", b"").decode().removeprefix("refs/heads/") or None
                entries.append(Entry(r, wt / r.path, block[b"HEAD"].decode(), branch))
            block = {}
    return entries


def is_ancestor(cwd: Path, commit: str, ref: str) -> bool:
    """Whether ``commit`` is reachable from ``ref``; a git error raises rather than answering."""
    cmd = ["git", "-C", str(cwd), "merge-base", "--is-ancestor", commit, ref]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)
    return proc.returncode == 0


def remove(umbrella: Path, name: str, force: bool) -> int:
    """Remove family worktree ``name`` after checking every checkout first."""
    try:
        entries = family_entries(umbrella, name)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not entries:
        print(f"no family worktree named {name!r}", file=sys.stderr)
        return 2

    problems, merged, rescue = [], set(), []
    for e in entries:
        dirty = git(e.path, "status", "--porcelain")
        if dirty:
            problems.append(f"{e.repo.label}: {len(dirty.splitlines())} uncommitted or untracked file(s)")
        if is_ancestor(umbrella / e.repo.path, e.head, base(e.repo)):
            if e.branch == name:
                merged.add(e.repo.label)
        elif e.branch is not None:
            problems.append(f"{e.repo.label}: branch {e.branch!r} is not merged into {base(e.repo)}")
        else:
            problems.append(
                f"{e.repo.label}: detached HEAD {e.head[:12]} is not merged into {base(e.repo)}; "
                f"save it with `git -C {e.path} switch -c <branch>`"
            )
            rescue.append(e)
    if problems and not force:
        print("refusing to remove (pass --force to remove the worktrees anyway):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    for e in rescue:
        ref = f"worktree-rescue-{name}-{e.head[:12]}"
        main = umbrella / e.repo.path
        if git(main, "branch", "--list", ref):
            if git(main, "rev-parse", ref) != e.head:
                print(f"{e.repo.label}: branch {ref!r} exists and points elsewhere; nothing removed", file=sys.stderr)
                return 1
        else:
            git(main, "branch", ref, e.head)
        print(f"{e.repo.label}: saved detached HEAD as branch {ref!r}")

    for e in reversed(entries):
        git(umbrella / e.repo.path, "worktree", "remove", "--force", str(e.path))
        if e.repo.label in merged:
            # -D: merged into the base was checked above; -d would test the main checkout's HEAD.
            git(umbrella / e.repo.path, "branch", "-D", name)
            print(f"{e.repo.label}: removed worktree, deleted merged branch {name!r}")
        elif e.branch is not None:
            print(f"{e.repo.label}: removed worktree, kept branch {e.branch!r}")
        else:
            print(f"{e.repo.label}: removed worktree")
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
        try:
            entries = family_entries(umbrella, name)
        except ValueError as exc:
            print(f"  {exc}")
            continue
        for e in entries:
            ahead = git(e.path, "rev-list", "--count", f"{base(e.repo)}..HEAD")
            dirty = len(git(e.path, "status", "--porcelain").splitlines())
            print(f"  {e.repo.label:16} {e.branch or 'detached':24} {ahead} ahead of {base(e.repo)}, {dirty} changed")
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
