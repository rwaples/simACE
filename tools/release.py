#!/usr/bin/env python
"""Cut a lockstep family release: tag all ten simACE/fitACE repos at one CalVer.

Run from anywhere (repo paths resolve relative to this file's location).  The
helper creates an annotated git tag ``vYYYY.MM[.patch]`` in each of the ten
lockstep family repos **locally**, then PRINTS the per-repo ``git push``
commands for the maintainer to run.  It never pushes — that is the maintainer's
job, per the repo-wide no-``git push`` rule.

The ten members (nine Python repos + the nested ``ace_iter_reml`` C++ binary)
are tagged all-or-nothing: the helper refuses to tag anything unless *every*
repo is present, has a clean working tree, and is not already tagged at the
requested version.  If a tag creation fails partway, the tags already created
in this run are rolled back so the family stays consistent.  ``--dry-run`` runs
the same checks and prints the would-tag / would-push actions without creating
tags.

setuptools-scm reads *local* tags, so the runtime version / ``FAMILY_FLOOR``
guard clears as soon as the local tags exist + the family is reinstalled — the
push is only needed to publish.  See simACE ADR 0012 (lockstep family
versioning) and the Cutover section of the implementation plan.

Examples:
    python tools/release.py v2026.06            # tag all ten repos locally
    python tools/release.py v2026.06 --dry-run  # check + report, tag nothing
    python tools/release.py v2026.06.1 -m "hotfix: ..."
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: The ten lockstep family repos, relative to the simACE root (this file's
#: grandparent): nine Python repos (simACE, fitACE core, the seven ``fitACE_*``
#: sisters) plus the nested ``ace_iter_reml`` C++ binary.
FAMILY_REPOS: tuple[str, ...] = (
    ".",
    "fitACE",
    "fitACE/fitACE_epimight",
    "fitACE/fitACE_pcgc",
    "fitACE/fitACE_iter_reml",
    "fitACE/fitACE_tetraher",
    "fitACE/fitACE_pafgrs",
    "fitACE/fitACE_stan",
    "fitACE/fitACE_frailty",
    "fitACE/fitACE_iter_reml/ace_iter_reml",
)

_SIMACE_ROOT = Path(__file__).resolve().parent.parent

#: ``vYYYY.MM`` with an optional ``.patch`` (e.g. ``v2026.06`` / ``v2026.06.1``).
_TAG_RE = re.compile(r"^v\d{4}\.\d{2}(\.\d+)?$")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``git -C <repo> <args>`` and capture output (never raises)."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _is_git_repo(repo: Path) -> bool:
    """True if *repo* is an existing git work tree."""
    if not repo.is_dir():
        return False
    result = _git(repo, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == "true"


def _is_dirty(repo: Path) -> bool:
    """True if *repo* has uncommitted changes or untracked (non-ignored) files."""
    return bool(_git(repo, "status", "--porcelain").stdout.strip())


def _has_tag(repo: Path, tag: str) -> bool:
    """True if *repo* already has a tag named exactly *tag*."""
    return bool(_git(repo, "tag", "--list", tag).stdout.strip())


def check_repos(tag: str) -> list[tuple[str, str]]:
    """Return ``(repo, reason)`` pairs for every family repo that can't be tagged.

    A repo is not ready if it is missing / not a git work tree, has a dirty
    working tree, or is already tagged at *tag*.  An empty list means the family
    is ready for an all-or-nothing tag.
    """
    problems: list[tuple[str, str]] = []
    for rel in FAMILY_REPOS:
        repo = (_SIMACE_ROOT / rel).resolve()
        if not _is_git_repo(repo):
            problems.append((rel, "not a git repo (missing checkout?)"))
            continue
        if _is_dirty(repo):
            problems.append((rel, "working tree is dirty (commit or stash first)"))
        if _has_tag(repo, tag):
            problems.append((rel, f"already tagged {tag}"))
    return problems


def _push_commands(tag: str) -> list[str]:
    """The per-repo ``git push`` commands the maintainer runs to publish *tag*."""
    return [f"git -C {(_SIMACE_ROOT / rel).resolve()} push origin {tag}" for rel in FAMILY_REPOS]


def main(argv: list[str] | None = None) -> int:
    """Parse args, verify the family, tag all ten repos locally, print pushes."""
    parser = argparse.ArgumentParser(
        prog="release.py",
        description="Tag the ten lockstep simACE/fitACE repos at one CalVer (never pushes).",
    )
    parser.add_argument("version", help="Release tag, e.g. v2026.06 or v2026.06.1")
    parser.add_argument(
        "-m",
        "--message",
        default=None,
        help="Annotated-tag message (default: 'Lockstep family release <tag>').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the readiness checks and print would-tag / would-push actions; tag nothing.",
    )
    args = parser.parse_args(argv)

    tag = args.version
    if not _TAG_RE.match(tag):
        parser.error(f"version {tag!r} must look like vYYYY.MM or vYYYY.MM.patch")
    message = args.message or f"Lockstep family release {tag}"

    # 1. All-or-nothing readiness check across every family member.
    problems = check_repos(tag)
    if problems:
        print(f"Refusing to tag {tag}: {len(problems)} repo(s) not ready:", file=sys.stderr)
        for rel, reason in problems:
            print(f"  - {rel}: {reason}", file=sys.stderr)
        return 1

    abspaths = [(rel, (_SIMACE_ROOT / rel).resolve()) for rel in FAMILY_REPOS]

    # 2. Tag (or, in dry-run, just report).
    if args.dry_run:
        print(f"[dry-run] all {len(abspaths)} family repos are clean and untagged.")
        print(f"[dry-run] would create annotated tag {tag} (message: {message!r}) in:")
        for _rel, repo in abspaths:
            print(f"  would tag:  git -C {repo} tag -a {tag} -m {message!r}")
    else:
        created: list[tuple[str, Path]] = []
        for rel, repo in abspaths:
            result = _git(repo, "tag", "-a", tag, "-m", message)
            if result.returncode != 0:
                print(f"ERROR tagging {rel}: {result.stderr.strip()}", file=sys.stderr)
                for crel, crepo in created:
                    _git(crepo, "tag", "-d", tag)
                    print(f"  rolled back {tag} in {crel}", file=sys.stderr)
                return 2
            created.append((rel, repo))
            print(f"tagged {rel} -> {tag}")
        print(f"\nCreated {tag} in all {len(abspaths)} repos (local tags only).")

    # 3. Print the push commands.  This helper NEVER pushes.
    if args.dry_run:
        print("\n[dry-run] after a real tag, push with:")
    else:
        print("\nPush the tags yourself (this helper never pushes):")
    for cmd in _push_commands(tag):
        print(f"  {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
