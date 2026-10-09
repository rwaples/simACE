#!/usr/bin/env python
"""Cut a lockstep family release: tag the three simACE/fitACE checkouts at one SemVer.

Run from anywhere (repo paths resolve relative to this file's location).  The
helper creates an annotated git tag ``vMAJOR.MINOR.PATCH`` in each of the three
lockstep family repos **locally**, then PRINTS the per-repo ``git push``
commands for the maintainer to run.  It never pushes — that is the maintainer's
job, per the repo-wide no-``git push`` rule.

The three members (simACE, the fitACE monorepo, fitACE_epimight — ADR 0017)
are tagged all-or-nothing: the helper refuses to tag anything unless *every*
repo is present, has a clean working tree, and is not already tagged at the
requested version.  If a tag creation fails partway, the tags already created
in this run are rolled back so the family stays consistent.  ``--dry-run`` runs
the same checks and prints the would-tag / would-push actions without creating
tags.

setuptools-scm reads *local* tags, so the runtime version / ``FAMILY_FLOOR``
guard clears as soon as the local tags exist + the family is reinstalled — the
push is only needed to publish.  See simACE ADR 0012 (lockstep family
versioning) and ADR 0023 (the SemVer scheme).

``--repo`` tags one independently versioned checkout instead (pedigree-graph
or pg-phenotype), at its own version.  It refuses unless that checkout is clean and untagged,
every file that states its version agrees with the tag, and ``CHANGELOG.md``
has the tag's ``## vX.Y.Z`` section, which its publish workflow turns into
the release notes.  The tag push publishes it.

Examples:
    python tools/release.py --next             # print the next patch and minor tags
    python tools/release.py v0.1.0             # tag the three checkouts locally
    python tools/release.py v0.1.0 --dry-run   # check + report, tag nothing
    python tools/release.py v0.1.1 -m "fix: ..."
    python tools/release.py --repo pg-phenotype v0.2.0 --dry-run
    python tools/release.py --repo pedigree-graph v0.12.3 -m "pedigree-graph 0.12.3: ..."
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from family_repos import all_repos, lockstep_repos

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The three lockstep checkouts, relative to the simACE root (ADR 0017):
#: simACE, the fitACE monorepo (whose seven distributions + C++ binary all
#: read its tag), and fitACE_epimight.  Sourced from the shared
#: ``family_repos`` manifest so the repo list lives in exactly one place.
FAMILY_REPOS: tuple[str, ...] = tuple(repo.path for repo in lockstep_repos())

#: The independently versioned checkouts ``--repo`` can tag, by family label,
#: each with the files that state its version: a TOML key path, or a DCF field
#: for an R ``DESCRIPTION``.  Every one must equal the tag.  Both state it in
#: the Rust workspace, the R binding crate and the R package.
_RUST_AND_R_VERSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Cargo.toml", ("workspace", "package", "version")),
    ("r/src/rust/Cargo.toml", ("package", "version")),
    ("r/DESCRIPTION", ("Version",)),
)
INDEPENDENT_VERSIONS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "pedigree-graph": _RUST_AND_R_VERSIONS,
    "pg-phenotype": _RUST_AND_R_VERSIONS,
}

_SIMACE_ROOT = Path(__file__).resolve().parent.parent

_TAG_RE = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)")

#: Majors from here up are CalVer-era tags (``v2026.09.2``), never family SemVer.
_CALVER_ERA_MAJOR = 2000


def tag_error(tag: str) -> str | None:
    """Why *tag* is not a family SemVer tag (ADR 0023), or ``None`` if it is one."""
    match = _TAG_RE.fullmatch(tag)
    if match is None:
        return "must look like vMAJOR.MINOR.PATCH, e.g. v0.1.0"
    if any(len(part) > 1 and part.startswith("0") for part in match.groups()):
        return "has a leading zero"
    if int(match[1]) >= _CALVER_ERA_MAJOR:
        return "is a CalVer-era tag, not a SemVer family tag"
    return None


def parse_family_tag(tag: str) -> tuple[int, int, int] | None:
    """``(major, minor, patch)`` for a family SemVer tag, ``None`` for anything else."""
    if tag_error(tag) is not None:
        return None
    major, minor, patch = (int(part) for part in tag[1:].split("."))
    return major, minor, patch


def next_versions(tags: Iterable[str]) -> tuple[str, str]:
    """The next ``(patch, minor)`` tags after the highest family tag in *tags*.

    Before 1.0 the minor is the breaking digit (ADR 0023).  With no family tag
    yet, both candidates are ``v0.1.0``.
    """
    versions = [v for v in map(parse_family_tag, tags) if v is not None]
    if not versions:
        return "v0.1.0", "v0.1.0"
    major, minor, patch = max(versions)
    return f"v{major}.{minor}.{patch + 1}", f"v{major}.{minor + 1}.0"


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


def check_repos(tag: str, repos: Iterable[str] = FAMILY_REPOS) -> list[tuple[str, str]]:
    """Return ``(repo, reason)`` pairs for every repo in *repos* that can't be tagged.

    A repo is not ready if it is missing / not a git work tree, has a dirty
    working tree, or is already tagged at *tag*.  An empty list means the family
    is ready for an all-or-nothing tag.
    """
    problems: list[tuple[str, str]] = []
    for rel in repos:
        repo = (_SIMACE_ROOT / rel).resolve()
        if not _is_git_repo(repo):
            problems.append((rel, "not a git repo (missing checkout?)"))
            continue
        if _is_dirty(repo):
            problems.append((rel, "working tree is dirty (commit or stash first)"))
        if _has_tag(repo, tag):
            problems.append((rel, f"already tagged {tag}"))
    return problems


def stated_version(path: Path, key: tuple[str, ...]) -> str | None:
    """The version *path* states at *key* (TOML key path or DCF field), or ``None``."""
    if not path.is_file():
        return None
    if path.suffix == ".toml":
        node = tomllib.loads(path.read_text())
        for part in key:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node if isinstance(node, str) else None
    match = re.search(rf"^{re.escape(key[0])}:[ \t]*(\S+)", path.read_text(), re.MULTILINE)
    return match[1] if match else None


def check_independent(label: str, tag: str) -> list[tuple[str, str]]:
    """``(file, reason)`` pairs for every way *label*'s checkout disagrees with *tag*."""
    rel = next(r.path for r in all_repos() if r.label == label)
    repo = _SIMACE_ROOT / rel
    problems = check_repos(tag, [rel])
    for file, key in INDEPENDENT_VERSIONS[label]:
        version = stated_version(repo / file, key)
        if version != tag[1:]:
            problems.append((f"{rel}/{file}", f"states version {version!r}, not {tag[1:]!r}"))
    changelog = repo / "CHANGELOG.md"
    lines = changelog.read_text().splitlines() if changelog.is_file() else []
    if not any(line == f"## {tag}" or line.startswith(f"## {tag} ") for line in lines):
        problems.append((f"{rel}/CHANGELOG.md", f"has no '## {tag}' section"))
    return problems


def _push_commands(tag: str, repos: Iterable[str] = FAMILY_REPOS) -> list[str]:
    """The per-repo ``git push`` commands the maintainer runs to publish *tag*."""
    return [f"git -C {(_SIMACE_ROOT / rel).resolve()} push origin {tag}" for rel in repos]


def main(argv: list[str] | None = None) -> int:
    """Parse args, verify the repos, tag them locally, print pushes."""
    parser = argparse.ArgumentParser(
        prog="release.py",
        description=(
            "Tag the three lockstep simACE/fitACE checkouts at one SemVer, or with --repo one "
            "independently versioned checkout (never pushes)."
        ),
    )
    parser.add_argument("version", nargs="?", help="Release tag, e.g. v0.1.0")
    parser.add_argument(
        "--repo",
        choices=tuple(INDEPENDENT_VERSIONS),
        default=None,
        help="Tag this independently versioned checkout alone, at its own version.",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="Print the next patch and minor tags after the highest tag (simACE's, or --repo's); tag nothing.",
    )
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

    repos = FAMILY_REPOS
    if args.repo is not None:
        repos = (next(r.path for r in all_repos() if r.label == args.repo),)

    if args.next:
        if args.version is not None:
            parser.error("--next takes no version")
        patch, minor = next_versions(_git(_SIMACE_ROOT / repos[0], "tag", "--list").stdout.split())
        print(f"next patch: {patch}")
        print(f"next minor: {minor}  (breaks the CLI/config or result-file contract)")
        return 0
    if args.version is None:
        parser.error("a version (e.g. v0.1.0) or --next is required")

    tag = args.version
    if (reason := tag_error(tag)) is not None:
        parser.error(f"version {tag!r} {reason}")
    if args.repo is None:
        message = args.message or f"Lockstep family release {tag}"
        problems = check_repos(tag)
        ready = f"all {len(repos)} family repos are clean and untagged."
    else:
        message = args.message or f"{args.repo} {tag[1:]}"
        problems = check_independent(args.repo, tag)
        ready = f"{args.repo} is clean, untagged, and at version {tag[1:]} with a CHANGELOG section."

    # 1. All-or-nothing readiness check across every repo to tag.
    if problems:
        print(f"Refusing to tag {tag}: {len(problems)} problem(s):", file=sys.stderr)
        for rel, reason in problems:
            print(f"  - {rel}: {reason}", file=sys.stderr)
        return 1

    abspaths = [(rel, (_SIMACE_ROOT / rel).resolve()) for rel in repos]

    # 2. Tag (or, in dry-run, just report).
    if args.dry_run:
        print(f"[dry-run] {ready}")
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
        print(f"\nCreated {tag} in {len(abspaths)} repo(s) (local tags only).")

    # 3. Print the push commands.  This helper NEVER pushes.
    if args.dry_run:
        print("\n[dry-run] after a real tag, push with:")
    else:
        print("\nPush the tags yourself (this helper never pushes):")
    for cmd in _push_commands(tag, repos):
        print(f"  {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
