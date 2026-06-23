#!/usr/bin/env python
"""Single source of truth for the simACE/fitACE family repo list.

Every tool that iterates the family imports its subset from here instead of
hardcoding a list, so adding/removing a repo is a one-line edit in one place:

- ``tools/typecheck_family.py`` -> ``python_repos()``  (the 12 ty/ruff/pytest repos)
- ``tools/release.py``          -> ``lockstep_repos()`` (the 10 CalVer-tagged repos)
- ``.agents/skills/repo-status/scripts/repo-status.sh``
                                -> ``--subset all --format lines`` (all 13)

Paths are relative to the simACE umbrella root (this file's grandparent), with
``"."`` for simACE itself.  ``TY_PIN`` is the one ``ty`` version every family
``typecheck`` extra must pin -- enforced by ``tests/test_ty_pin_consistency.py``.

CLI (for the bash consumer):
    python tools/family_repos.py --subset all       # label|path per line
    python tools/family_repos.py --subset python
    python tools/family_repos.py --subset lockstep
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

#: The ``ty`` version pinned in every family ``typecheck`` extra.
TY_PIN = "0.0.51"

#: The simACE umbrella root (this file lives in ``<root>/tools/``).
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Repo:
    """One family checkout, located relative to the simACE umbrella root."""

    label: str
    """Display name and ``--repo`` / repo-status filter key (``simACE`` for ``.``)."""
    path: str
    """Path relative to :data:`ROOT` (``"."`` for simACE itself)."""
    python: bool
    """``True`` if ruff / ty / pytest apply (``False`` only for the C++ binary)."""
    lockstep: bool
    """``True`` if tagged in the lockstep CalVer release (see ``tools/release.py``)."""


#: Every family checkout, in repo-status display order.  The derived subsets
#: reproduce exactly the lists the tools used to hardcode: ``python_repos()`` is
#: the 12 ty/ruff/pytest repos (all but the C++ binary); ``lockstep_repos()`` is
#: the 10 CalVer-tagged repos (simACE + fitACE + 7 sisters + the binary).
FAMILY: tuple[Repo, ...] = (
    Repo("simACE", ".", python=True, lockstep=True),
    Repo("fitACE", "fitACE", python=True, lockstep=True),
    Repo("fitACE_epimight", "fitACE/fitACE_epimight", python=True, lockstep=True),
    Repo("fitACE_pcgc", "fitACE/fitACE_pcgc", python=True, lockstep=True),
    Repo("fitACE_iter_reml", "fitACE/fitACE_iter_reml", python=True, lockstep=True),
    Repo("fitACE_tetraher", "fitACE/fitACE_tetraher", python=True, lockstep=True),
    Repo("fitACE_pafgrs", "fitACE/fitACE_pafgrs", python=True, lockstep=True),
    Repo("fitACE_stan", "fitACE/fitACE_stan", python=True, lockstep=True),
    Repo("fitACE_frailty", "fitACE/fitACE_frailty", python=True, lockstep=True),
    Repo("ace_iter_reml", "fitACE/fitACE_iter_reml/ace_iter_reml", python=False, lockstep=True),
    Repo("tetraher_simace", "external/tetraher_simace", python=True, lockstep=False),
    Repo("pedigree-graph", "external/pedigree-graph", python=True, lockstep=False),
    Repo("pedsum", "external/pedsum", python=True, lockstep=False),
)


def all_repos() -> tuple[Repo, ...]:
    """Every family checkout (13), in display order."""
    return FAMILY


def python_repos() -> tuple[Repo, ...]:
    """The 12 repos ty / ruff / pytest apply to (excludes the C++ binary)."""
    return tuple(r for r in FAMILY if r.python)


def lockstep_repos() -> tuple[Repo, ...]:
    """The 10 repos tagged together in a lockstep CalVer release."""
    return tuple(r for r in FAMILY if r.lockstep)


_SUBSETS = {"all": all_repos, "python": python_repos, "lockstep": lockstep_repos}


def main(argv: list[str] | None = None) -> int:
    """Emit a subset of the family as ``label|path`` lines for shell consumers."""
    parser = argparse.ArgumentParser(description="Emit the simACE/fitACE family repo list.")
    parser.add_argument("--subset", choices=tuple(_SUBSETS), default="all")
    args = parser.parse_args(argv)
    for repo in _SUBSETS[args.subset]():
        print(f"{repo.label}|{repo.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
