#!/usr/bin/env python
"""Single source of truth for the simACE/fitACE family repo list.

Every tool that iterates the family imports its subset from here instead of
hardcoding a list, so adding/removing an entry is a one-line edit in one place:

- ``tools/typecheck_family.py`` -> ``python_repos()``  (the 12 ty/ruff/pytest check units)
- ``tools/release.py``          -> ``lockstep_repos()`` (the 3 CalVer-tagged checkouts)
- ``.agents/skills/repo-status/scripts/repo-status.sh``
                                -> ``--subset all --format lines``

Since ADR 0017 (family monorepo) an entry is not necessarily its own git
checkout: the method packages, the ``ace_iter_reml`` C++ source, and the
``tetraher_simace`` LDAK fork are subdirectories of the fitACE monorepo
(``checkout=False``) but remain independent *check units* (own pyproject with
ty/ruff/pytest config). Only five entries are checkouts: simACE, fitACE,
fitACE_epimight, pedigree-graph, pedsum.

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
TY_PIN = "0.0.70"

#: The ``python-version`` every family ``[tool.ty]`` / ``ty.toml`` config pins --
#: one shared value, enforced by ``tests/test_ty_python_version_consistency.py``.
TY_PYTHON_VERSION = "3.13"

#: The simACE umbrella root (this file lives in ``<root>/tools/``).
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Repo:
    """One family entry (a checkout or a monorepo subdirectory check unit)."""

    label: str
    """Display name and ``--repo`` / repo-status filter key (``simACE`` for ``.``)."""
    path: str
    """Path relative to :data:`ROOT` (``"."`` for simACE itself)."""
    python: bool
    """``True`` if ruff / ty / pytest apply (``False`` only for the C++ source)."""
    lockstep: bool
    """``True`` if tagged in the lockstep CalVer release (see ``tools/release.py``)."""
    checkout: bool
    """``True`` if this entry is its own git repo (``False``: fitACE subdir, ADR 0017)."""


#: Every family entry, in repo-status display order.  ``python_repos()`` is the
#: 12 ty/ruff/pytest check units (all but the C++ source); ``lockstep_repos()``
#: is the 3 CalVer-tagged checkouts (simACE, fitACE, fitACE_epimight — within
#: the fitACE monorepo all seven distributions and the binary read fitACE's
#: tag, so lockstep is structural there); ``checkout_repos()`` is the 5 git
#: checkouts.
FAMILY: tuple[Repo, ...] = (
    Repo("simACE", ".", python=True, lockstep=True, checkout=True),
    Repo("fitACE", "fitACE", python=True, lockstep=True, checkout=True),
    Repo("fitACE_epimight", "fitACE/fitACE_epimight", python=True, lockstep=True, checkout=True),
    Repo("fitACE_pcgc", "fitACE/fitACE_pcgc", python=True, lockstep=False, checkout=False),
    Repo("fitACE_iter_reml", "fitACE/fitACE_iter_reml", python=True, lockstep=False, checkout=False),
    Repo("fitACE_tetraher", "fitACE/fitACE_tetraher", python=True, lockstep=False, checkout=False),
    Repo("fitACE_pafgrs", "fitACE/fitACE_pafgrs", python=True, lockstep=False, checkout=False),
    Repo("fitACE_stan", "fitACE/fitACE_stan", python=True, lockstep=False, checkout=False),
    Repo("fitACE_frailty", "fitACE/fitACE_frailty", python=True, lockstep=False, checkout=False),
    Repo("ace_iter_reml", "fitACE/fitACE_iter_reml/ace_iter_reml", python=False, lockstep=False, checkout=False),
    Repo("tetraher_simace", "fitACE/tetraher_simace", python=True, lockstep=False, checkout=False),
    Repo("pedigree-graph", "external/pedigree-graph", python=True, lockstep=False, checkout=True),
    Repo("pedsum", "external/pedsum", python=True, lockstep=False, checkout=True),
)


def all_repos() -> tuple[Repo, ...]:
    """Every family entry (13), in display order."""
    return FAMILY


def python_repos() -> tuple[Repo, ...]:
    """The 12 check units ty / ruff / pytest apply to (excludes the C++ source)."""
    return tuple(r for r in FAMILY if r.python)


def lockstep_repos() -> tuple[Repo, ...]:
    """The 3 checkouts tagged together in a lockstep CalVer release."""
    return tuple(r for r in FAMILY if r.lockstep)


def checkout_repos() -> tuple[Repo, ...]:
    """The 5 entries that are their own git repos (ADR 0017 world)."""
    return tuple(r for r in FAMILY if r.checkout)


_SUBSETS = {"all": all_repos, "python": python_repos, "lockstep": lockstep_repos, "checkouts": checkout_repos}


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
