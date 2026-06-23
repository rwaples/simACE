"""Consistency test: every family pyproject pins the one canonical ``ty``.

``tools/family_repos.py`` (the shared family manifest) is the single source of
the ``ty`` pin via its ``TY_PIN`` constant.  Static ``pyproject.toml`` metadata
can't import that constant, so this test enforces the link: each family repo that
ships a ``pyproject.toml`` declaring a ``ty`` requirement must pin exactly
``ty==TY_PIN``.  Bumping the pin in one repo and forgetting the rest fails here.

The two ``ty.toml``-only repos (pedsum, tetraher_simace) declare no Python
dependency metadata, so they have no pin to check and are out of scope.  In a
standalone simACE checkout without the nested repos, only simACE's own pyproject
is discovered and checked -- acceptable, same as ``test_dependency_floors``.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from family_repos import TY_PIN, python_repos  # noqa: E402  (needs the sys.path tweak above)


def _pyproject_repos() -> list[tuple[str, Path]]:
    """``(label, pyproject path)`` for every python family repo that ships one."""
    cases = []
    for repo in python_repos():
        pyproject = _ROOT / repo.path / "pyproject.toml"
        if pyproject.is_file():
            cases.append((repo.label, pyproject))
    return cases


def _ty_pin(pyproject: Path) -> str | None:
    """The ``==`` version of the ``ty`` requirement in any extra, if pinned."""
    data = tomllib.loads(pyproject.read_text())
    for group in data.get("project", {}).get("optional-dependencies", {}).values():
        for dep in group:
            req = Requirement(dep)
            if req.name == "ty":
                return next((spec.version for spec in req.specifier if spec.operator == "=="), None)
    return None


_CASES = _pyproject_repos()


@pytest.mark.parametrize(("label", "pyproject"), _CASES, ids=[label for label, _ in _CASES])
def test_ty_pin_is_canonical(label: str, pyproject: Path) -> None:
    pin = _ty_pin(pyproject)
    assert pin is not None, f"{label}: {pyproject} declares no 'ty==' pin in any optional-dependencies group"
    assert pin == TY_PIN, (
        f"{label}: 'ty=={pin}' does not match canonical 'ty=={TY_PIN}' "
        f"(tools/family_repos.py TY_PIN). Update the pin or TY_PIN."
    )
