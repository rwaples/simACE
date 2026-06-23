"""Consistency test: every family repo pins the one canonical ty ``python-version``.

``tools/family_repos.py`` is the single source via its ``TY_PYTHON_VERSION``
constant.  Static ty config can't import that constant, so this test enforces the
link: each present family repo's ty config -- ``[tool.ty.environment]`` in a
``pyproject.toml`` or ``[environment]`` in a standalone ``ty.toml`` -- must set
``python-version`` to exactly ``TY_PYTHON_VERSION``.  Bumping it in one repo (a
deliberate, family-wide step like a ``ty`` pin bump) and forgetting the rest
fails here, before ty silently resolves a sibling against a different Python.

Unlike ``test_ty_pin_consistency`` (pyproject-only), this also covers the two
``ty.toml``-only repos (pedsum, tetraher_simace).  Repos that are not checked out
are skipped.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from family_repos import TY_PYTHON_VERSION, python_repos  # noqa: E402  (needs the sys.path tweak above)


def _ty_python_version(repo: Path) -> str | None:
    """The ty ``python-version`` for *repo*, from pyproject or ty.toml (None if neither)."""
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text())
        env = data.get("tool", {}).get("ty", {}).get("environment", {})
        if "python-version" in env:
            return env["python-version"]
    ty_toml = repo / "ty.toml"
    if ty_toml.is_file():
        data = tomllib.loads(ty_toml.read_text())
        return data.get("environment", {}).get("python-version")
    return None


def _ty_config_repos() -> list[tuple[str, Path]]:
    """``(label, repo path)`` for every present python repo that ships a ty config."""
    cases = []
    for repo in python_repos():
        path = _ROOT / repo.path
        if (path / "pyproject.toml").is_file() or (path / "ty.toml").is_file():
            cases.append((repo.label, path))
    return cases


_CASES = _ty_config_repos()


@pytest.mark.parametrize(("label", "repo"), _CASES, ids=[label for label, _ in _CASES])
def test_ty_python_version_is_canonical(label: str, repo: Path) -> None:
    version = _ty_python_version(repo)
    assert version is not None, f"{label}: ty config declares no [environment] python-version"
    assert version == TY_PYTHON_VERSION, (
        f"{label}: ty python-version '{version}' does not match canonical '{TY_PYTHON_VERSION}' "
        f"(tools/family_repos.py TY_PYTHON_VERSION). Update the config or TY_PYTHON_VERSION."
    )
