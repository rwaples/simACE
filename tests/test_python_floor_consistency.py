"""Consistency test: each family check unit's ruff and ty targets match its Python floor.

A unit's floor is the lower bound of ``requires-python`` in its own
``pyproject.toml`` or, for a unit without one (tetraher_simace), in the nearest
enclosing one.  ruff's ``target-version`` and ty's ``python-version`` are static
config that can't read that value, so this test enforces the link: a floor bump
that forgets either tool fails here, before ruff or ty checks the unit against a
Python it no longer supports, or lets through code that breaks on one it still does.

Floors may differ between units: pedigree-graph is published on PyPI and keeps a
lower floor than the lockstep family.  Covers both ``[tool.*]`` tables in a
``pyproject.toml`` and standalone ``ruff.toml`` / ``ty.toml`` (pedsum,
tetraher_simace).  Repos that are not checked out are skipped.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from family_repos import python_repos  # noqa: E402  (needs the sys.path tweak above)


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text()) if path.is_file() else {}


def _floor(unit: Path) -> str | None:
    """``"X.Y"`` from the nearest ``requires-python = ">=X.Y"`` at or above *unit*, within the umbrella."""
    for directory in (unit, *unit.parents):
        spec = _toml(directory / "pyproject.toml").get("project", {}).get("requires-python")
        if spec is not None:
            match = re.fullmatch(r">=\s*(\d+\.\d+)", spec.strip())
            assert match, f"{directory}: requires-python '{spec}' is not a plain '>=X.Y' floor"
            return match[1]
        if directory == _ROOT:
            return None
    return None


def _tool_setting(unit: Path, tool: str, table: tuple[str, ...], key: str) -> str | None:
    """*key* from ``[tool.<tool>.<table>]`` in pyproject, else ``[<table>]`` in ``<tool>.toml``."""
    for config in (_toml(unit / "pyproject.toml").get("tool", {}).get(tool, {}), _toml(unit / f"{tool}.toml")):
        section = config
        for name in table:
            section = section.get(name, {})
        if key in section:
            return section[key]
    return None


def _units() -> list[tuple[str, Path]]:
    """``(label, path)`` for every present python unit that ships ruff or ty config."""
    cases = []
    for repo in python_repos():
        path = _ROOT / repo.path
        if any((path / name).is_file() for name in ("pyproject.toml", "ruff.toml", "ty.toml")):
            cases.append((repo.label, path))
    return cases


_CASES = _units()
_IDS = [label for label, _ in _CASES]


@pytest.mark.parametrize(("label", "unit"), _CASES, ids=_IDS)
def test_ty_python_version_matches_floor(label: str, unit: Path) -> None:
    floor = _floor(unit)
    assert floor is not None, f"{label}: no requires-python at or above {unit}"
    version = _tool_setting(unit, "ty", ("environment",), "python-version")
    assert version == floor, f"{label}: ty python-version '{version}' does not match requires-python floor '{floor}'"


@pytest.mark.parametrize(("label", "unit"), _CASES, ids=_IDS)
def test_ruff_target_version_matches_floor(label: str, unit: Path) -> None:
    floor = _floor(unit)
    assert floor is not None, f"{label}: no requires-python at or above {unit}"
    target = _tool_setting(unit, "ruff", (), "target-version")
    expected = "py" + floor.replace(".", "")
    assert target == expected, f"{label}: ruff target-version '{target}' does not match requires-python floor '{floor}'"
