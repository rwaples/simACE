"""Consistency test: the documented pixi version lies inside every family ``requires-pixi``.

``docs/getting-started/installation.md`` pins the version the pixi installer
fetches (``PIXI_VERSION=vX.Y.Z``) and the version ``pixi self-update`` moves an
existing install to.  ``pixi.toml`` states the range the manifest accepts in
``requires-pixi``.  Neither file can read the other, so this test enforces the
link: every exact version the doc names must satisfy simACE's ``requires-pixi``,
and every family checkout that ships a ``pixi.toml`` must carry the same range.
Bumping one and forgetting the other fails here (issue #26).

In a standalone simACE checkout without the nested repos, only simACE's own
manifest is discovered and checked -- acceptable, same as ``test_ty_pin_consistency``.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from family_repos import checkout_repos  # noqa: E402  (needs the sys.path tweak above)

_INSTALL_DOC = _ROOT / "docs" / "getting-started" / "installation.md"
_DOC_VERSION = re.compile(r"(?:PIXI_VERSION=|self-update --version )v(\d+\.\d+\.\d+)")


def _requires_pixi(pixi_toml: Path) -> str:
    return tomllib.loads(pixi_toml.read_text())["workspace"]["requires-pixi"]


def _manifests() -> list[tuple[str, Path]]:
    """``(label, pixi.toml path)`` for every family checkout that ships one."""
    return [
        (repo.label, _ROOT / repo.path / "pixi.toml")
        for repo in checkout_repos()
        if (_ROOT / repo.path / "pixi.toml").is_file()
    ]


_MANIFESTS = _manifests()
_SIMACE_RANGE = _requires_pixi(_ROOT / "pixi.toml")


def test_install_doc_names_a_pixi_version() -> None:
    versions = set(_DOC_VERSION.findall(_INSTALL_DOC.read_text()))
    assert versions, f"{_INSTALL_DOC}: no 'PIXI_VERSION=vX.Y.Z' or 'self-update --version vX.Y.Z' found"
    assert len(versions) == 1, f"{_INSTALL_DOC}: names more than one pixi version: {sorted(versions)}"


def test_install_doc_version_satisfies_requires_pixi() -> None:
    spec = SpecifierSet(_SIMACE_RANGE)
    for version in _DOC_VERSION.findall(_INSTALL_DOC.read_text()):
        assert spec.contains(version), (
            f"{_INSTALL_DOC}: documents pixi v{version}, outside requires-pixi "
            f"'{_SIMACE_RANGE}' (pixi.toml). Move the doc version and the manifest range together."
        )


@pytest.mark.parametrize(("label", "pixi_toml"), _MANIFESTS, ids=[label for label, _ in _MANIFESTS])
def test_requires_pixi_is_shared(label: str, pixi_toml: Path) -> None:
    assert _requires_pixi(pixi_toml) == _SIMACE_RANGE, (
        f"{label}: requires-pixi '{_requires_pixi(pixi_toml)}' differs from simACE's '{_SIMACE_RANGE}'"
    )
