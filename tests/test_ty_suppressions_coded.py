"""Discipline test: every ``ty: ignore`` suppression names its rule code.

``ty`` 0.0.51 does **not** flag a codeless suppression of the form
``# ty: ignore[...]`` written without the ``[...]`` (verified). A bare suppression
silently swallows *every* diagnostic on its line -- including a real cross-repo
signature-drift error the family sweep is meant to catch. This test fails on any
``ty: ignore`` that omits its ``[rule-code]``, keeping suppressions narrow and
self-documenting. It does not touch existing (correctly-coded) suppressions.

Files are enumerated via ``git ls-files`` *per repo* so the umbrella simACE scan
does not descend into the nested (gitignored) sister checkouts. Repos that are
not checked out are skipped, same as ``test_ty_pin_consistency``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from family_repos import python_repos  # noqa: E402  (needs the sys.path tweak above)

# A ``ty`` suppression comment whose ``ignore`` is *not* followed by a ``[rule]``
# code. The ``\b`` keeps prose like "ignores" from matching; the negative
# lookahead passes coded suppressions (``ignore[...]``) through untouched.
_BARE_IGNORE = re.compile(r"#\s*ty:\s*ignore\b(?!\s*\[)")


def _tracked_py(repo: Path) -> list[Path]:
    """Tracked ``*.py`` files in *repo* (empty if it is not a git checkout)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [repo / line for line in result.stdout.splitlines() if line]


_CASES = [(r.label, _ROOT / r.path) for r in python_repos() if (_ROOT / r.path).is_dir()]


@pytest.mark.parametrize(("label", "repo"), _CASES, ids=[label for label, _ in _CASES])
def test_ty_ignores_are_rule_coded(label: str, repo: Path) -> None:
    offenders: list[str] = []
    for path in _tracked_py(repo):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _BARE_IGNORE.search(line):
                offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"{label}: found codeless ty suppression(s) -- add the specific [rule-code] "
        f"that each one silences:\n" + "\n".join(offenders)
    )
