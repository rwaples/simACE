"""Every repo path that ``CLAUDE.md`` names in backticks exists.

A path in a nested checkout goes stale when that checkout renames it, and the simACE pre-commit hook never
sees the rename; this test does, on every ``pixi run test``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_SPAN = re.compile(r"`([^`\s]+)`")
#: A checkout nested at one of these depths that is absent here is skipped, not reported (a fresh clone).
NESTED = re.compile(r"^(external/[^/]+|fitACE(/fitACE_epimight)?)(/|$)")


def named_paths(text: str) -> list[str]:
    """The backticked tokens of ``text`` that start with an entry of the repo root, minus templates."""
    paths = []
    for span in CODE_SPAN.findall(text):
        token = span.split("::", 1)[0]
        if "/" not in token or token.startswith("/") or re.search(r"[{}<>]|://", token):
            continue
        if (ROOT / token.split("/", 1)[0]).exists():
            paths.append(token)
    return paths


def missing(token: str) -> bool:
    """Whether ``token`` names nothing under the repo root; a glob counts when it matches something."""
    m = NESTED.match(token)
    if m and not (ROOT / m.group(1)).exists():
        return False
    return not (any(ROOT.glob(token.rstrip("/"))) if "*" in token else (ROOT / token).exists())


def test_claude_md_names_only_existing_paths() -> None:
    paths = named_paths((ROOT / "CLAUDE.md").read_text())
    assert len(paths) >= 10, "the extractor stopped finding CLAUDE.md's paths"
    assert [p for p in paths if missing(p)] == []


def test_a_renamed_path_is_reported() -> None:
    assert named_paths("see `tools/no_such_tool.py` and `results/{folder}/x`") == ["tools/no_such_tool.py"]
    assert missing("tools/no_such_tool.py")
    assert not missing("tools/family_repos.py")
    assert not missing("external/epimight*")
