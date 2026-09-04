"""Guard: no new pedigree-graph 0.7.1 consumer lands before the 0.8.0 migration.

``tools/pedigree_graph_old_api.py`` inventories every 0.7.1 API use in the family
and ``tools/pedigree_graph_old_api_inventory.json`` is the accepted snapshot of
that inventory. A fresh scan that turns up a ``(repo, path, symbol)`` the
snapshot does not know about fails here, in production code and test code alike,
because slice 8 has to migrate both. Guard and snapshot are deleted in slice 8.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from pedigree_graph_old_api import new_entries, scan  # noqa: E402  (needs the sys.path tweak above)

_SNAPSHOT = _ROOT / "tools" / "pedigree_graph_old_api_inventory.json"
_REGENERATE = "python tools/pedigree_graph_old_api.py scan --out tools/pedigree_graph_old_api_inventory.json"


def _snapshot() -> dict:
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> dict:
    scanned = scan(_ROOT)
    if scanned["repos_unavailable"]:
        pytest.skip(f"family checkouts unavailable: {', '.join(scanned['repos_unavailable'])}")
    return scanned


def test_no_new_old_api_consumers(report: dict) -> None:
    added = new_entries(report, _snapshot())
    listed = "\n".join(
        f"  {e['repo']}/{e['path']}:{e['line']}: {e['symbol']} ({e['kind']})  {e['text']}" for e in added
    )
    assert not added, (
        f"{len(added)} new pedigree-graph 0.7.1 use(s) since the inventory snapshot:\n"
        f"{listed}\n"
        f"Migrate each off the 0.7.1 API, or regenerate the snapshot with:\n  {_REGENERATE}"
    )


def test_snapshot_is_a_populated_inventory() -> None:
    snapshot = _snapshot()
    assert snapshot["schema"] == 1
    assert snapshot["production"]
