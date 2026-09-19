"""Consistency test: the pedigree-graph release gate covers every family check unit.

``tools/family_repos.py`` is the single source of the family membership list;
``tools/pg08_release_gate.py`` spells its units out by hand, because each one
carries bespoke steps (a pytest path, a C++ binary, a Snakemake smoke) that the
manifest does not describe.  The 0.9.0 release found the gate two units short --
``fitACE_stan`` and ``tetraher_simace`` had been added to the family and never
reached the gate -- so the link is enforced here instead of by eye.

A unit label that is not a family label fails too: a typo in either file would
otherwise read as coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from family_repos import all_repos  # noqa: E402  (needs the sys.path tweak above)
from pg08_release_gate import units  # noqa: E402


def test_gate_units_are_exactly_the_family() -> None:
    """Every family entry is a gate unit and every gate unit is a family entry."""
    assert {u.label for u in units()} == {r.label for r in all_repos()}


def test_gate_unit_labels_are_unique() -> None:
    """Two units sharing a label would overwrite each other's evidence record."""
    labels = [u.label for u in units()]
    assert len(labels) == len(set(labels))
