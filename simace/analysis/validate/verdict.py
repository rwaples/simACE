"""Verdicts of calibrated tests across reps (ADR 0028 §2–5).

One rep's cells are one draw, so a calibrated test is decided across reps:
over one scenario's reps, or over a folder's scenarios plus their reps pooled
per cell. Each verdict is one Bonferroni family at a familywise 5%, every
test in it decided at ``alpha = 0.05 / n_tests``:

- a scenario's family holds its tests' scenario-level tests, and is the same
  whatever ``simace validate`` was asked for;
- a folder's family holds every scenario's tests plus the pooled-per-cell
  tests. It is the one to cite, and is refused when two scenarios' rep seed
  ranges overlap, since pooling would then count the same draws twice.

Coverage is a requirement, not a test, and is not counted in the family. A
test with fewer than its ``min_reps`` reps is ``insufficient``, never pass;
a failed test outranks an insufficient one.

``validation_summary.yaml`` carries ``schema: {kind: calibration_verdict,
version: 1}`` and :func:`load_verdict` refuses a file without it.
"""

from __future__ import annotations

__all__ = [
    "SCHEMA",
    "VERDICT_ALPHA",
    "Entry",
    "Verdict",
    "ZCalibration",
    "decide",
    "folder_family",
    "load_verdict",
    "overlapping_seed_ranges",
    "record_name",
    "records",
    "scenario_family",
    "write_verdict",
]

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy import stats

from .schema_file import load_schemad

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from .calibrated import CalibratedTest

    #: One rep's section of ``validation.yaml`` for one test: ``{"scope", "cells", "summary"}``.
    Section = Mapping[str, Any]

#: The ``schema`` block of every ``validation_summary.yaml`` this simace writes and reads.
SCHEMA = {"kind": "calibration_verdict", "version": 1}

#: Each verdict family's chance of failing at least one test when the simulation is correct.
VERDICT_ALPHA = 0.05


class Verdict(StrEnum):
    """A family's outcome."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class MeanZ:
    """The mean of n scored z lies within ``Φ⁻¹(1 − α/2)/√n``."""

    z: np.ndarray
    counted: ClassVar[bool] = True

    def decide(self, alpha: float) -> tuple[bool, dict[str, Any]]:
        n = len(self.z)
        if n < 2:
            return False, {"n": n, "mean": None, "bound": None, "passed": False}
        mean = float(self.z.mean())
        bound = float(stats.norm.isf(alpha / 2)) / math.sqrt(n)
        passed = abs(mean) <= bound
        return passed, {"n": n, "mean": mean, "bound": bound, "passed": passed}


@dataclass(frozen=True)
class SdZ:
    """``SD² · (n − 1)`` of n scored z lies within the ``χ²_{n−1}`` quantiles at ``α/2`` and ``1 − α/2``.

    The record gives the interval on the SD scale.
    """

    z: np.ndarray
    counted: ClassVar[bool] = True

    def decide(self, alpha: float) -> tuple[bool, dict[str, Any]]:
        n = len(self.z)
        if n < 2:
            return False, {"n": n, "sd": None, "lo": None, "hi": None, "passed": False}
        sd = float(self.z.std(ddof=1))
        lo, hi = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], n - 1)
        passed = bool(lo <= sd**2 * (n - 1) <= hi)
        record = {"n": n, "sd": sd, "lo": math.sqrt(lo / (n - 1)), "hi": math.sqrt(hi / (n - 1)), "passed": passed}
        return passed, record


@dataclass(frozen=True)
class RepsOverBound:
    """The reps whose largest |z| exceeds their per-rep bound number at most the binomial(reps, rate) quantile at ``1 − α``."""

    over_bound: int
    reps: int
    rate: float
    counted: ClassVar[bool] = True

    def decide(self, alpha: float) -> tuple[bool, dict[str, Any]]:
        allowed = int(stats.binom.isf(alpha, self.reps, self.rate))
        passed = self.over_bound <= allowed
        return passed, {"reps": self.reps, "over_bound": self.over_bound, "allowed": allowed, "passed": passed}


@dataclass(frozen=True)
class Coverage:
    """A requirement, not a test: a cell is scored in at least ``minimum`` of the reps where it applies.

    ``reps`` excludes the reps whose cell is ``not_applicable`` (ADR 0029 §4).
    """

    scored: int
    reps: int
    minimum: float
    counted: ClassVar[bool] = False

    def decide(self, alpha: float) -> tuple[bool, dict[str, Any]]:
        share = self.scored / self.reps
        passed = share >= self.minimum
        return passed, {"scored": self.scored, "reps": self.reps, "share": share, "passed": passed}


@dataclass(frozen=True)
class Insufficient:
    """A test that cannot be decided, with the reason: never a pass."""

    reason: str
    counted: ClassVar[bool] = False

    def decide(self, alpha: float) -> tuple[None, dict[str, Any]]:
        return None, {"insufficient": self.reason}


Statistic = MeanZ | SdZ | RepsOverBound | Coverage | Insufficient


@dataclass(frozen=True)
class Entry:
    """One statistic of a family, at its place in ``validation_summary.yaml``."""

    path: tuple[str, ...]
    statistic: Statistic


def decide(entries: Sequence[Entry]) -> dict[str, Any]:
    """Decide a family: ``{"verdict", "n_tests", "alpha_each", ...}`` plus each entry's record at its path."""
    n_tests = sum(entry.statistic.counted for entry in entries)
    alpha = VERDICT_ALPHA / max(n_tests, 1)
    tree: dict[str, Any] = {}
    outcomes = []
    for entry in entries:
        passed, record = entry.statistic.decide(alpha)
        outcomes.append(passed)
        node = tree
        for key in entry.path[:-1]:
            node = node.setdefault(key, {})
        node[entry.path[-1]] = record
    if False in outcomes:
        verdict = Verdict.FAIL
    elif None in outcomes or not outcomes:
        verdict = Verdict.INSUFFICIENT
    else:
        verdict = Verdict.PASS
    return {"verdict": verdict.value, "n_tests": n_tests, "alpha_each": alpha if n_tests else None, **tree}


def _cell_names(sections: Sequence[Section]) -> list[str]:
    return list(dict.fromkeys(name for section in sections for name in section["cells"]))


def _applicable_reps(sections: Sequence[Section], cell: str) -> int:
    """The reps whose cell is not ``not_applicable``: one missing the cell counts, as not scored."""
    return sum((section["cells"].get(cell) or {}).get("status") != "not_applicable" for section in sections)


def _scored_z(sections: Sequence[Section], cell: str) -> np.ndarray:
    found = (section["cells"].get(cell) for section in sections)
    return np.array([c["z"] for c in found if c is not None and c["status"] == "scored"], dtype=np.float64)


@dataclass(frozen=True)
class ZCalibration:
    """Scored z behave as standard normals, and reps exceed their per-rep Bonferroni bound at about ``rep_rate``.

    Per scenario, for every cell any rep reports and some rep has
    applicable: coverage over the applicable reps, then the mean-z and SD-z
    tests over the reps that scored it; and one count of the reps
    whose ``summary.max_abs_z`` exceeds ``summary.z_bound``, the per-rep
    bound that a correct rep exceeds with probability ``rep_rate``. Pooled
    over scenarios: the mean-z and SD-z tests of every cell scored anywhere.
    """

    rep_rate: float
    min_coverage: float = 0.95

    def scenario(self, sections: Sequence[Section]) -> Iterator[Entry]:
        """The statistics over one scenario's reps; insufficient when no rep reports a cell."""
        if not _cell_names(sections):
            yield Entry((), Insufficient("no rep reports a cell"))
            return
        for cell in _cell_names(sections):
            if not (applicable := _applicable_reps(sections, cell)):
                continue
            z = _scored_z(sections, cell)
            yield Entry(("cells", cell, "coverage"), Coverage(len(z), applicable, self.min_coverage))
            yield Entry(("cells", cell, "mean_z"), MeanZ(z))
            yield Entry(("cells", cell, "sd_z"), SdZ(z))
        over = sum(
            s["summary"]["max_abs_z"] is not None and s["summary"]["max_abs_z"] > s["summary"]["z_bound"]
            for s in sections
        )
        yield Entry(("reps_over_bound",), RepsOverBound(over, len(sections), self.rep_rate))

    def pooled(self, by_scenario: Sequence[Sequence[Section]]) -> Iterator[Entry]:
        """The statistics over every scenario's reps, pooled per cell."""
        cells: dict[str, list[np.ndarray]] = {}
        for sections in by_scenario:
            for cell in _cell_names(sections):
                if len(z := _scored_z(sections, cell)):
                    cells.setdefault(cell, []).append(z)
        for cell, parts in cells.items():
            z = np.concatenate(parts)
            yield Entry(("cells", cell, "mean_z"), MeanZ(z))
            yield Entry(("cells", cell, "sd_z"), SdZ(z))


def scenario_family(
    sections: Mapping[str, Sequence[Section]], tests: Mapping[str, CalibratedTest], prefix: tuple[str, ...] = ("tests",)
) -> list[Entry]:
    """One scenario's family: each test's statistics over the reps' sections, keyed by test name."""
    entries = []
    for name, reps in sections.items():
        test = tests[name]
        if len(reps) < test.min_reps:
            entries.append(
                Entry((*prefix, name), Insufficient(f"{len(reps)} rep(s); the verdict needs {test.min_reps}"))
            )
        else:
            entries.extend(Entry((*prefix, name, *e.path), e.statistic) for e in test.rule.scenario(reps))
    return entries


def folder_family(
    by_scenario: Mapping[str, Mapping[str, Sequence[Section]]],
    tests: Mapping[str, CalibratedTest],
    pooling_refused: str | None = None,
) -> list[Entry]:
    """A folder's family: every scenario's family, plus each test pooled over the scenarios.

    A test is pooled only when every scenario has its ``min_reps``, and no
    test is when ``pooling_refused`` gives a reason; such pooled tests are
    insufficient.
    """
    entries = [
        e
        for scenario, sections in by_scenario.items()
        for e in scenario_family(sections, tests, ("scenarios", scenario))
    ]
    for name, test in tests.items():
        short = [scenario for scenario, sections in by_scenario.items() if len(sections[name]) < test.min_reps]
        if pooling_refused is not None:
            entries.append(Entry(("pooled", name), Insufficient(pooling_refused)))
        elif short:
            reason = f"{', '.join(short)} below {test.min_reps} rep(s)"
            entries.append(Entry(("pooled", name), Insufficient(reason)))
        else:
            parts = [sections[name] for sections in by_scenario.values()]
            entries.extend(Entry(("pooled", name, *e.path), e.statistic) for e in test.rule.pooled(parts))
    return entries


def overlapping_seed_ranges(ranges: Mapping[str, tuple[int, int]]) -> list[tuple[str, str]]:
    """The pairs of scenarios whose inclusive rep seed ranges ``[first, last]`` overlap."""
    names = list(ranges)
    return [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if ranges[a][0] <= ranges[b][1] and ranges[b][0] <= ranges[a][1]
    ]


def write_verdict(path: Path, header: Mapping[str, Any], decided: Mapping[str, Any]) -> None:
    """Write a ``validation_summary.yaml``: the schema, then ``header`` (who and which reps), then the decided family.

    An unchanged verdict leaves the file untouched, so the plots that show it stay current.
    """
    import yaml

    from simace.core.publish import publish
    from simace.core.yaml_io import to_native

    text = yaml.dump(
        to_native({"schema": dict(SCHEMA), **header, **decided}), default_flow_style=False, sort_keys=False
    )
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    with publish(path) as (tmp,):
        tmp.write_text(text, encoding="utf-8")


#: The keys under which :func:`decide` puts a family's records: a scenario's tests, or a folder's scenarios and pooled tests.
_RECORD_KEYS = ("tests", "scenarios", "pooled")


def records(verdict: Mapping[str, Any]) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    """Each record of a decided family, with its path: a record holds ``passed``, or ``insufficient`` and the reason."""

    def walk(path: tuple[str, ...], node: Mapping[str, Any]) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
        if "passed" in node or "insufficient" in node:
            yield path, dict(node)
            return
        for key, child in node.items():
            yield from walk((*path, key), child)

    for key in _RECORD_KEYS:
        if key in verdict:
            yield from walk((key,), verdict[key])


def record_name(path: tuple[str, ...]) -> str:
    """A record's path without its grouping keys, such as ``b/onset_tau/trait1_FS/mean_z``.

    That is ``("scenarios", "b", "onset_tau", "cells", "trait1_FS", "mean_z")`` in a folder's verdict. The
    keys go by position, so a scenario or cell that shares a grouping key's name keeps it.
    """
    # The family key, then the test's name: after the scenario under ``scenarios`` and ``pooled``.
    family, *rest = path
    kept = rest if family in {"tests", "scenarios"} else [family, *rest]
    test_at = 0 if family == "tests" else 1
    if len(kept) > test_at + 1 and kept[test_at + 1] == "cells":
        del kept[test_at + 1]
    return "/".join(kept)


def load_verdict(path: Path) -> dict[str, Any] | None:
    """A ``validation_summary.yaml``, or None when there is none or it lacks this simace's schema."""
    return load_schemad(path, SCHEMA, "validation_summary.yaml")
