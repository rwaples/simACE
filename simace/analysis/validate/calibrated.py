"""Calibrated tests: one rep's inputs, the registry entry, and the per-cell results (ADR 0028).

A calibrated test has a designed false-positive rate, so one rep's result is
one draw: each rep's ``validation.yaml`` holds its cells and no per-rep pass.
A cell is scored (observed, expected, SE, z), information (reported with the
reason it is not scored; it counts against coverage), or not applicable
(the quantity is degenerate in this rep, such as a residual with zero
variance; left out of the coverage count, ADR 0029 §4). The verdict is
decided across reps.

The registry also has a slot for an on-demand quality check, which keeps a
per-rep pass; no check needs one yet, so it has no type.
"""

from __future__ import annotations

__all__ = [
    "FAMILYWISE_ALPHA",
    "CalibratedTest",
    "RepInputs",
    "information_cell",
    "not_applicable_cell",
    "scored_cell",
    "summarize",
]

from dataclasses import dataclass
from functools import cached_property
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import polars as pl

    from .verdict import ZCalibration

#: Share of correct reps whose largest |z| exceeds the per-rep bound of :func:`summarize`.
FAMILYWISE_ALPHA = 0.05


@dataclass(frozen=True)
class RepInputs:
    """What a calibrated test may read of one rep; each file is read only when a test asks for it."""

    pedigree_path: Path
    #: The rep's ``report.yaml``; None when no requested check reads :attr:`params`.
    report_path: Path | None
    seed: int
    #: ``run_phenotype``'s keyword arguments other than ``seed``.
    phenotype: Mapping[str, Any]
    #: The rep's ``mating.yaml``; None when no requested check reads :attr:`mating_log`.
    mating_path: Path | None = None

    @cached_property
    def pedigree(self) -> pl.DataFrame:
        """The rep's recorded pedigree (``pedigree.parquet``)."""
        from simace.core.cohort import read_pedigree

        return read_pedigree(self.pedigree_path)

    @cached_property
    def params(self) -> Mapping[str, Any]:
        """The scenario parameters the rep was built with (``report.yaml``'s ``inputs.parameters``)."""
        from simace.core.yaml_io import load_yaml

        if self.report_path is None:
            raise ValueError("this check reads the scenario parameters: pass the rep's report.yaml (--report)")
        return load_yaml(self.report_path)["inputs"]["parameters"]

    @cached_property
    def mating_log(self) -> list[dict[str, Any]]:
        """The simulator's pairing record per mating iteration (``mating.yaml``, ADR 0032)."""
        from simace.simulation.assortment import read_mating_log

        if self.mating_path is None:
            raise ValueError("this check reads the pairing records: pass the rep's mating.yaml (--mating)")
        if not self.mating_path.exists():
            raise ValueError(
                f"{self.mating_path} is missing: the rep predates pairing records (ADR 0032); recompute it"
            )
        return read_mating_log(self.mating_path)


@dataclass(frozen=True)
class CalibratedTest:
    """One calibrated test: its name in ``validation.yaml``, a one-line description, its per-rep step, and its rule.

    ``per_rep`` returns the rep's section of ``validation.yaml``:
    ``{"scope", "cells": {name: cell}, "summary": {...}}``. ``rule`` decides
    the verdict across reps from those sections; with fewer than
    ``min_reps`` reps the verdict is ``insufficient``. ``cell_names`` lists
    every cell the test can report, in order: ``simace gather`` gives each a
    z column, so every row carries every column. A test with ``own_page``
    has its own atlas page; the others share the calibrated-tests page.
    """

    name: str
    description: str
    per_rep: Callable[[RepInputs], dict[str, Any]]
    rule: ZCalibration
    min_reps: int
    cell_names: Sequence[str]
    own_page: bool = False


def scored_cell(observed: float, expected: float, se: float, **stats: Any) -> dict[str, Any]:
    """A scored cell: its z is ``(observed − expected) / SE``."""
    return {
        "status": "scored",
        "observed": observed,
        "expected": expected,
        "se": se,
        "z": (observed - expected) / se,
        **stats,
    }


def information_cell(reason: str, **stats: Any) -> dict[str, Any]:
    """A cell reported for the record but not scored, with the reason."""
    return {"status": "information", "reason": reason, **stats}


def not_applicable_cell(reason: str, **stats: Any) -> dict[str, Any]:
    """A cell whose quantity is degenerate in this rep, with the reason: reported, and left out of coverage."""
    return {"status": "not_applicable", "reason": reason, **stats}


def summarize(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A rep's ``summary``: ``n_scored``, the largest ``|z|`` and the per-rep Bonferroni bound over the scored cells.

    ``z_bound = Φ⁻¹(1 − FAMILYWISE_ALPHA/(2K))`` over the ``K`` scored cells,
    which a correct rep exceeds with probability about ``FAMILYWISE_ALPHA``;
    both are None with nothing scored.
    """
    abs_z = [abs(cell["z"]) for cell in cells.values() if cell["status"] == "scored"]
    return {
        "n_scored": len(abs_z),
        "max_abs_z": max(abs_z) if abs_z else None,
        "z_bound": NormalDist().inv_cdf(1 - FAMILYWISE_ALPHA / (2 * len(abs_z))) if abs_z else None,
    }
