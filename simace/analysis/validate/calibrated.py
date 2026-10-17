"""Calibrated tests: one rep's inputs, the registry entry, and the per-cell results (ADR 0028).

A calibrated test has a designed false-positive rate, so one rep's result is
one draw: each rep's ``validation.yaml`` holds its cells (observed, expected,
SE, z, or information with the reason it is not scored) and no per-rep pass.
The verdict is decided across reps.

The registry also has a slot for an on-demand quality check, which keeps a
per-rep pass; no check needs one yet, so it has no type.
"""

from __future__ import annotations

__all__ = ["CalibratedTest", "RepInputs", "information_cell", "scored_cell"]

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import polars as pl

    from .verdict import ZCalibration


@dataclass(frozen=True)
class RepInputs:
    """What a calibrated test may read of one rep; each file is read only when a test asks for it."""

    pedigree_path: Path
    seed: int
    #: ``run_phenotype``'s keyword arguments other than ``seed``.
    phenotype: Mapping[str, Any]

    @cached_property
    def pedigree(self) -> pl.DataFrame:
        """The rep's recorded pedigree (``pedigree.parquet``)."""
        from simace.core.cohort import read_pedigree

        return read_pedigree(self.pedigree_path)


@dataclass(frozen=True)
class CalibratedTest:
    """One calibrated test: its name in ``validation.yaml``, a one-line description, its per-rep step, and its rule.

    ``per_rep`` returns the rep's section of ``validation.yaml``:
    ``{"scope", "cells": {name: cell}, "summary": {...}}``. ``rule`` decides
    the verdict across reps from those sections; with fewer than
    ``min_reps`` reps the verdict is ``insufficient``.
    """

    name: str
    description: str
    per_rep: Callable[[RepInputs], dict[str, Any]]
    rule: ZCalibration
    min_reps: int


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
