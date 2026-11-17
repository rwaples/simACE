"""Naive h² estimators: closed-form maps from relative-pair correlations to h².

Each is unbiased under random mating with no shared environment and drifts
under assortative mating or C, by an amount that differs by relationship
class. The report's observed-scale estimators
(:func:`~simace.analysis.stats.correlations.compute_observed_h2_estimators`)
and the Examples comparison figures both read them from here. NaN marks a
missing input and propagates.
"""

from __future__ import annotations

__all__ = ["NaiveH2", "falconer_h2", "naive_h2", "pooled_mean"]

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def falconer_h2(r_mz: float, r_fs: float) -> float:
    """Falconer's ``2 * (r_MZ - r_FS)``; NaN when either correlation is NaN."""
    return 2.0 * (r_mz - r_fs)


def pooled_mean(values: Iterable[float]) -> float:
    """Equal-weight mean of the finite *values*; NaN when none is finite."""
    finite = [float(v) for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


@dataclass(frozen=True)
class NaiveH2:
    """The five naive h² estimates from one set of correlations.

    Attributes:
        falconer: ``2 * (r_MZ - r_FS)``.
        sibs: ``2 * r_FS``.
        po: The midparent-offspring regression slope.
        hs: ``4 * r_HS``, with MHS and PHS pooled by :func:`pooled_mean`.
        cousins: ``8 * r_1C``.
    """

    falconer: float
    sibs: float
    po: float
    hs: float
    cousins: float


def naive_h2(r: Mapping[str, float], po_slope: float) -> NaiveH2:
    """The five naive estimates from correlations *r* (keyed ``MZ``, ``FS``, ``MHS``, ``PHS``, ``1C``)."""
    return NaiveH2(
        falconer=falconer_h2(r["MZ"], r["FS"]),
        sibs=2.0 * r["FS"],
        po=po_slope,
        hs=4.0 * pooled_mean((r["MHS"], r["PHS"])),
        cousins=8.0 * r["1C"],
    )
