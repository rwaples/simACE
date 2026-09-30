"""Shared helpers for the validation subdomain modules.

Cross-cutting result envelope, correlation-tolerance, and sibling-moment
helpers used by more than one validation module. Generic numerics live in
:mod:`simace.core.numerics`.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from pedigree_graph import PedigreeGraph

from simace.core.numerics import _ZERO_VAR_THRESHOLD

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments

    from simace.core.pedigree_arrays import PedigreeArrays

logger = logging.getLogger(__name__)

_MIN_PAIRS_FOR_CORR = 10  # below this, skip the correlation check
SIBLING_CATEGORIES = ("FS", "MHS", "PHS")


def _result(passed: bool, details: str, **extra: Any) -> dict[str, Any]:
    """Build a standardized validation result dict."""
    d: dict[str, Any] = {"passed": passed, "details": details}
    d.update(extra)
    return d


def _info(details: str, **extra: Any) -> dict[str, Any]:
    """Build an informational (non-scored) validation result dict.

    Unlike :func:`_result`, ``_info`` carries no ``passed`` key and stamps
    ``informational=True``. These are metrics with no closed-form expected
    value to assert against (e.g. observed liability correlations, regression
    slopes), so there is no meaningful pass/fail.
    ``report.normalize_quality_checks`` skips any result flagged informational
    (or lacking ``passed``), so they are reported for the record but never
    counted toward the pass/fail tally — the explicit marker makes that intent
    legible instead of leaving it inferred from an absent key.
    """
    return {"informational": True, "details": details, **extra}


def _corr_se(expected_r: float, n_pairs: int) -> float:
    """Approximate SE of Pearson correlation: (1 - r^2) / sqrt(n - 1)."""
    return (1 - expected_r**2) / np.sqrt(max(n_pairs - 1, 1))


def _corr_tolerance(expected_r: float, n_pairs: int, min_tol: float = 0.05, n_se: int = 4) -> float:
    """Compute SE-based tolerance for correlation checks."""
    se = _corr_se(expected_r, n_pairs)
    return max(n_se * se, min_tol)


def _extract_comp_vals(ped: PedigreeArrays) -> dict[str, np.ndarray]:
    """Pull A/C/E component arrays for both traits as numpy views (no copy)."""
    return {f"{c}{t}": ped[f"{c}{t}"] for c in ("A", "C", "E") for t in (1, 2)}


def sibling_moments(df: pd.DataFrame | pl.DataFrame, ped: PedigreeArrays) -> RelationshipMoments:
    """Exact FS/MHS/PHS pair moments of ``A{t}``, ``C{t}`` and ``P{t} = A + C + E`` per trait.

    One engine pass over the recorded pedigree, one cell per category (ADR
    0020): the sibling counts and every sibling correlation the validation
    checks report come from this table, over all pairs.
    """
    comp = _extract_comp_vals(ped)
    values = {}
    for t in (1, 2):
        values[f"A{t}"] = comp[f"A{t}"]
        values[f"C{t}"] = comp[f"C{t}"]
        values[f"P{t}"] = comp[f"A{t}"] + comp[f"C{t}"] + comp[f"E{t}"]
    t0 = time.perf_counter()
    graph = PedigreeGraph.from_frame(df)
    logger.info("Recorded pedigree graph build completed in %.1fs", time.perf_counter() - t0)
    return graph.relationship_moments(categories=SIBLING_CATEGORIES, values=values)


def category_cell(moments: RelationshipMoments, *categories: str) -> RelationshipMoments:
    """The moments of *categories* pooled into one 0-d cell."""
    return moments.select(category=list(categories)).sum("category")


def pair_correlation(cell: RelationshipMoments, column: str) -> float:
    """Pearson correlation of *column* between the two members of a one-cell result.

    The rule of :func:`simace.core.numerics.safe_corrcoef`: NaN when either
    side's standard deviation (``sqrt(m2 / n)``, the population value) is
    below the zero-variance threshold.
    """
    n = int(cell.counts)
    i = cell.columns.index(column)
    if n == 0:
        return math.nan
    if math.sqrt(float(cell.m2_first[i]) / n) < _ZERO_VAR_THRESHOLD:
        return math.nan
    if math.sqrt(float(cell.m2_second[i]) / n) < _ZERO_VAR_THRESHOLD:
        return math.nan
    return float(cell.pearson(f"first.{column}", f"second.{column}"))


def _unique_mating_pairs(df: pd.DataFrame | pl.DataFrame, ped: PedigreeArrays) -> tuple[np.ndarray, np.ndarray]:
    """Return unique (mother, father) id arrays with both parents present.

    Library-agnostic (ADR 0015): non-founder rows are selected with a NumPy
    mask and matings deduplicated via an int64 pair key (ids are int32, so
    ``base**2`` fits int64). Pairs come back in sorted-key order; every
    consumer computes order-invariant statistics over them. Ascertainment
    severs mother and father independently, so a row can pass the
    ``mother != -1`` filter while carrying a severed father — pairs whose
    parents are not both present in ``ped`` are dropped.
    """
    mothers_all = df["mother"].to_numpy()
    fathers_all = df["father"].to_numpy()
    mask = mothers_all != -1
    m = mothers_all[mask].astype(np.int64)
    f = fathers_all[mask].astype(np.int64)
    if m.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    # Shift fathers by +1 so a severed father (-1) keys injectively.
    base = np.int64(max(int(m.max()), int(f.max())) + 2)
    uniq = np.unique(m * base + (f + 1))
    mothers = uniq // base
    fathers = uniq % base - 1
    both = ped.contains(mothers) & ped.contains(fathers)
    return mothers[both], fathers[both]
