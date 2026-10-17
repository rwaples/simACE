"""Shared helpers for the validation subdomain modules.

Cross-cutting result envelope, correlation-tolerance, sibling-moment and
mate-correlation helpers used by more than one validation module. Generic
numerics live in :mod:`simace.core.numerics`.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pedigree_graph import PedigreeGraph

from simace.analysis.mate_correlation import mate_correlation_matrix
from simace.core.numerics import _ZERO_VAR_THRESHOLD

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments
    from pg_phenotype import Pedigree

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


def sorted_runs(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The stable sort order of ``keys`` and the start of each run of equal keys in that order.

    A sort, not ``np.unique``, whose hash path is slow on big int64 keys.
    """
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    return order, np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])


def cluster_codes(keys: np.ndarray) -> np.ndarray:
    """Dense codes ``0..G-1`` for the distinct ``keys``, in sorted key order."""
    order, starts = sorted_runs(keys)
    codes = np.empty(len(keys), dtype=np.int64)
    codes[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(keys)]))
    return codes


def sibling_moments(
    df: pd.DataFrame | pl.DataFrame, ped: PedigreeArrays, graph: PedigreeGraph | None = None
) -> RelationshipMoments:
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
    if graph is None:
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


@dataclass(frozen=True)
class MateCorrelations:
    """The recorded pedigree's mate correlations, computed once per validation run.

    Attributes:
        n_pairs: Mating Pairs (distinct mother and father, both rows).
        liability: ``liability[i, j]`` correlates the mothers'
            ``liability{i+1}`` with the fathers' ``liability{j+1}``; NaN where
            undefined.
        genetic: The genetic mate correlation ``mu_A = Corr(A_mother,
            A_father)`` per trait, ``genetic[t - 1]``; NaN where undefined.
    """

    n_pairs: int
    liability: np.ndarray
    genetic: tuple[float, float]


def mate_correlations(df: pd.DataFrame | pl.DataFrame, pedigree: Pedigree) -> MateCorrelations:
    """Liability and genetic mate correlations of ``df`` over its :func:`~simace.analysis.mate_correlation.mate_pedigree`."""
    t0 = time.perf_counter()
    liability, n_pairs = mate_correlation_matrix(pedigree, df, ["liability1", "liability2"])
    mu_a = [float(mate_correlation_matrix(pedigree, df, [f"A{t}"])[0][0, 0]) for t in (1, 2)]
    logger.info("Mate correlations over %d pairs completed in %.1fs", n_pairs, time.perf_counter() - t0)
    return MateCorrelations(n_pairs, liability, (mu_a[0], mu_a[1]))
