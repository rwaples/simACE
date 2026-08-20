"""Shared helpers for the validation subdomain modules.

Cross-cutting result envelope, correlation-tolerance, and pair-subsampling
helpers used by more than one validation module. Generic numerics live in
:mod:`simace.core.numerics`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from simace.core.pedigree_arrays import PedigreeArrays

_MAX_CORR_PAIRS = 5000  # cap pair-correlation samples for cost
_MIN_PAIRS_FOR_CORR = 10  # below this, skip the correlation check
_DEFAULT_RNG_SEED = 42


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


def _subsample_pairs(
    idx1: np.ndarray, idx2: np.ndarray, rng: np.random.Generator, max_pairs: int = _MAX_CORR_PAIRS
) -> tuple[np.ndarray, np.ndarray, int]:
    """Cap (idx1, idx2) pair arrays at ``max_pairs`` via without-replacement sampling."""
    n = len(idx1)
    if n <= max_pairs:
        return idx1, idx2, n
    sel = rng.choice(n, max_pairs, replace=False)
    return idx1[sel], idx2[sel], max_pairs


def _extract_comp_vals(ped: PedigreeArrays) -> dict[str, np.ndarray]:
    """Pull A/C/E component arrays for both traits as numpy views (no copy)."""
    return {f"{c}{t}": ped[f"{c}{t}"] for c in ("A", "C", "E") for t in (1, 2)}


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
