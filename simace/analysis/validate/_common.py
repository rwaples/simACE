"""Shared helpers for the validation subdomain modules.

Cross-cutting result envelope, correlation-tolerance, and pair-subsampling
helpers used by more than one validation module. Generic numerics live in
:mod:`simace.core.numerics`.
"""

from typing import Any

import numpy as np
import pandas as pd

_MAX_CORR_PAIRS = 5000  # cap pair-correlation samples for cost
_MIN_PAIRS_FOR_CORR = 10  # below this, skip the correlation check
_DEFAULT_RNG_SEED = 42


def _result(passed: bool, details: str, **extra: Any) -> dict[str, Any]:
    """Build a standardized validation result dict."""
    d: dict[str, Any] = {"passed": passed, "details": details}
    d.update(extra)
    return d


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


def _extract_comp_vals(df_indexed: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pull A/C/E component arrays for both traits as numpy views (no copy)."""
    return {f"{c}{t}": df_indexed[f"{c}{t}"].values for c in ("A", "C", "E") for t in (1, 2)}
