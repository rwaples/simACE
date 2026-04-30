"""Numerical helpers: safe and numba-accelerated correlation/regression."""

__all__ = [
    "fast_linregress",
    "fast_pearsonr",
    "safe_corrcoef",
    "safe_linregress",
]

from typing import Any

import numpy as np
from scipy import stats

from simace.core._numba_utils import _linregress_core, _pearsonr_core, _t_sf


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning nan if either array has zero variance.

    Args:
        x: First array of observations.
        y: Second array of observations, same length as *x*.

    Returns:
        Pearson correlation coefficient, or nan if variance is near-zero.
    """
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(_pearsonr_core(x, y))


def safe_linregress(x: np.ndarray, y: np.ndarray) -> Any:
    """Run linear regression, returning None if x has zero variance.

    Args:
        x: Independent variable array.
        y: Dependent variable array, same length as *x*.

    Returns:
        ``scipy.stats.LinregressResult`` or None if *x* has near-zero variance.
    """
    if np.std(x) < 1e-10:
        return None
    return stats.linregress(x, y)


def fast_linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fast linear regression via numba-accelerated core.

    Args:
        x: Independent variable array.
        y: Dependent variable array, same length as *x*.

    Returns:
        Tuple of (slope, intercept, r, stderr, pvalue).
    """
    slope, intercept, r, stderr, t_stat = _linregress_core(x, y)
    pvalue = float(2.0 * _t_sf(abs(t_stat), len(x) - 2))
    return float(slope), float(intercept), float(r), float(stderr), pvalue


def fast_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Pearson r with two-sided p-value via numba-accelerated core.

    Args:
        x: First array of observations.
        y: Second array of observations, same length as *x*.

    Returns:
        Tuple of (correlation, p-value).
    """
    r = float(_pearsonr_core(x, y))
    n = len(x)
    denom = 1.0 - r * r
    if denom < 1e-30 or n <= 2:
        return r, 0.0
    t_stat = r * np.sqrt((n - 2) / denom)
    pvalue = float(2.0 * _t_sf(abs(t_stat), n - 2))
    return r, pvalue
