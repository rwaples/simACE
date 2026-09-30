"""Tetrachoric correlation primitives.

Low-level helpers for tetrachoric MLE from a 2×2 table or two binary arrays.
"""

import logging

import numpy as np

from simace.core._numba_utils import _tetrachoric_core

logger = logging.getLogger(__name__)


def tetrachoric_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return the MLE tetrachoric correlation between two binary arrays.

    Args:
        a: First binary array.
        b: Second binary array, same length as *a*.

    Returns:
        Tetrachoric correlation coefficient.
    """
    r, _ = tetrachoric_corr_se(a, b)
    return r


def tetrachoric_corr_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Estimate tetrachoric correlation and SE from two binary arrays via MLE."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    return tetrachoric_from_table(int(np.sum(a & b)), int(np.sum(a & ~b)), int(np.sum(~a & b)), int(np.sum(~a & ~b)))


def tetrachoric_from_table(n11: int, n10: int, n01: int, n00: int) -> tuple[float, float]:
    """Estimate tetrachoric correlation and SE from a 2×2 table via MLE.

    ``n11`` counts pairs with both members affected, ``n10`` first affected
    only, ``n01`` second affected only, ``n00`` neither. Delegates the
    numerical work (Brent optimization + bivariate normal CDF) to the
    numba-jitted ``_tetrachoric_core``. NaN for an empty table or a
    degenerate marginal, which leaves the core's thresholds undefined.
    """
    n_pairs = n11 + n10 + n01 + n00
    if n_pairs == 0:
        return np.nan, np.nan

    if n_pairs < 50:
        logger.warning("tetrachoric_corr_se: n_pairs=%d < 50, SE may be unreliable", n_pairs)

    p_a = (n11 + n10) / n_pairs
    p_b = (n11 + n01) / n_pairs
    if p_a in (0, 1) or p_b in (0, 1):
        return np.nan, np.nan

    return _tetrachoric_core(float(n11), float(n10), float(n01), float(n00))
