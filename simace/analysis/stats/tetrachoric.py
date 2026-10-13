"""Tetrachoric correlation primitives.

Low-level helpers for the tetrachoric correlation of a 2×2 table or two binary
arrays, computed by :func:`pg_phenotype.correlation.tetrachoric`.
"""

import logging

import numpy as np
from pg_phenotype.correlation import tetrachoric

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
    only, ``n01`` second affected only, ``n00`` neither. The fit is two-step
    maximum likelihood (thresholds from the margins, then rho) and the SE is
    its sandwich. r is NaN for an empty table or a degenerate marginal. The SE
    is also NaN when rho sits at its bound (an empty cell, or |rho| near 1),
    where rho is not identified and r is only a point on a flat likelihood.
    """
    n_pairs = n11 + n10 + n01 + n00
    if 0 < n_pairs < 50:
        logger.warning("tetrachoric_from_table: n_pairs=%d < 50, SE may be unreliable", n_pairs)

    (c00, c01, c10, c11), sign = _canonical_table(n11, n10, n01, n00)
    fit = tetrachoric(table=[[c00, c01], [c10, c11]])
    if fit.value is None:
        return np.nan, np.nan
    r = sign * fit.value if sign else 0.0
    se = np.nan if fit.se is None else fit.se
    return r, se


def _canonical_table(n11: int, n10: int, n01: int, n00: int) -> tuple[tuple[int, int, int, int], int]:
    """One orientation ``(n00, n01, n10, n11)`` of a 2×2 table across swaps and relabelings, and r's sign.

    Swapping the variables or complementing both leaves the tetrachoric
    correlation unchanged; complementing one reverses its sign. At a boundary
    fit the likelihood is flat and each orientation stops at a different point
    (simACE #14), so every orientation is fitted as the greatest one. The sign
    is 0 when the table equals a sign-reversing relabeling of itself, so r = 0.
    """
    same = [(n00, n01, n10, n11), (n00, n10, n01, n11), (n11, n10, n01, n00), (n11, n01, n10, n00)]
    reversed_ = [(n10, n11, n00, n01), (n10, n00, n11, n01), (n01, n00, n11, n10), (n01, n11, n00, n10)]
    best = max(same + reversed_)
    return best, (best in same) - (best in reversed_)
