"""Tetrachoric correlation primitives.

Low-level helpers for tetrachoric MLE on binary arrays plus the
``_tetrachoric_for_pairs`` pair-subset helper used across pairwise-correlation
computations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from simace.core._numba_utils import _ndtri_approx, _norm_cdf, _pearsonr_core, _tetrachoric_core

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
    """Estimate tetrachoric correlation and SE from two binary arrays via MLE.

    Delegates the numerical work (Brent optimization + bivariate normal CDF)
    to the numba-jitted ``_tetrachoric_core`` for speed.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n_pairs = len(a)
    if n_pairs < 50:
        logger.warning("tetrachoric_corr_se: n_pairs=%d < 50, SE may be unreliable", n_pairs)

    n11 = float(np.sum(a & b))
    n10 = float(np.sum(a & ~b))
    n01 = float(np.sum(~a & b))
    n00 = float(np.sum(~a & ~b))

    p_a, p_b = a.mean(), b.mean()
    if p_a in (0, 1) or p_b in (0, 1):
        return np.nan, np.nan

    t_a = float(_ndtri_approx(1.0 - p_a))
    t_b = float(_ndtri_approx(1.0 - p_b))
    phi_ta = float(_norm_cdf(t_a))
    phi_tb = float(_norm_cdf(t_b))

    return _tetrachoric_core(n11, n10, n01, n00, t_a, t_b, phi_ta, phi_tb)


def _tetrachoric_for_pairs(
    idx1: np.ndarray,
    idx2: np.ndarray,
    affected: np.ndarray,
    liability: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute tetrachoric r, SE, and optionally liability r for one pair subset."""
    n_p = len(idx1)
    if n_p < 10:
        entry: dict[str, Any] = {"r": None, "se": None, "n_pairs": int(n_p)}
        if liability is not None:
            entry["liability_r"] = None
        return entry
    r, se = tetrachoric_corr_se(affected[idx1], affected[idx2])
    entry = {
        "r": float(r) if not np.isnan(r) else None,
        "se": float(se) if not np.isnan(se) else None,
        "n_pairs": int(n_p),
    }
    if liability is not None:
        liab_r = float(_pearsonr_core(liability[idx1], liability[idx2]))
        entry["liability_r"] = liab_r if not np.isnan(liab_r) else None
    return entry
