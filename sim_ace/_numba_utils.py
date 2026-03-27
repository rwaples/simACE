"""Shared Numba-compiled utility functions."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _ndtri_approx_python(p):
    """Acklam rational approximation for the normal quantile (~1e-9 accuracy)."""
    a0, a1, a2 = -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02
    a3, a4, a5 = 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00
    b0, b1, b2 = -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02
    b3, b4 = 6.680131188771972e01, -1.328068155288572e01
    c0, c1, c2 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00
    c3, c4, c5 = -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00
    d0, d1, d2, d3 = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        num = ((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5
        den = (((d0 * q + d1) * q + d2) * q + d3) * q + 1.0
        return num / den
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        num = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
        den = ((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0
        return num / den
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    num = ((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5
    den = (((d0 * q + d1) * q + d2) * q + d3) * q + 1.0
    return -(num / den)


if njit is not None:
    _ndtri_approx = njit(cache=True)(_ndtri_approx_python)
else:
    _ndtri_approx = _ndtri_approx_python
