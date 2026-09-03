"""Numerical helpers: safe and numba-accelerated correlation/regression."""

__all__ = [
    "as_kernel_input",
    "bvn_cdf",
    "fast_linregress",
    "fast_pearsonr",
    "ndtri",
    "norm_cdf",
    "norm_pdf",
    "norm_sf",
    "safe_corrcoef",
    "safe_linregress",
    "tetrachoric_core",
]

from typing import Any

import numpy as np

# Public re-exports of the numba distributional / bivariate-normal kernels for
# downstream packages (fitACE PA-FGRS). Re-exported, not re-implemented, so the
# object identity is the same jitted dispatcher — the kernels stay callable from
# within downstream ``@njit`` code. This is simACE's public surface for these
# primitives; the underlying ``_numba_utils`` module is internal.
from simace.core._numba_utils import (
    _bvn_cdf as bvn_cdf,
)
from simace.core._numba_utils import _linregress_core, _pearsonr_core, _t_sf
from simace.core._numba_utils import (
    _ndtri_approx as ndtri,
)
from simace.core._numba_utils import (
    _norm_cdf as norm_cdf,
)
from simace.core._numba_utils import (
    _norm_pdf as norm_pdf,
)
from simace.core._numba_utils import (
    _norm_sf as norm_sf,
)
from simace.core._numba_utils import (
    _tetrachoric_core as tetrachoric_core,
)

_ZERO_VAR_THRESHOLD = 1e-10


def as_kernel_input(a: np.ndarray) -> np.ndarray:
    """Return *a* as a writable C-contiguous array for the numba kernels.

    numba types a read-only array (``readonly array(float64, 1d, C)``) as a
    **distinct** type from a writable one, so a kernel fed both compiles — and,
    with ``cache=True``, stores on disk — two separate signatures.

    Both variants reach these kernels routinely. Under pandas 3 (and pandas 2
    with Copy-on-Write) an array taken from a DataFrame is read-only, while a
    fancy-indexed selection or a freshly allocated array is writable. Left
    unnormalised that doubles compile time and misses the on-disk cache on
    whichever variant was not warmed.

    Normalising on the **writable** variant is what keeps that cache useful: it
    is the signature every existing ``__pycache__/*.nbi`` entry across the
    family was warmed with, so no call site is pushed into a fresh compile.
    That matters beyond speed — a late compile inside a process that has
    already pinned numba's threading layer (as
    ``fitACE_pafgrs/workflow/scripts/pafgrs_score.py`` does via
    ``NUMBA_NUM_THREADS``) raises ``RuntimeError`` outright. Normalising on the
    read-only variant instead is zero-copy but invalidates every warm cache
    entry, which trips exactly that failure.

    The copy is therefore deliberate, and taken only when the input is
    read-only. The caller's array is never modified. A non-contiguous input is
    made contiguous, which also avoids a third ``'A'``-layout signature.

    Args:
        a: Array destined for a numba kernel.

    Returns:
        A writable, C-contiguous array holding the values of *a*.
    """
    arr = np.ascontiguousarray(a)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning nan if either array has zero variance.

    Args:
        x: First array of observations.
        y: Second array of observations, same length as *x*.

    Returns:
        Pearson correlation coefficient, or nan if variance is near-zero.
    """
    if np.std(x) < _ZERO_VAR_THRESHOLD or np.std(y) < _ZERO_VAR_THRESHOLD:
        return float("nan")
    return float(_pearsonr_core(as_kernel_input(x), as_kernel_input(y)))


def safe_linregress(x: np.ndarray, y: np.ndarray) -> Any:
    """Run linear regression, returning None if x has zero variance.

    Args:
        x: Independent variable array.
        y: Dependent variable array, same length as *x*.

    Returns:
        ``scipy.stats.LinregressResult`` or None if *x* has near-zero variance.
    """
    if np.std(x) < _ZERO_VAR_THRESHOLD:
        return None
    # Imported here, not at module level: scipy.stats costs ~0.5 s to import and
    # this is the only caller, so keep it off the per-job startup path.
    from scipy import stats

    return stats.linregress(x, y)


def fast_linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fast linear regression via numba-accelerated core.

    Args:
        x: Independent variable array.
        y: Dependent variable array, same length as *x*.

    Returns:
        Tuple of (slope, intercept, r, stderr, pvalue).
    """
    slope, intercept, r, stderr, t_stat = _linregress_core(as_kernel_input(x), as_kernel_input(y))
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
    r = float(_pearsonr_core(as_kernel_input(x), as_kernel_input(y)))
    n = len(x)
    denom = 1.0 - r * r
    if denom < 1e-30 or n <= 2:
        return r, 0.0
    t_stat = r * np.sqrt((n - 2) / denom)
    pvalue = float(2.0 * _t_sf(abs(t_stat), n - 2))
    return r, pvalue
