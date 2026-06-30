"""Property-based parity tests: pure-Python kernels vs their numba-jitted twins.

Each ``_xxx_python`` reference and the ``_xxx`` symbol actually executed at
runtime (jitted when numba is present) must agree across the input domain.
These kernels are re-exported via ``simace.core.numerics`` as the public
primitive surface that downstream packages (fitACE PA-FGRS) call from inside
their own ``@njit`` code, so a silent divergence would bias every consumer.
The existing parity tests pin only fixed parametrize points.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.core._numba_utils import (
    _bvn_cdf,
    _bvn_cdf_python,
    _linregress_core,
    _linregress_core_python,
    _norm_cdf,
    _owens_t,
    _owens_t_python,
    _pearsonr_core,
    _pearsonr_core_python,
)

_finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def _xy(draw):
    """Two equal-length float arrays, each with non-degenerate variance."""
    n = draw(st.integers(min_value=3, max_value=50))
    x = draw(hnp.arrays(np.float64, n, elements=_finite))
    y = draw(hnp.arrays(np.float64, n, elements=_finite))
    assume(np.std(x) > 1e-3)
    assume(np.std(y) > 1e-3)
    return x, y


@settings(deadline=None)
@given(_xy())
def test_pearsonr_core_python_jit_parity(xy):
    x, y = xy
    assert np.isclose(_pearsonr_core_python(x, y), _pearsonr_core(x, y), rtol=1e-9, atol=1e-12)


@settings(deadline=None)
@given(_xy())
def test_linregress_core_python_jit_parity(xy):
    x, y = xy
    py = np.array(_linregress_core_python(x, y), dtype=float)
    # Skip perfectly-collinear data (residual variance 0): the numpy reference
    # returns inf for t_stat=slope/0, whereas the numba jit (error_model
    # 'python') raises ZeroDivisionError. That degenerate input is outside the
    # regression domain the callers guard against; parity is asserted where the
    # estimate is finite.
    assume(np.all(np.isfinite(py)))
    jit = np.array(_linregress_core(x, y), dtype=float)
    assert np.allclose(py, jit, rtol=1e-7, atol=1e-10, equal_nan=True)


@settings(deadline=None)
@given(
    h=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    k=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    r=st.floats(-0.97, 0.97, allow_nan=False, allow_infinity=False),
)
def test_bvn_cdf_python_jit_parity(h, k, r):
    assert np.isclose(_bvn_cdf_python(h, k, r), _bvn_cdf(h, k, r), rtol=1e-9, atol=1e-12)


@settings(deadline=None)
@given(
    h=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    a=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
)
def test_owens_t_python_jit_parity(h, a):
    assert np.isclose(_owens_t_python(h, a), _owens_t(h, a), rtol=1e-9, atol=1e-12)


# Bound sanity that holds for any valid CDF — a cheap companion that the parity
# tests alone do not assert.
@settings(deadline=None)
@given(
    h=st.floats(-6, 6, allow_nan=False, allow_infinity=False),
    k=st.floats(-6, 6, allow_nan=False, allow_infinity=False),
    r=st.floats(-0.97, 0.97, allow_nan=False, allow_infinity=False),
)
def test_bvn_cdf_in_unit_interval(h, k, r):
    p = _bvn_cdf(h, k, r)
    assert -1e-12 <= p <= 1.0 + 1e-12
    # At r == 0 the joint CDF factorizes exactly; near zero, the first-order
    # departure is O(r * phi(h) * phi(k)), so allow tolerance proportional to r.
    if abs(r) < 1e-9:
        assert p == pytest.approx(_norm_cdf(h) * _norm_cdf(k), abs=1e-12 + 0.2 * abs(r))
