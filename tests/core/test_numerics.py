"""Tests for ``simace.core.numerics.as_kernel_input`` and the numba boundary.

``as_kernel_input`` exists to keep one numba signature per kernel. numba types
a read-only array (``readonly array(float64, 1d, C)``) as a **distinct** type
from a writable one, so a kernel fed both compiles — and, with ``cache=True``,
stores on disk — two signatures. Arrays taken from a DataFrame column are
read-only — polars' zero-copy ``Series.to_numpy()`` hands out a read-only view,
as did pandas under Copy-on-Write — while fancy-indexed selections and fresh
allocations are writable, so both variants reach the same kernels.

The direction matters: normalising on the **writable** variant is what keeps
the warm on-disk cache useful, because that is the signature every existing
``__pycache__/*.nbi`` entry was built with. Normalising on read-only is
zero-copy but invalidates those caches, forcing a fresh compile — which is
fatal inside a process that has pinned numba's threading layer.

``test_kernel_compiles_only_mutable_signatures`` is the regression test for
that bug; the rest pin the helper's contract.
"""

import numpy as np
import polars as pl
import pytest

from simace.core._numba_utils import _pearsonr_core
from simace.core.numerics import as_kernel_input, fast_linregress, fast_pearsonr, safe_corrcoef


def _readonly(a: np.ndarray) -> np.ndarray:
    """Return a read-only view of *a*, leaving *a* itself writable."""
    view = a.view()
    view.setflags(write=False)
    return view


# ---------------------------------------------------------------------------
# as_kernel_input contract
# ---------------------------------------------------------------------------


def test_readonly_input_becomes_writable():
    src = np.arange(6.0)
    out = as_kernel_input(_readonly(src))

    assert out.flags.writeable
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, src)


def test_writable_input_is_not_copied():
    src = np.arange(6.0)
    out = as_kernel_input(src)

    # Already writable and contiguous — the helper must not pay for a copy.
    assert out is src


def test_callers_array_is_never_modified():
    src = np.arange(6.0)
    ro = _readonly(src)

    out = as_kernel_input(ro)
    out[0] = 999.0

    # The copy is independent, and the caller's view keeps its flags.
    assert not ro.flags.writeable
    assert ro[0] == 0.0
    assert src[0] == 0.0


def test_noncontiguous_input_becomes_contiguous():
    src = np.arange(12.0)[::2]
    assert not src.flags.c_contiguous

    out = as_kernel_input(src)

    assert out.flags.c_contiguous
    assert out.flags.writeable
    np.testing.assert_array_equal(out, src)


def test_frame_column_is_accepted():
    """The production path: DataFrame columns arrive read-only.

    Polars' ``Series.to_numpy()`` is zero-copy for a null-free numeric column
    and therefore hands back a read-only view — the same hazard pandas'
    Copy-on-Write posed before the polars migration (ADR 0015).
    """
    df = pl.DataFrame({"liability": np.arange(8.0)})
    col = df["liability"].to_numpy()
    assert not col.flags.writeable, "precondition: zero-copy to_numpy yields a read-only array"

    out = as_kernel_input(col)

    assert out.flags.writeable
    np.testing.assert_array_equal(out, np.arange(8.0))


# ---------------------------------------------------------------------------
# Boundary behaviour: one signature, identical values
# ---------------------------------------------------------------------------


def test_kernel_compiles_only_mutable_signatures():
    """Regression: a read-only argument must not create a second signature.

    Guards the bug that broke ``fitACE_pafgrs`` under pandas 3 semantics — a
    read-only array reaching the kernel compiled a fresh signature, missing the
    warm cache and raising inside a thread-pinned process.
    """
    try:
        from numba import njit  # noqa: F401
    except ImportError:
        pytest.skip("numba not installed; runtime symbols == Python fallbacks")

    x = np.linspace(0.0, 1.0, 64)
    y = x**1.5

    fast_pearsonr(_readonly(x), _readonly(y))
    fast_pearsonr(x, y)

    readonly_sigs = [
        sig for sig in _pearsonr_core.signatures if any(getattr(arg, "mutable", True) is False for arg in sig)
    ]
    assert not readonly_sigs, f"read-only signatures leaked past as_kernel_input: {readonly_sigs}"


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(fast_pearsonr, id="fast_pearsonr"),
        pytest.param(fast_linregress, id="fast_linregress"),
        pytest.param(safe_corrcoef, id="safe_corrcoef"),
    ],
)
def test_results_identical_regardless_of_writability(call):
    x = np.linspace(0.0, 1.0, 64)
    y = x**1.5 + 0.25

    assert call(_readonly(x), _readonly(y)) == call(x, y)
