"""Python ↔ JIT parity for simace.core._numba_utils.

Each numerical primitive in ``_numba_utils`` ships in two forms:

* ``_xxx_python`` — pure-Python implementation (the reference / fallback);
* ``_xxx``        — module-level alias bound to either the njit-compiled
                    version (when numba is installed, the normal case) or
                    the Python fallback otherwise.

These tests pin each `_xxx_python` against:

  1. the runtime ``_xxx`` symbol (catching JIT/Python divergence);
  2. an independent reference (scipy / numpy / mpmath-grade closed forms)
     where a closed-form check is meaningful.

Calling the ``_python`` symbols directly also makes their source lines
visible to coverage instrumentation — by default those lines are dark
because the runtime symbols dispatch through njit.
"""

import math

import numpy as np
import pytest
from scipy import stats

from simace.core import _numba_utils as nu

# ---------------------------------------------------------------------------
# Tier 0 — leaf functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1e-6, 0.001, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.999, 1 - 1e-6])
def test_ndtri_approx_matches_scipy(p):
    py = nu._ndtri_approx_python(p)
    jit = nu._ndtri_approx(p)
    ref = stats.norm.ppf(p)
    assert py == pytest.approx(jit, abs=1e-12)
    # Acklam approximation: ~1e-9 absolute accuracy
    assert py == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("x", [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
def test_norm_cdf_matches_scipy(x):
    py = nu._norm_cdf_python(x)
    jit = nu._norm_cdf(x)
    ref = stats.norm.cdf(x)
    assert py == pytest.approx(jit, abs=1e-15)
    assert py == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_norm_sf_matches_scipy(x):
    py = nu._norm_sf_python(x)
    jit = nu._norm_sf(x)
    ref = stats.norm.sf(x)
    assert py == pytest.approx(jit, abs=1e-15)
    assert py == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("x", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_norm_pdf_matches_scipy(x):
    py = nu._norm_pdf_python(x)
    jit = nu._norm_pdf(x)
    ref = stats.norm.pdf(x)
    assert py == pytest.approx(jit, abs=1e-15)
    assert py == pytest.approx(ref, abs=1e-15)


def test_pearsonr_core_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 0.6 * x + rng.standard_normal(500)
    py = nu._pearsonr_core_python(x, y)
    jit = nu._pearsonr_core(x, y)
    ref = float(stats.pearsonr(x, y).statistic)
    assert py == pytest.approx(jit, abs=1e-12)
    assert py == pytest.approx(ref, abs=1e-12)


def test_pearsonr_core_zero_variance_returns_zero():
    x = np.zeros(10)
    y = np.arange(10, dtype=float)
    assert nu._pearsonr_core_python(x, y) == 0.0
    assert nu._pearsonr_core(x, y) == 0.0


def test_linregress_core_matches_scipy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    y = 2.5 * x + 1.0 + rng.standard_normal(300) * 0.1
    py = nu._linregress_core_python(x, y)
    jit = nu._linregress_core(x, y)
    ref = stats.linregress(x, y)

    # py and jit are 5-tuples (slope, intercept, r, stderr, t_stat)
    for a, b in zip(py, jit, strict=True):
        assert a == pytest.approx(b, abs=1e-12)

    assert py[0] == pytest.approx(ref.slope, abs=1e-10)
    assert py[1] == pytest.approx(ref.intercept, abs=1e-10)
    assert py[2] == pytest.approx(ref.rvalue, abs=1e-10)
    assert py[3] == pytest.approx(ref.stderr, abs=1e-10)


# ---------------------------------------------------------------------------
# Tier 1 — depends on Tier 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("t_stat", "df"), [(0.5, 30), (-1.96, 100), (3.0, 50), (0.0, 20)])
def test_t_sf_matches_jit(t_stat, df):
    # Pure parity: this is a coarse Cornish-Fisher approximation used
    # only for display, so we don't pin it to scipy.t.sf.
    py = nu._t_sf_python(t_stat, df)
    jit = nu._t_sf(t_stat, df)
    assert py == pytest.approx(jit, abs=1e-15)


@pytest.mark.parametrize(("h", "a"), [(0.5, 0.3), (1.0, 0.7), (2.0, 1.0), (0.1, 0.5)])
def test_owens_t_quad_matches_jit(h, a):
    py = nu._owens_t_quad_python(h, a)
    jit = nu._owens_t_quad(h, a)
    assert py == pytest.approx(jit, abs=1e-15)


# ---------------------------------------------------------------------------
# Tier 2 — Owen's T full
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("h", "a"),
    [
        (0.0, 0.5),  # |a| ≤ 1 path
        (0.5, -0.3),  # negative a, |a| ≤ 1
        (0.5, 1.5),  # |a| > 1 identity path
        (1.0, -2.0),  # negative a, |a| > 1
        (2.0, 1e-16),  # near-zero a → 0.0
    ],
)
def test_owens_t_matches_jit(h, a):
    py = nu._owens_t_python(h, a)
    jit = nu._owens_t(h, a)
    assert py == pytest.approx(jit, abs=1e-15)


def test_owens_t_zero_a_returns_zero():
    assert nu._owens_t_python(1.0, 0.0) == 0.0


def test_owens_t_negation_symmetry():
    # T(h, -a) = -T(h, a)
    h = 1.5
    for a in [0.3, 0.8, 1.5, 3.0]:
        assert nu._owens_t_python(h, -a) == pytest.approx(-nu._owens_t_python(h, a), abs=1e-12)


# ---------------------------------------------------------------------------
# Tier 3-4 — bivariate normal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("h", "k", "r"),
    [
        (0.5, 0.5, 0.3),
        (1.0, 1.0, 0.5),
        (0.0, 0.0, 0.6),
        (0.5, 0.0, 0.4),
        (0.0, 0.7, -0.3),
    ],
)
def test_bvn_pos_matches_jit(h, k, r):
    sq = math.sqrt(1.0 - r * r)
    py = nu._bvn_pos_python(h, k, r, sq)
    jit = nu._bvn_pos(h, k, r, sq)
    assert py == pytest.approx(jit, abs=1e-14)


@pytest.mark.parametrize(
    ("h", "k", "r"),
    [
        (0.5, 0.5, 0.3),
        (-1.0, -1.0, 0.4),
        (1.0, -0.5, -0.2),
        (-0.5, 1.0, 0.6),
        (0.0, 0.0, 0.0),  # r ≈ 0 fast path
    ],
)
def test_bvn_cdf_matches_scipy(h, k, r):
    py = nu._bvn_cdf_python(h, k, r)
    jit = nu._bvn_cdf(h, k, r)
    ref = float(
        stats.multivariate_normal.cdf(
            [h, k],
            mean=[0, 0],
            cov=[[1, r], [r, 1]],
        )
    )
    assert py == pytest.approx(jit, abs=1e-12)
    assert py == pytest.approx(ref, abs=1e-6)


# ---------------------------------------------------------------------------
# Tier 5-6 — tetrachoric NLL and full estimator
# ---------------------------------------------------------------------------


def test_tetrachoric_nll_matches_jit_both_positive_branch():
    # All four cells positive → both_positive=True path
    n11, n10, n01, n00 = 50, 30, 25, 100
    n_total = n11 + n10 + n01 + n00
    p_a = (n11 + n10) / n_total
    p_b = (n11 + n01) / n_total
    t_a = nu._ndtri_approx_python(p_a)
    t_b = nu._ndtri_approx_python(p_b)
    phi_ta = nu._norm_cdf_python(t_a)
    phi_tb = nu._norm_cdf_python(t_b)

    for r in [-0.2, 0.0, 0.3, 0.5, 0.8]:
        py = nu._tetrachoric_nll_python(r, t_a, t_b, phi_ta, phi_tb, True, n11, n10, n01, n00)
        jit = nu._tetrachoric_nll(r, t_a, t_b, phi_ta, phi_tb, True, n11, n10, n01, n00)
        assert py == pytest.approx(jit, abs=1e-10)


def test_tetrachoric_nll_negative_threshold_branch():
    # Force at least one threshold ≤ 0 to take the BVN path
    n11, n10, n01, n00 = 200, 30, 25, 50  # majority class affected → t_a < 0
    n_total = n11 + n10 + n01 + n00
    t_a = nu._ndtri_approx_python((n11 + n10) / n_total)
    t_b = nu._ndtri_approx_python((n11 + n01) / n_total)
    phi_ta = nu._norm_cdf_python(t_a)
    phi_tb = nu._norm_cdf_python(t_b)
    py = nu._tetrachoric_nll_python(0.4, t_a, t_b, phi_ta, phi_tb, False, n11, n10, n01, n00)
    jit = nu._tetrachoric_nll(0.4, t_a, t_b, phi_ta, phi_tb, False, n11, n10, n01, n00)
    assert py == pytest.approx(jit, abs=1e-10)


def test_tetrachoric_core_matches_jit():
    # Synthesize a 2×2 from a known r via the same cell-probability decomposition
    # the NLL uses internally, then estimate r back from those counts.
    r_true = 0.45
    p_a = 0.3  # P(a affected) = P(X > t_a)
    p_b = 0.4  # P(b affected) = P(Y > t_b)
    t_a = nu._ndtri_approx_python(1 - p_a)
    t_b = nu._ndtri_approx_python(1 - p_b)
    phi_ta = nu._norm_cdf_python(t_a)  # = 1 - p_a
    phi_tb = nu._norm_cdf_python(t_b)  # = 1 - p_b

    # Match the NLL convention exactly:
    #   p00 = bvn_cdf(t_a, t_b, r) = P(neither affected)
    #   p01 = phi_ta - p00  = P(a not, b affected)
    #   p10 = phi_tb - p00  = P(b not, a affected)
    #   p11 = 1 - p00 - p01 - p10
    n_total = 5000
    p00 = nu._bvn_cdf_python(t_a, t_b, r_true)
    p01 = phi_ta - p00
    p10 = phi_tb - p00

    n00 = round(p00 * n_total)
    n01 = round(p01 * n_total)
    n10 = round(p10 * n_total)
    n11 = n_total - n00 - n01 - n10

    r_py, se_py = nu._tetrachoric_core_python(n11, n10, n01, n00, t_a, t_b, phi_ta, phi_tb)
    r_jit, se_jit = nu._tetrachoric_core(n11, n10, n01, n00, t_a, t_b, phi_ta, phi_tb)

    assert r_py == pytest.approx(r_jit, abs=1e-8)
    assert se_py == pytest.approx(se_jit, abs=1e-8)
    # Recovered r close to truth (rounding to integer counts adds some bias)
    assert r_py == pytest.approx(r_true, abs=0.05)
    assert se_py > 0


# ---------------------------------------------------------------------------
# Cross-check: the runtime symbols are the JIT versions when numba present
# ---------------------------------------------------------------------------


def test_runtime_symbols_are_jit_compiled():
    try:
        from numba import njit  # noqa: F401
    except ImportError:
        pytest.skip("numba not installed; runtime symbols == Python fallbacks")

    # When numba is available, every runtime symbol should differ in identity
    # from its Python fallback (they're separate JIT-compiled wrappers).
    pairs = [
        (nu._ndtri_approx, nu._ndtri_approx_python),
        (nu._norm_cdf, nu._norm_cdf_python),
        (nu._norm_sf, nu._norm_sf_python),
        (nu._norm_pdf, nu._norm_pdf_python),
        (nu._pearsonr_core, nu._pearsonr_core_python),
        (nu._linregress_core, nu._linregress_core_python),
        (nu._t_sf, nu._t_sf_python),
        (nu._owens_t_quad, nu._owens_t_quad_python),
        (nu._owens_t, nu._owens_t_python),
        (nu._bvn_pos, nu._bvn_pos_python),
        (nu._bvn_cdf, nu._bvn_cdf_python),
        (nu._tetrachoric_nll, nu._tetrachoric_nll_python),
        (nu._tetrachoric_core, nu._tetrachoric_core_python),
    ]
    for jit, py in pairs:
        assert jit is not py, f"{py.__name__}: runtime symbol still bound to Python fallback"
