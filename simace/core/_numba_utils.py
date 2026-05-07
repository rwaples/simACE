"""Shared Numba-compiled utility functions.

All ``_xxx_python`` functions are pure-Python implementations that can also be
JIT-compiled by Numba.  Cross-function calls use the module-level names
(``_norm_cdf``, ``_owens_t``, etc.) which are set to either the Python or the
JIT-compiled version at import time.  The ``if njit`` block at the bottom
compiles them in dependency order so that each function resolves its callees
to already-jitted versions.
"""

import math

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)
_GOLDEN = 0.3819660112501051  # (3 - sqrt(5)) / 2

# 20-point Gauss-Legendre: positive half of symmetric nodes on [-1, 1]
_GL20_NODES_SYM = np.array(
    [
        0.07652652113349733,
        0.22778585114164507,
        0.37370608871541955,
        0.51086700195082710,
        0.63605368072651503,
        0.74633190646015079,
        0.83911697182221882,
        0.91223442825132591,
        0.96397192727791379,
        0.99312859918509492,
    ]
)
_GL20_WEIGHTS_SYM = np.array(
    [
        0.15275338713072585,
        0.14917298647260375,
        0.14209610931838205,
        0.13168863844917663,
        0.11819453196151842,
        0.10193011981724044,
        0.08327674157670475,
        0.06267204833410907,
        0.04060142980038694,
        0.01761400713915212,
    ]
)

# Transform to [0, 1]: x_i = (1 + t_i)/2, w_i = w_sym_i / 2
_GL20_X01 = np.empty(20)
_GL20_W01 = np.empty(20)
for _i in range(10):
    _GL20_X01[_i] = 0.5 * (1.0 - _GL20_NODES_SYM[9 - _i])
    _GL20_X01[19 - _i] = 0.5 * (1.0 + _GL20_NODES_SYM[9 - _i])
    _GL20_W01[_i] = 0.5 * _GL20_WEIGHTS_SYM[9 - _i]
    _GL20_W01[19 - _i] = 0.5 * _GL20_WEIGHTS_SYM[9 - _i]


# ---------------------------------------------------------------------------
# Leaf functions (no cross-dependencies)
# ---------------------------------------------------------------------------


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


def _norm_cdf_python(x):
    """Normal CDF via erfc. Exact to machine precision."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _norm_sf_python(x):
    """Normal survival function via erfc."""
    return 0.5 * math.erfc(x / _SQRT2)


def _norm_pdf_python(x):
    """Standard normal PDF."""
    return _INV_SQRT2PI * math.exp(-0.5 * x * x)


def _pearsonr_core_python(x, y):
    """Single-pass Pearson r. Returns float."""
    n = len(x)
    sx = 0.0
    sy = 0.0
    for i in range(n):
        sx += x[i]
        sy += y[i]
    mx = sx / n
    my = sy / n
    ss_xx = 0.0
    ss_yy = 0.0
    ss_xy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        ss_xx += dx * dx
        ss_yy += dy * dy
        ss_xy += dx * dy
    denom = ss_xx * ss_yy
    if denom == 0.0:
        return 0.0
    return ss_xy / np.sqrt(denom)


def _linregress_core_python(x, y):
    """Single-pass linear regression. Returns (slope, intercept, r, stderr, t_stat)."""
    n = len(x)
    sx = 0.0
    sy = 0.0
    for i in range(n):
        sx += x[i]
        sy += y[i]
    mx = sx / n
    my = sy / n
    ss_xx = 0.0
    ss_yy = 0.0
    ss_xy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        ss_xx += dx * dx
        ss_yy += dy * dy
        ss_xy += dx * dy
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    r = ss_xy / np.sqrt(ss_xx * ss_yy)
    ss_res = 0.0
    for i in range(n):
        resid = y[i] - (slope * x[i] + intercept)
        ss_res += resid * resid
    s2 = ss_res / (n - 2)
    stderr = np.sqrt(s2 / ss_xx)
    t_stat = slope / stderr
    return slope, intercept, r, stderr, t_stat


# ---------------------------------------------------------------------------
# Dependent functions — call module-level names resolved at JIT-compile time
# ---------------------------------------------------------------------------


def _t_sf_python(t_stat, df):
    """Survival function of the t-distribution (one-tailed), normal approx.

    Accurate to <1e-4 relative error for df >= 30, which covers our use case
    (n >= 10 everywhere, so df >= 8).  For smaller df the error grows but the
    result is only used for display annotation, not inference.
    """
    # Cornish-Fisher first-order correction improves accuracy at moderate df
    z = t_stat * (1.0 - 1.0 / (4.0 * df))
    return _norm_sf(abs(z))


def _owens_t_quad_python(h, a_abs):
    """Owen's T via 20-point Gauss-Legendre quadrature for 0 < a_abs <= 1."""
    half_h2 = 0.5 * h * h
    s = 0.0
    for i in range(20):
        t = a_abs * _GL20_X01[i]
        s += _GL20_W01[i] / (1.0 + t * t) * math.exp(-half_h2 * (1.0 + t * t))
    return a_abs * s * _INV_2PI


def _owens_t_python(h, a):
    """Owen's T function: T(h,a) = (1/2pi) int_0^a exp(-h^2(1+t^2)/2)/(1+t^2) dt."""
    if abs(a) < 1e-15:
        return 0.0
    neg = a < 0.0
    a_abs = abs(a)
    if a_abs <= 1.0:
        result = _owens_t_quad(h, a_abs)
    else:
        # Identity: T(h,a) + T(ah,1/a) = 0.5*Phi(h) + 0.5*Phi(ah) - Phi(h)*Phi(ah)
        ah = a_abs * h
        phi_h = _norm_cdf(h)
        phi_ah = _norm_cdf(ah)
        inner = _owens_t_quad(ah, 1.0 / a_abs)
        result = 0.5 * phi_h + 0.5 * phi_ah - phi_h * phi_ah - inner
    return -result if neg else result


def _bvn_pos_python(h, k, r, sq):
    """P(X > -h, Y > -k) for bivariate normal with correlation r, sq = sqrt(1-r^2)."""
    if h < 1e-15 and k < 1e-15:
        return 0.25 + math.asin(r) / (2.0 * math.pi)
    if h < 1e-15:
        return 0.5 * _norm_cdf(k) - _owens_t(k, -r / sq)
    if k < 1e-15:
        return 0.5 * _norm_cdf(h) - _owens_t(h, -r / sq)
    return (
        0.5 * _norm_cdf(h)
        + 0.5 * _norm_cdf(k)
        - _owens_t(h, (k - r * h) / (h * sq))
        - _owens_t(k, (h - r * k) / (k * sq))
    )


def _bvn_cdf_python(h, k, r):
    """Bivariate normal CDF: P(X <= h, Y <= k) with correlation r."""
    if abs(r) < 1e-15:
        return _norm_cdf(h) * _norm_cdf(k)
    sq = math.sqrt(1.0 - r * r)
    if h < 0 and k < 0:
        return 1.0 - _norm_cdf(-h) - _norm_cdf(-k) + _bvn_pos(-h, -k, r, sq)
    if h < 0:
        return _norm_cdf(k) - _bvn_pos(-h, k, -r, sq)
    if k < 0:
        return _norm_cdf(h) - _bvn_pos(h, -k, -r, sq)
    return _bvn_pos(h, k, r, sq)


def _tetrachoric_nll_python(r, t_a, t_b, phi_ta, phi_tb, both_positive, n11, n10, n01, n00):
    """Negative log-likelihood for tetrachoric correlation at a given r."""
    if both_positive:
        sq = math.sqrt(1.0 - r * r)
        p00 = (
            0.5 * phi_ta
            + 0.5 * phi_tb
            - _owens_t(t_a, (t_b - r * t_a) / (t_a * sq))
            - _owens_t(t_b, (t_a - r * t_b) / (t_b * sq))
        )
    else:
        p00 = _bvn_cdf(t_a, t_b, r)
    p01 = phi_ta - p00
    p10 = phi_tb - p00
    p11 = 1.0 - p00 - p01 - p10
    eps = 1e-15
    lp11 = math.log(p11) if p11 > eps else math.log(eps)
    lp10 = math.log(p10) if p10 > eps else math.log(eps)
    lp01 = math.log(p01) if p01 > eps else math.log(eps)
    lp00 = math.log(p00) if p00 > eps else math.log(eps)
    return -(n11 * lp11 + n10 * lp10 + n01 * lp01 + n00 * lp00)


def _tetrachoric_core_python(n11, n10, n01, n00, t_a, t_b, phi_ta, phi_tb):
    """Full tetrachoric correlation + SE via MLE. Returns (r, se)."""
    both_pos = t_a > 1e-15 and t_b > 1e-15

    # Brent's bounded minimization inlined for tetrachoric NLL
    xa = -0.999
    xb = 0.999
    x = w = v = xa + _GOLDEN * (xb - xa)
    fw = fv = fx = _tetrachoric_nll(x, t_a, t_b, phi_ta, phi_tb, both_pos, n11, n10, n01, n00)
    d = 0.0
    e = 0.0

    for _ in range(500):
        midpoint = 0.5 * (xa + xb)
        tol1 = 1e-10 * abs(x) + 1e-10
        tol2 = 2.0 * tol1

        if abs(x - midpoint) <= (tol2 - 0.5 * (xb - xa)):
            break

        if abs(e) > tol1:
            r_b = (x - w) * (fx - fv)
            q_b = (x - v) * (fx - fw)
            p_b = (x - v) * q_b - (x - w) * r_b
            q_b = 2.0 * (q_b - r_b)
            if q_b > 0.0:
                p_b = -p_b
            else:
                q_b = -q_b
            if abs(p_b) < abs(0.5 * q_b * e) and p_b > q_b * (xa - x) and p_b < q_b * (xb - x):
                e = d
                d = p_b / q_b
            else:
                e = (xb - x) if x < midpoint else (xa - x)
                d = _GOLDEN * e
        else:
            e = (xb - x) if x < midpoint else (xa - x)
            d = _GOLDEN * e

        u = (x + d) if abs(d) >= tol1 else (x + tol1 if d > 0 else x - tol1)

        fu = _tetrachoric_nll(u, t_a, t_b, phi_ta, phi_tb, both_pos, n11, n10, n01, n00)

        if fu <= fx:
            if u < x:
                xb = x
            else:
                xa = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                xa = u
            else:
                xb = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v in (x, w):
                v = u
                fv = fu

    r = x
    if r != r:  # NaN check
        return np.nan, np.nan

    # SE from Fisher information
    one_minus_r2 = 1.0 - r * r
    if one_minus_r2 <= 0:
        return r, np.nan

    bvn_pdf = (
        _INV_2PI
        / math.sqrt(one_minus_r2)
        * math.exp(-(t_a * t_a - 2.0 * r * t_a * t_b + t_b * t_b) / (2.0 * one_minus_r2))
    )

    if both_pos:
        sq = math.sqrt(one_minus_r2)
        p00 = (
            0.5 * phi_ta
            + 0.5 * phi_tb
            - _owens_t(t_a, (t_b - r * t_a) / (t_a * sq))
            - _owens_t(t_b, (t_a - r * t_b) / (t_b * sq))
        )
    else:
        p00 = _bvn_cdf(t_a, t_b, r)
    p01 = phi_ta - p00
    p10 = phi_tb - p00
    p11 = 1.0 - p00 - p01 - p10
    denom = p00 * p01 * p10 * p11
    n_pairs = n11 + n10 + n01 + n00
    if denom <= 0:
        return r, np.nan
    se = 1.0 / math.sqrt(n_pairs * bvn_pdf * bvn_pdf / denom)
    return r, se


# ---------------------------------------------------------------------------
# JIT compilation — dependency order matters!
# ---------------------------------------------------------------------------

# Set module-level names to Python fallbacks first
_ndtri_approx = _ndtri_approx_python
_norm_cdf = _norm_cdf_python
_norm_sf = _norm_sf_python
_norm_pdf = _norm_pdf_python
_pearsonr_core = _pearsonr_core_python
_linregress_core = _linregress_core_python
_t_sf = _t_sf_python
_owens_t_quad = _owens_t_quad_python
_owens_t = _owens_t_python
_bvn_pos = _bvn_pos_python
_bvn_cdf = _bvn_cdf_python
_tetrachoric_nll = _tetrachoric_nll_python
_tetrachoric_core = _tetrachoric_core_python

if njit is not None:
    # Tier 0: no dependencies
    _ndtri_approx = njit(cache=True)(_ndtri_approx_python)
    _norm_cdf = njit(cache=True)(_norm_cdf_python)
    _norm_sf = njit(cache=True)(_norm_sf_python)
    _norm_pdf = njit(cache=True)(_norm_pdf_python)
    _pearsonr_core = njit(cache=True)(_pearsonr_core_python)
    _linregress_core = njit(cache=True)(_linregress_core_python)

    # Tier 1: depends on _norm_sf / _norm_cdf
    _t_sf = njit(cache=True)(_t_sf_python)
    _owens_t_quad = njit(cache=True)(_owens_t_quad_python)

    # Tier 2: depends on _owens_t_quad, _norm_cdf
    _owens_t = njit(cache=True)(_owens_t_python)

    # Tier 3: depends on _norm_cdf, _owens_t
    _bvn_pos = njit(cache=True)(_bvn_pos_python)

    # Tier 4: depends on _norm_cdf, _bvn_pos
    _bvn_cdf = njit(cache=True)(_bvn_cdf_python)

    # Tier 5: depends on _owens_t, _bvn_cdf
    _tetrachoric_nll = njit(cache=True)(_tetrachoric_nll_python)

    # Tier 6: depends on _tetrachoric_nll, _owens_t, _bvn_cdf
    _tetrachoric_core = njit(cache=True)(_tetrachoric_core_python)
