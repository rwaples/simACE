"""Benchmark: scipy linregress/pearsonr vs numpy/numba implementations."""

import timeit

import numba as nb
import numpy as np
from scipy.stats import linregress, pearsonr
from scipy.stats import t as t_dist

from sim_ace.core.utils import fast_linregress as fast_linregress_prod
from sim_ace.core.utils import fast_pearsonr as fast_pearsonr_prod


def np_linregress(x: np.ndarray, y: np.ndarray):
    """Numpy-only linear regression returning slope, intercept, r, stderr, pvalue."""
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    dx = x - x_mean
    dy = y - y_mean
    ss_xx = dx @ dx
    ss_yy = dy @ dy
    ss_xy = dx @ dy
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r = ss_xy / np.sqrt(ss_xx * ss_yy)
    # Residual standard error of slope
    resid = y - (slope * x + intercept)
    s2 = (resid @ resid) / (n - 2)
    stderr = np.sqrt(s2 / ss_xx)
    # Two-sided p-value from t-distribution
    t_stat = slope / stderr
    pvalue = 2.0 * t_dist.sf(np.abs(t_stat), df=n - 2)
    return slope, intercept, r, stderr, pvalue


@nb.njit(cache=True)
def _nb_linregress_core(x, y):
    """Numba-jitted core: returns slope, intercept, r, stderr, t_stat."""
    n = len(x)
    sx = 0.0
    sy = 0.0
    for i in range(n):
        sx += x[i]
        sy += y[i]
    x_mean = sx / n
    y_mean = sy / n
    ss_xx = 0.0
    ss_yy = 0.0
    ss_xy = 0.0
    for i in range(n):
        dx = x[i] - x_mean
        dy = y[i] - y_mean
        ss_xx += dx * dx
        ss_yy += dy * dy
        ss_xy += dx * dy
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r = ss_xy / np.sqrt(ss_xx * ss_yy)
    # Residual sum of squares in single pass
    ss_res = 0.0
    for i in range(n):
        resid = y[i] - (slope * x[i] + intercept)
        ss_res += resid * resid
    s2 = ss_res / (n - 2)
    stderr = np.sqrt(s2 / ss_xx)
    t_stat = slope / stderr
    return slope, intercept, r, stderr, t_stat


def nb_linregress(x: np.ndarray, y: np.ndarray):
    """Numba linear regression with p-value."""
    slope, intercept, r, stderr, t_stat = _nb_linregress_core(x, y)
    pvalue = 2.0 * t_dist.sf(abs(t_stat), df=len(x) - 2)
    return slope, intercept, r, stderr, pvalue


@nb.njit(cache=True)
def _nb_pearsonr_core(x, y):
    """Numba-jitted Pearson r + t-statistic."""
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
    r = ss_xy / np.sqrt(ss_xx * ss_yy)
    t_stat = r * np.sqrt((n - 2) / (1.0 - r * r))
    return r, t_stat


def nb_pearsonr(x: np.ndarray, y: np.ndarray):
    """Numba Pearson r with p-value."""
    r, t_stat = _nb_pearsonr_core(x, y)
    pvalue = 2.0 * t_dist.sf(abs(t_stat), df=len(x) - 2)
    return r, pvalue


@nb.njit(cache=True)
def nb_pearsonr_no_p(x, y):
    """Numba r only — no p-value, no scipy."""
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
    return ss_xy / np.sqrt(ss_xx * ss_yy)


def np_pearsonr(x: np.ndarray, y: np.ndarray):
    """Numpy-only Pearson r with p-value."""
    n = len(x)
    dx = x - x.mean()
    dy = y - y.mean()
    r = (dx @ dy) / np.sqrt((dx @ dx) * (dy @ dy))
    t_stat = r * np.sqrt((n - 2) / (1.0 - r * r))
    pvalue = 2.0 * t_dist.sf(np.abs(t_stat), df=n - 2)
    return r, pvalue


def np_pearsonr_no_p(x: np.ndarray, y: np.ndarray):
    """Numpy-only Pearson r without p-value (like np.corrcoef but cheaper)."""
    dx = x - x.mean()
    dy = y - y.mean()
    return (dx @ dy) / np.sqrt((dx @ dx) * (dy @ dy))


def bench(label, func, args, n_runs=500):
    # Warm up
    func(*args)
    t = timeit.timeit(lambda: func(*args), number=n_runs)
    us = t / n_runs * 1e6
    print(f"  {label:40s} {us:8.1f} µs")
    return us


def validate(n=10_000):
    """Check that all implementations match scipy."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n)
    y = 0.5 * x + rng.standard_normal(n)

    # Warm up numba
    _nb_linregress_core(x, y)
    _nb_pearsonr_core(x, y)
    nb_pearsonr_no_p(x, y)

    sp = linregress(x, y)
    np_s, np_i, np_r, np_se, np_p = np_linregress(x, y)
    nb_s, nb_i, nb_r, nb_se, nb_p = nb_linregress(x, y)
    print("linregress validation:")
    print(f"  slope:     scipy={sp.slope:.8f}  numpy={np_s:.8f}  numba={nb_s:.8f}")
    print(f"  intercept: scipy={sp.intercept:.8f}  numpy={np_i:.8f}  numba={nb_i:.8f}")
    print(f"  r:         scipy={sp.rvalue:.8f}  numpy={np_r:.8f}  numba={nb_r:.8f}")
    print(f"  stderr:    scipy={sp.stderr:.8f}  numpy={np_se:.8f}  numba={nb_se:.8f}")
    print(f"  pvalue:    scipy={sp.pvalue:.8e}  numpy={np_p:.8e}  numba={nb_p:.8e}")

    sp_r, sp_p = pearsonr(x, y)
    np_r2, np_p2 = np_pearsonr(x, y)
    nb_r2, nb_p2 = nb_pearsonr(x, y)
    print("\npearsonr validation:")
    print(f"  r:      scipy={sp_r:.8f}  numpy={np_r2:.8f}  numba={nb_r2:.8f}")
    print(f"  pvalue: scipy={sp_p:.8e}  numpy={np_p2:.8e}  numba={nb_p2:.8e}")
    print()


def main():
    validate()

    rng = np.random.default_rng(42)

    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        x = rng.standard_normal(n).astype(np.float64)
        y = (0.5 * x + rng.standard_normal(n)).astype(np.float64)

        print(f"n = {n:>10,}")

        n_runs = max(50, 5000 // max(1, n // 1000))

        # --- linregress ---
        t_scipy = bench("scipy.stats.linregress", linregress, (x, y), n_runs)
        t_numpy = bench("np_linregress (with p)", np_linregress, (x, y), n_runs)
        t_numba = bench("nb_linregress (with p)", nb_linregress, (x, y), n_runs)
        t_fast = bench("fast_linregress (production)", fast_linregress_prod, (x, y), n_runs)
        print(f"  {'numpy vs scipy':40s} {t_scipy / t_numpy:8.2f}x")
        print(f"  {'numba vs scipy':40s} {t_scipy / t_numba:8.2f}x")
        print(f"  {'fast_linregress vs scipy':40s} {t_scipy / t_fast:8.2f}x")

        # --- pearsonr ---
        t_scipy_r = bench("scipy.stats.pearsonr", pearsonr, (x, y), n_runs)
        bench("np_pearsonr (with p)", np_pearsonr, (x, y), n_runs)
        bench("nb_pearsonr (with p)", nb_pearsonr, (x, y), n_runs)
        t_fast_r = bench("fast_pearsonr (production)", fast_pearsonr_prod, (x, y), n_runs)
        t_numba_r_nop = bench("_pearsonr_core (r only)", nb_pearsonr_no_p, (x, y), n_runs)
        t_corrcoef = bench("np.corrcoef", lambda x, y: np.corrcoef(x, y)[0, 1], (x, y), n_runs)
        print(f"  {'fast_pearsonr vs scipy':40s} {t_scipy_r / t_fast_r:8.2f}x")
        print(f"  {'_pearsonr_core vs corrcoef':40s} {t_corrcoef / t_numba_r_nop:8.2f}x")
        print()


if __name__ == "__main__":
    main()
