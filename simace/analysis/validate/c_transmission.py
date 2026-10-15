"""C transmission check: regress offspring C on both parents' C (ADR 0025).

The offspring C is ``b_m * C_mother + b_f * C_father + u`` with ``u`` drawn
per maternal household independently of the parents, so OLS of child C on
(mother C, father C) is unbiased for ``(b_m, b_f)`` under any mating model.
Children of one mother share ``u``, so the standard errors are
cluster-robust (CR1) over maternal households.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core.numerics import _ZERO_VAR_THRESHOLD
from simace.core.relationships import transmission_coefficients

from ._common import _info, _result

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from simace.core.pedigree_arrays import PedigreeArrays

#: Fewer maternal households than this leaves the cluster-robust SE unreliable.
MIN_HOUSEHOLDS = 50

#: Standard errors either side of the configured coefficient that still pass.
SE_MULTIPLIER = 4.0

#: Tolerance floor, so an exact fit (no fresh draw, SE ~ 0) is judged on numerics.
TOLERANCE_FLOOR = 1e-6


def cluster_codes(keys: np.ndarray) -> np.ndarray:
    """Dense codes ``0..G-1`` for the distinct ``keys``, by a stable sort (bare ``np.unique`` is slow on big int64)."""
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    codes = np.empty(len(keys), dtype=np.int64)
    codes[order] = np.cumsum(np.r_[False, ordered[1:] != ordered[:-1]])
    return codes


def cluster_robust_ols(y: np.ndarray, X: np.ndarray, cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return OLS coefficients of ``y`` on ``X`` and their CR1 cluster-robust standard errors.

    ``X`` carries its own intercept column and ``cluster`` holds dense codes
    ``0..G-1`` (:func:`cluster_codes`). CR1 scales the sandwich by
    ``G / (G - 1) * (n - 1) / (n - k)`` for ``G`` clusters, ``n`` rows and
    ``k`` columns.
    """
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ (X.T @ y)
    scores = X * (y - X @ beta)[:, None]
    n_clusters = int(cluster.max()) + 1
    summed = np.column_stack([np.bincount(cluster, weights=scores[:, j], minlength=n_clusters) for j in range(k)])
    scale = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)
    cov = scale * bread @ (summed.T @ summed) @ bread
    return beta, np.sqrt(np.diag(cov))


def validate_c_transmission(
    df: pd.DataFrame | pl.DataFrame, params: dict[str, Any], ped: PedigreeArrays
) -> dict[str, Any]:
    """Check that offspring C regresses on parental C with the configured ``(b_m, b_f)``, per trait.

    Informational (not scored) when the trait has no C transmission, when a
    parent's C has no variance, or when fewer than :data:`MIN_HOUSEHOLDS`
    mothers have recorded children.
    """
    mother = df["mother"].to_numpy()
    father = df["father"].to_numpy()
    kids = np.flatnonzero((mother != -1) & ped.contains(mother) & ped.contains(father))
    household = cluster_codes(mother[kids])
    n_households = int(household.max()) + 1 if len(kids) else 0

    results: dict[str, Any] = {}
    for t in (1, 2):
        name = f"c{t}_transmission"
        b_m, b_f = transmission_coefficients(params, t)
        if b_m == b_f == 0.0:
            results[name] = _info(f"No C{t} transmission configured")
            continue
        if n_households < MIN_HOUSEHOLDS:
            results[name] = _info(f"C{t} transmission not checked: {n_households} households < {MIN_HOUSEHOLDS}")
            continue
        child = df[f"C{t}"].to_numpy()[kids].astype(np.float64)
        mom = ped.gather(f"C{t}", mother[kids]).astype(np.float64)
        dad = ped.gather(f"C{t}", father[kids]).astype(np.float64)
        if min(np.var(mom), np.var(dad)) < _ZERO_VAR_THRESHOLD:
            results[name] = _info(f"C{t} transmission not checked: parental C{t} has no variance")
            continue
        X = np.column_stack([np.ones_like(child), mom, dad])
        beta, se = cluster_robust_ols(child, X, household)
        expected = (b_m, b_f)
        tolerance = [max(SE_MULTIPLIER * s, TOLERANCE_FLOOR) for s in se[1:]]
        passed = all(abs(beta[j + 1] - expected[j]) <= tolerance[j] for j in range(2))
        results[name] = _result(
            passed,
            f"C{t} on (mother, father) C: b_m {beta[1]:.4f} (expected {b_m}, tol {tolerance[0]:.2g}), "
            f"b_f {beta[2]:.4f} (expected {b_f}, tol {tolerance[1]:.2g}); "
            f"cluster-robust over {n_households} households",
            observed=[float(beta[1]), float(beta[2])],
            expected=[b_m, b_f],
            tolerance=tolerance,
            n_offspring=len(kids),
            n_households=n_households,
        )
    return results
