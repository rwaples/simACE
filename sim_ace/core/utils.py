"""Shared utility functions for sim_ace."""

from __future__ import annotations

__all__ = [
    "PAIR_TYPES",
    "expected_mate_corr_matrix",
    "fast_linregress",
    "fast_pearsonr",
    "get_nested",
    "optimize_dtypes",
    "safe_corrcoef",
    "safe_linregress",
    "save_parquet",
    "to_native",
    "validation_result",
    "yaml_loader",
]

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from sim_ace.core._numba_utils import _linregress_core, _pearsonr_core, _t_sf

if TYPE_CHECKING:
    import pandas as pd

# Canonical 7 relationship pair types used for tetrachoric / liability correlations
PAIR_TYPES: list[str] = [
    "MZ",
    "FS",
    "MO",
    "FO",
    "MHS",
    "PHS",
    "1C",
]


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning nan if either array has zero variance.

    Args:
        x: First array of observations.
        y: Second array of observations, same length as *x*.

    Returns:
        Pearson correlation coefficient, or nan if variance is near-zero.
    """
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(_pearsonr_core(x, y))


def safe_linregress(x: np.ndarray, y: np.ndarray) -> Any:
    """Run linear regression, returning None if x has zero variance.

    Args:
        x: Independent variable array.
        y: Dependent variable array, same length as *x*.

    Returns:
        ``scipy.stats.LinregressResult`` or None if *x* has near-zero variance.
    """
    if np.std(x) < 1e-10:
        return None
    return stats.linregress(x, y)


def fast_linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fast linear regression via numba-accelerated core.

    Args:
        x: Independent variable array.
        y: Dependent variable array, same length as *x*.

    Returns:
        Tuple of (slope, intercept, r, stderr, pvalue).
    """
    slope, intercept, r, stderr, t_stat = _linregress_core(x, y)
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
    r = float(_pearsonr_core(x, y))
    n = len(x)
    denom = 1.0 - r * r
    if denom < 1e-30 or n <= 2:
        return r, 0.0
    t_stat = r * np.sqrt((n - 2) / denom)
    pvalue = float(2.0 * _t_sf(abs(t_stat), n - 2))
    return r, pvalue


def to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for YAML serialization.

    Args:
        obj: Value or nested structure (dict, list, ndarray, numpy scalar).

    Returns:
        Equivalent structure with all numpy types replaced by Python builtins.
    """
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def validation_result(passed: bool, details: str, **extra: Any) -> dict[str, Any]:
    """Build a standardized validation result dict.

    Args:
        passed: Whether the validation check passed.
        details: Human-readable description of the result.
        **extra: Additional key-value pairs merged into the result.

    Returns:
        Dict with keys ``'passed'``, ``'details'``, plus any *extra* entries.
    """
    d: dict[str, Any] = {"passed": passed, "details": details}
    d.update(extra)
    return d


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast DataFrame columns for compact parquet storage.

    Dtype strategy (matching pedigree generation-time dtypes):
    - int32 for ID columns and generation (supports up to 2.1B individuals)
    - int8 for sex (0/1)
    - float32 for ACE components and event times (~7 significant digits)
    - float64 for liabilities (full precision for phenotype models)

    Args:
        df: Pedigree DataFrame to optimize in-place.

    Returns:
        The same DataFrame with downcasted column dtypes.
    """
    int32_cols = ["id", "mother", "father", "twin", "household_id", "generation"]
    int8_cols = ["sex"]
    float32_cols = [
        "A1",
        "C1",
        "E1",
        "A2",
        "C2",
        "E2",
        "t1",
        "t2",
        "death_age",
        "t_observed1",
        "t_observed2",
    ]
    for c in int32_cols:
        if c in df.columns:
            df[c] = df[c].astype("int32")
    for c in int8_cols:
        if c in df.columns:
            df[c] = df[c].astype("int8")
    for c in float32_cols:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    return df


def save_parquet(df: pd.DataFrame, path: Any, **kwargs: Any) -> None:
    """Save DataFrame as parquet with optimized dtypes and zstd compression.

    Calls ``optimize_dtypes`` before writing to minimize file size.

    Args:
        df: DataFrame to save.
        path: Output file path.
        **kwargs: Extra keyword arguments passed to ``DataFrame.to_parquet``.
    """
    optimize_dtypes(df)
    df.to_parquet(path, index=False, compression="zstd", **kwargs)


def yaml_loader() -> type:
    """Return the fastest available YAML SafeLoader.

    Returns:
        ``yaml.CSafeLoader`` if the C extension is available, else ``yaml.SafeLoader``.
    """
    import yaml

    return getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def get_nested(d: Any, *keys: str, default: Any = None) -> Any:
    """Traverse nested dicts by key path, returning default if any key is missing.

    Args:
        d: Root dict (or nested structure) to traverse.
        *keys: Sequence of keys to follow.
        default: Value returned if any key is absent.

    Returns:
        The value at the end of the key path, or *default*.
    """
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def expected_mate_corr_matrix(
    assort1: float,
    assort2: float,
    rA: float,
    rC: float,
    A1: float,
    C1: float,
    A2: float,
    C2: float,
    assort_matrix: np.ndarray | list | None = None,
    rE: float = 0.0,
    E1: float = 0.0,
    E2: float = 0.0,
) -> np.ndarray:
    """Compute the 2x2 expected mate liability correlation matrix.

    Returns E[corr(F_i, M_j)] for i,j in {1,2} given assortative mating
    parameters and ACE variance components.

    With the 4-variate copula algorithm, assort1 and assort2 are target
    Pearson mate correlations. The cross-mate cross-trait correlation follows
    from the mechanistic path: c = rho_w * sqrt(|r1*r2|) * sign(r1*r2),
    where rho_w is the within-person cross-trait liability correlation.

    When ``assort_matrix`` is provided, it is returned directly (the user
    has specified the full R_mf).

    Args:
        assort1: Target mate Pearson correlation for trait 1.
        assort2: Target mate Pearson correlation for trait 2.
        rA: Genetic correlation between traits.
        rC: Shared-environment correlation between traits.
        A1: Additive genetic variance for trait 1.
        C1: Shared-environment variance for trait 1.
        A2: Additive genetic variance for trait 2.
        C2: Shared-environment variance for trait 2.
        assort_matrix: If provided, returned directly as the full R_mf matrix.
        rE: Unique-environment correlation between traits.
        E1: Unique-environment variance for trait 1.
        E2: Unique-environment variance for trait 2.

    Returns:
        2x2 array of expected mate liability correlations ``E[corr(F_i, M_j)]``.
    """
    if assort_matrix is not None:
        return np.asarray(assort_matrix, dtype=np.float64)

    if assort1 == 0 and assort2 == 0:
        return np.zeros((2, 2))

    # Within-person cross-trait correlation
    rho_w = rA * np.sqrt(A1 * A2) + rC * np.sqrt(C1 * C2) + rE * np.sqrt(E1 * E2)

    if assort1 != 0 and assort2 != 0:
        # Both traits: diagonal = targets, off-diagonal from rho_w mediation
        c = rho_w * np.sqrt(abs(assort1 * assort2)) * np.sign(assort1 * assort2)
        return np.array([[assort1, c], [c, assort2]])
    if assort1 != 0:
        # Single-trait on trait 1: propagate via rho_w
        a = assort1
        return np.array(
            [
                [a, a * rho_w],
                [a * rho_w, a * rho_w**2],
            ]
        )
    # Single-trait on trait 2: propagate via rho_w
    a = assort2
    return np.array(
        [
            [a * rho_w**2, a * rho_w],
            [a * rho_w, a],
        ]
    )
