"""Shared utility functions for sim_ace."""

from __future__ import annotations

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
    """Compute Pearson correlation, returning nan if either array has zero variance."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(_pearsonr_core(x, y))


def safe_linregress(x: np.ndarray, y: np.ndarray) -> Any:
    """Run linear regression, returning None if x has zero variance."""
    if np.std(x) < 1e-10:
        return None
    return stats.linregress(x, y)


def fast_linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fast linear regression returning (slope, intercept, r, stderr, pvalue)."""
    slope, intercept, r, stderr, t_stat = _linregress_core(x, y)
    pvalue = float(2.0 * _t_sf(abs(t_stat), len(x) - 2))
    return float(slope), float(intercept), float(r), float(stderr), pvalue


def fast_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fast Pearson r with p-value. Returns (r, pvalue)."""
    r = float(_pearsonr_core(x, y))
    n = len(x)
    denom = 1.0 - r * r
    if denom < 1e-30 or n <= 2:
        return r, 0.0
    t_stat = r * np.sqrt((n - 2) / denom)
    pvalue = float(2.0 * _t_sf(abs(t_stat), n - 2))
    return r, pvalue


def to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for YAML serialization."""
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
    """Build a result dict with passed/details and optional extra keys."""
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
    """Save DataFrame as parquet with optimized dtypes and zstd compression."""
    optimize_dtypes(df)
    df.to_parquet(path, index=False, compression="zstd", **kwargs)


def yaml_loader() -> type:
    """Return the fastest available YAML SafeLoader."""
    import yaml

    return getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def get_nested(d: Any, *keys: str, default: Any = None) -> Any:
    """Traverse nested dicts by key path, returning default if any key is missing."""
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
