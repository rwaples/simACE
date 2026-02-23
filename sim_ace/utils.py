"""Shared utility functions for sim_ace."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning nan if either array has zero variance."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return np.corrcoef(x, y)[0, 1]


def safe_linregress(x: np.ndarray, y: np.ndarray) -> Any:
    """Run linear regression, returning None if x has zero variance."""
    if np.std(x) < 1e-10:
        return None
    return stats.linregress(x, y)


def to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for YAML serialization."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def validation_result(passed: bool, details: str, **extra: Any) -> dict[str, Any]:
    """Build a result dict with passed/details and optional extra keys."""
    d: dict[str, Any] = {"passed": passed, "details": details}
    d.update(extra)
    return d


def get_nested(d: Any, *keys: str, default: Any = None) -> Any:
    """Traverse nested dicts by key path, returning default if any key is missing."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d
