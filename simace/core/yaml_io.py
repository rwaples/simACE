"""YAML serialization helpers: numpy → Python conversion and fast loader factory."""

from __future__ import annotations

__all__ = ["to_native", "yaml_loader"]

from typing import Any

import numpy as np


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


def yaml_loader() -> type:
    """Return the fastest available YAML SafeLoader.

    Returns:
        ``yaml.CSafeLoader`` if the C extension is available, else ``yaml.SafeLoader``.
    """
    import yaml

    return getattr(yaml, "CSafeLoader", yaml.SafeLoader)
