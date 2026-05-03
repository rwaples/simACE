"""YAML serialization helpers: numpy → Python conversion, fast loader, and file I/O wrappers."""

__all__ = ["dump_yaml", "load_yaml", "to_native", "yaml_loader"]

from pathlib import Path
from typing import Any

import numpy as np
import yaml

_LOADER: type = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


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
    """Return the fastest available YAML SafeLoader (``CSafeLoader`` if present)."""
    return _LOADER


def load_yaml(path: str | Path) -> Any:
    """Load a YAML file using the fastest available SafeLoader."""
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_LOADER)


def dump_yaml(obj: Any, path: str | Path, *, sort_keys: bool = False) -> None:
    """Dump ``obj`` to ``path`` as YAML, normalizing numpy types via ``to_native``."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(to_native(obj), fh, default_flow_style=False, sort_keys=sort_keys)
