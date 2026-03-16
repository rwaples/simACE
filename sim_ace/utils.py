"""Shared utility functions for sim_ace."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# Canonical 7 relationship pair types used for tetrachoric / liability correlations
PAIR_TYPES: list[str] = [
    "MZ twin", "Full sib", "Mother-offspring", "Father-offspring",
    "Maternal half sib", "Paternal half sib", "1st cousin",
]

PAIR_COLORS: dict[str, str] = {
    "MZ twin": "C0", "Full sib": "C1", "Mother-offspring": "C3",
    "Father-offspring": "C5", "Maternal half sib": "C2",
    "Paternal half sib": "C6", "1st cousin": "C4",
}


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


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast DataFrame columns for compact parquet storage.

    Casts simulation float columns to float32 (~7 significant digits,
    ample for stochastic draws) and small integer columns to int8.
    ID columns (id, mother, father, twin, household_id) stay int64.
    """
    int8_cols = ["sex", "generation"]
    float32_cols = [
        "A1", "C1", "E1", "liability1",
        "A2", "C2", "E2", "liability2",
        "t1", "t2", "death_age", "t_observed1", "t_observed2",
    ]
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


def save_placeholder_plot(output_path: Any, message: str, figsize: tuple[float, float] = (6, 4), dpi: int = 150) -> None:
    """Save a single-panel figure with centered message text."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def finalize_plot(output_path: Any, dpi: int = 150, tight_rect: list[float] | None = None) -> None:
    """tight_layout + savefig(bbox_inches='tight') + close current figure."""
    import matplotlib.pyplot as plt
    if tight_rect is not None:
        plt.tight_layout(rect=tight_rect)
    else:
        plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
