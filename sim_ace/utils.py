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


def _make_heatmap_cmap():
    """Truncated GnBu (green-to-blue) starting at 40% — dark enough for white text."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    gnbu = plt.cm.GnBu
    return mcolors.LinearSegmentedColormap.from_list(
        "GnBu_dark", gnbu(np.linspace(0.4, 1.0, 256))
    )


HEATMAP_CMAP = _make_heatmap_cmap()


def annotate_heatmap(ax, proportions, counts, fmt_prop=".2f", prop_size=18, count_size=11) -> None:
    """Add two-line annotations to a heatmap: large bold proportion, smaller count.

    Args:
        ax: Matplotlib axes containing the heatmap.
        proportions: 2-D array-like of proportion values.
        counts: 2-D array-like of count values (int or float).
        fmt_prop: Format spec for proportion values.
        prop_size: Font size for the proportion line.
        count_size: Font size for the count line.
    """
    import numpy as np
    proportions = np.asarray(proportions)
    counts = np.asarray(counts)
    for i in range(proportions.shape[0]):
        for j in range(proportions.shape[1]):
            p = proportions[i, j]
            c = counts[i, j]
            c_str = f"n={int(c)}" if float(c) == int(c) else f"n={c:.0f}"
            ax.text(j + 0.5, i + 0.38, f"{p:{fmt_prop}}",
                    ha="center", va="center", fontsize=prop_size,
                    fontweight="bold", color="white")
            ax.text(j + 0.5, i + 0.62, c_str,
                    ha="center", va="center", fontsize=count_size,
                    color=(1, 1, 1, 0.7))


def finalize_plot(output_path: Any, dpi: int = 150, tight_rect: list[float] | None = None, subsample_note: str = "") -> None:
    """tight_layout + savefig(bbox_inches='tight') + close current figure."""
    import warnings
    import matplotlib.pyplot as plt
    if subsample_note:
        fig = plt.gcf()
        fig.text(0.99, 0.01, subsample_note, fontsize=8, color="0.5",
                 ha="right", va="bottom", fontstyle="italic",
                 transform=fig.transFigure)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
        if tight_rect is not None:
            plt.tight_layout(rect=tight_rect)
        else:
            plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
