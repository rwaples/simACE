"""Parquet writer with pedigree-aware dtype narrowing."""

from __future__ import annotations

__all__ = ["save_parquet"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def _optimized_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with columns downcast for compact parquet storage.

    Does **not** mutate the input — narrowing is applied via ``df.astype`` to a
    new DataFrame.

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
    mapping: dict[str, str] = {}
    for c in int32_cols:
        if c in df.columns:
            mapping[c] = "int32"
    for c in int8_cols:
        if c in df.columns:
            mapping[c] = "int8"
    for c in float32_cols:
        if c in df.columns:
            mapping[c] = "float32"

    if not mapping:
        return df
    return df.astype(mapping)


def save_parquet(df: pd.DataFrame, path: Any, **kwargs: Any) -> None:
    """Save DataFrame as parquet with optimized dtypes and zstd compression.

    Narrows dtypes via :func:`_optimized_dtypes` (to minimize file size) before
    writing. The caller's ``df`` is **not** mutated — narrowing is applied to an
    internal copy.

    Args:
        df: DataFrame to save.
        path: Output file path.
        **kwargs: Extra keyword arguments passed to ``DataFrame.to_parquet``.
    """
    _optimized_dtypes(df).to_parquet(path, index=False, compression="zstd", **kwargs)
