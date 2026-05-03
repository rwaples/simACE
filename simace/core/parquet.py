"""Parquet writer with pedigree-aware dtype narrowing."""

from __future__ import annotations

__all__ = ["save_parquet"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def _optimize_dtypes(df: pd.DataFrame) -> None:
    """Downcast DataFrame columns in-place for compact parquet storage.

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


def save_parquet(df: pd.DataFrame, path: Any, **kwargs: Any) -> None:
    """Save DataFrame as parquet with optimized dtypes and zstd compression.

    Calls ``_optimize_dtypes`` before writing to minimize file size.

    Args:
        df: DataFrame to save.
        path: Output file path.
        **kwargs: Extra keyword arguments passed to ``DataFrame.to_parquet``.
    """
    _optimize_dtypes(df)
    df.to_parquet(path, index=False, compression="zstd", **kwargs)
