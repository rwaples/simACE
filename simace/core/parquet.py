"""Parquet writer with pedigree-aware dtype narrowing.

Writes go out through polars, which is substantially faster than the pandas /
pyarrow writer at pedigree scale and produces smaller files (measured at 6M
rows: 3.6s → 0.55s, 273 MB → 248 MB). Frames are still handed in and narrowed
as pandas — the conversion is zero-copy for the numeric dtypes this pipeline
writes, so it costs nothing.

Reads deliberately stay on ``pandas.read_parquet``: ``pl.read_parquet`` is
faster on its own, but the ``to_pandas()`` copy needed to keep the existing
DataFrame-returning API more than cancels it out (410ms vs 297ms at 6M rows).
"""

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
    internal copy. The pandas index is dropped (polars has no index), matching
    the ``to_parquet(index=False)`` behavior this replaced.

    ``nan_to_null=False`` is required on the conversion: polars distinguishes
    NaN from null while pandas conflates them, and the default would rewrite
    float NaNs as parquet nulls. A pandas round-trip still *looks* correct
    either way, but the on-disk null mask differs — which matters for the
    non-pandas readers of these files (LDAK, EPIMIGHT's R driver).

    Args:
        df: DataFrame to save.
        path: Output file path, or any file-like object polars accepts.
        **kwargs: Extra keyword arguments passed to
            ``polars.DataFrame.write_parquet`` (previously ``to_parquet``; no
            in-tree caller passes any).
    """
    import polars as pl

    pl.from_pandas(_optimized_dtypes(df), nan_to_null=False).write_parquet(path, compression="zstd", **kwargs)
