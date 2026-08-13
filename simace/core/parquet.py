"""Parquet reader/writer with pedigree-aware dtype narrowing and the null contract.

Missing values are **parquet null** on disk and null in polars frames (ADR
0015). NumPy compute boundaries may transiently materialize nulls as NaN, so
the writer self-enforces the contract: float NaN is normalized to null before
every write. This restores the historical pandas-era on-disk contract
(``pd.to_parquet`` always wrote NaN as null); the ``nan_to_null=False`` escape
hatch ADR 0014 added briefly inverted it and is gone.

Writes narrow dtypes by column name (int32 ids, int8 sex, float32 components)
for compact storage; integer narrowing is range-checked, so overflow raises
instead of wrapping. Reads return an eager ``pl.DataFrame`` via
:func:`load_parquet`.

Transitional (Wave 1 of the polars migration): :func:`save_parquet` accepts
pandas *or* polars frames while pipeline stages migrate independently.
Unmigrated stages keep reading with ``pd.read_parquet``; migrated stages must
use :func:`load_parquet`. Pandas acceptance is removed in the coordinated
Wave 2 boundary break (ADR 0015).
"""

from __future__ import annotations

__all__ = ["load_parquet", "save_parquet"]

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

_INT32_COLS = ("id", "mother", "father", "twin", "household_id", "generation")
_INT8_COLS = ("sex",)
_FLOAT32_COLS = (
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
)


def _optimized_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Return ``df`` with columns downcast by name for compact parquet storage.

    Dtype strategy (matching pedigree generation-time dtypes):
    - int32 for ID columns and generation (supports up to 2.1B individuals)
    - int8 for sex (0/1)
    - float32 for ACE components and event times (~7 significant digits)
    - float64 for liabilities (full precision for phenotype models)

    Integer narrowing uses strict casts: a value outside the target range
    raises instead of wrapping.

    Raises:
        ValueError: If a narrowed integer column holds values outside the
            target dtype's range.
    """
    casts: list[tuple[str, pl.DataType]] = []
    for cols, target in ((_INT32_COLS, pl.Int32()), (_INT8_COLS, pl.Int8()), (_FLOAT32_COLS, pl.Float32())):
        casts.extend((c, target) for c in cols if c in df.columns and df.schema[c] != target)
    for col, target in casts:
        if target in (pl.Int32(), pl.Int8()):
            try:
                df = df.with_columns(pl.col(col).cast(target))
            except pl.exceptions.InvalidOperationError as e:
                raise ValueError(f"save_parquet: column {col!r} does not fit {target}: {e}") from e
        else:
            df = df.with_columns(pl.col(col).cast(target))
    return df


def save_parquet(df: pd.DataFrame | pl.DataFrame, path: Any, **kwargs: Any) -> None:
    """Save a DataFrame as parquet with optimized dtypes and zstd compression.

    Narrows dtypes via the name-based mapping (int32 ids, int8 sex, float32
    components; range-checked) and normalizes float NaN to null before writing,
    so on-disk missing values are always parquet null (ADR 0015 null contract).
    The caller's frame is never mutated. A pandas index is dropped — identity
    and order live in explicit columns.

    Args:
        df: Frame to save. ``pl.DataFrame``, or (transitionally, until the
            Wave 2 boundary break) ``pd.DataFrame``.
        path: Output file path, or any file-like object polars accepts.
        **kwargs: Extra keyword arguments passed to
            ``polars.DataFrame.write_parquet``.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    df = _optimized_dtypes(df)
    df = df.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))
    df.write_parquet(path, compression="zstd", **kwargs)


def load_parquet(path: Any, columns: Sequence[str] | None = None, **kwargs: Any) -> pl.DataFrame:
    """Read a parquet file into an eager ``pl.DataFrame``.

    The single read entry point for migrated stage code — stages do not call
    ``pl.read_parquet`` directly (and never expose ``LazyFrame``; lazy scanning
    is a separate benchmark-driven follow-up, per ADR 0015).

    Args:
        path: Parquet file path, or any source polars accepts.
        columns: Optional column subset to read.
        **kwargs: Extra keyword arguments passed to ``polars.read_parquet``.

    Returns:
        The file contents; missing values are null, never NaN.
    """
    return pl.read_parquet(path, columns=columns, **kwargs)
