"""Schema contracts for pedigree and hydrated in-memory trait frames.

The on-disk trait-family parquet contract is outcomes-only and lives in
:mod:`simace.core.trait_schema`.  The cumulative ``PHENOTYPE`` and ``CENSORED``
schemas below remain useful for hydrated in-memory frames and legacy unit-test
fixtures that include pedigree columns plus trait outcomes.

Dtypes are checked at the coarse ``numpy.dtype.kind`` level (``i`` integer,
``f`` float, ``b`` bool). This tolerates the int32/int8/float32 narrowing
applied by the parquet writer at save time without losing the contract.

Polars-only since the Wave 2 boundary break (ADR 0015): pandas frames are
rejected with an actionable ``TypeError``.
"""

from __future__ import annotations

__all__ = ["CENSORED", "PEDIGREE", "PHENOTYPE", "assert_schema"]

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

PEDIGREE: Mapping[str, str] = {
    "id": "iu",
    "generation": "iu",
    "sex": "iu",
    "mother": "iu",
    "father": "iu",
    "twin": "iu",
    "household_id": "iu",
    "A1": "f",
    "C1": "f",
    "E1": "f",
    "liability1": "f",
    "A2": "f",
    "C2": "f",
    "E2": "f",
    "liability2": "f",
}

PHENOTYPE: Mapping[str, str] = {
    **PEDIGREE,
    "t1": "f",
    "t2": "f",
}

CENSORED: Mapping[str, str] = {
    **PHENOTYPE,
    "death_age": "f",
    "age_censored1": "b",
    "t_observed1": "f",
    "death_censored1": "b",
    "affected1": "b",
    "age_censored2": "b",
    "t_observed2": "f",
    "death_censored2": "b",
    "affected2": "b",
}


def _polars_kind(dtype: pl.DataType) -> str:
    """Map a polars logical dtype onto the ``numpy.dtype.kind`` character set."""
    if dtype.is_signed_integer():
        return "i"
    if dtype.is_unsigned_integer():
        return "u"
    if dtype.is_float():
        return "f"
    if dtype == pl.Boolean():
        return "b"
    return "O"


def assert_schema(df: pl.DataFrame, schema: Mapping[str, str], *, where: str) -> None:
    """Verify ``df`` carries every column in ``schema`` with a compatible dtype kind.

    Args:
        df: DataFrame to check.
        schema: Mapping of required column name → allowed ``numpy.dtype.kind``
            characters (e.g. ``"f"`` for float, ``"iu"`` for any integer).
        where: Stage label included in the error message (e.g.
            ``"censor input"``) so failures pinpoint the offending boundary.

    Raises:
        ValueError: If columns are missing or have an unexpected dtype kind.
            Extra columns are allowed — stages are free to pass through
            additional fields.
    """
    if isinstance(df, pl.LazyFrame):
        raise TypeError(
            f"{where}: stage frames are eager pl.DataFrame, never LazyFrame — collect() before the boundary"
        )
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"{where}: stage frames must be polars DataFrames since the polars migration "
            f"(ADR 0015); got {type(df).__name__}. Convert with pl.from_pandas(...) at the call site."
        )

    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns {missing}")

    bad: list[str] = []
    for col, kinds in schema.items():
        dtype = df.schema[col]
        actual, name = _polars_kind(dtype), str(dtype)
        if actual not in kinds:
            bad.append(f"{col}={name} (expected kind in {kinds!r})")
    if bad:
        raise ValueError(f"{where}: dtype mismatch — {'; '.join(bad)}")
