"""Schema contracts for pedigree and hydrated in-memory trait frames.

The on-disk trait-family parquet contract is outcomes-only and lives in
:mod:`simace.core.trait_schema`.  The cumulative ``PHENOTYPE`` and ``CENSORED``
schemas below remain useful for hydrated in-memory frames and legacy unit-test
fixtures that include pedigree columns plus trait outcomes.

Dtypes are checked at the coarse ``numpy.dtype.kind`` level (``i`` integer,
``f`` float, ``b`` bool). This tolerates the int32/int8/float32 narrowing
applied by the parquet writer at save time without losing the contract.
"""

from __future__ import annotations

__all__ = ["CENSORED", "PEDIGREE", "PHENOTYPE", "assert_schema"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

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


def assert_schema(df: pd.DataFrame, schema: Mapping[str, str], *, where: str) -> None:
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
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns {missing}")

    bad: list[str] = []
    for col, kinds in schema.items():
        actual = df[col].dtype.kind
        if actual not in kinds:
            bad.append(f"{col}={df[col].dtype.name} (expected kind in {kinds!r})")
    if bad:
        raise ValueError(f"{where}: dtype mismatch — {'; '.join(bad)}")
