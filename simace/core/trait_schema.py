"""Outcomes-only trait file schemas and hydration helpers.

Trait-family parquet files store only per-individual trait outcomes. Pedigree,
demography, ACE components, and liabilities live in the corresponding pedigree
parquet. Consumers that need a self-contained in-memory frame should call
:func:`hydrate_trait` with the appropriate pedigree frame.

Transitional (Wave 1 of the polars migration, ADR 0015): the frame-returning
helpers here are same-type dual-frame APIs — polars in, polars out; pandas in,
pandas out; mixed-library inputs fail with a clear ``TypeError`` rather than
converting implicitly. Pandas acceptance is removed in the coordinated Wave 2
boundary break.
"""

from __future__ import annotations

__all__ = [
    "CENSORED_TRAIT",
    "RAW_TRAIT",
    "TRAIT_CENSORED_COLUMNS",
    "TRAIT_OUTCOME_COLUMNS_BY_KIND",
    "TRAIT_RAW_COLUMNS",
    "TraitKind",
    "hydrate_trait",
    "strip_trait_to_outcomes",
]

from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

TraitKind = Literal["raw", "censored"]

TRAIT_RAW_COLUMNS: tuple[str, ...] = ("id", "t1", "t2")
TRAIT_CENSORED_COLUMNS: tuple[str, ...] = (
    "id",
    "t1",
    "t2",
    "death_age",
    "age_censored1",
    "t_observed1",
    "death_censored1",
    "affected1",
    "age_censored2",
    "t_observed2",
    "death_censored2",
    "affected2",
)
TRAIT_OUTCOME_COLUMNS_BY_KIND: dict[TraitKind, tuple[str, ...]] = {
    "raw": TRAIT_RAW_COLUMNS,
    "censored": TRAIT_CENSORED_COLUMNS,
}

RAW_TRAIT: dict[str, str] = {"id": "iu", "t1": "f", "t2": "f"}
CENSORED_TRAIT: dict[str, str] = {
    **RAW_TRAIT,
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

_ID = "id"


def _columns_for_kind(kind: TraitKind) -> tuple[str, ...]:
    try:
        return TRAIT_OUTCOME_COLUMNS_BY_KIND[kind]
    except KeyError as exc:
        known = ", ".join(sorted(TRAIT_OUTCOME_COLUMNS_BY_KIND))
        raise ValueError(f"unknown trait kind {kind!r}; expected one of: {known}") from exc


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, where: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns {missing}")


def _require_unique_id(df: pd.DataFrame | pl.DataFrame, *, where: str) -> None:
    if isinstance(df, pl.DataFrame):
        duplicates = df[_ID].is_duplicated()
        if duplicates.any():
            example = df.filter(duplicates)[_ID].head(5).to_list()
            raise ValueError(f"{where}: duplicate id values are not allowed; examples: {example}")
        return
    duplicates = df[_ID].duplicated()
    if duplicates.any():
        example = df.loc[duplicates, _ID].head(5).tolist()
        raise ValueError(f"{where}: duplicate id values are not allowed; examples: {example}")


def _require_same_library(trait: pd.DataFrame | pl.DataFrame, pedigree: pd.DataFrame | pl.DataFrame) -> bool:
    """Return whether the pair is polars; reject mixed-library input clearly."""
    trait_pl, ped_pl = isinstance(trait, pl.DataFrame), isinstance(pedigree, pl.DataFrame)
    if trait_pl != ped_pl:

        def _lib(is_pl: bool) -> str:
            return "polars" if is_pl else "pandas"

        raise TypeError(
            f"mixed DataFrame libraries: trait={_lib(trait_pl)}, pedigree={_lib(ped_pl)} — "
            "pass both frames from the same library"
        )
    return trait_pl


def _normalize_pedigree_columns(columns: Sequence[str] | None, pedigree_columns: Sequence[str]) -> list[str]:
    if columns is None:
        requested = list(pedigree_columns)
    else:
        requested = [_ID, *[col for col in columns if col != _ID]]

    seen: set[str] = set()
    duplicates: list[str] = []
    for col in requested:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)
    if duplicates:
        raise ValueError(f"pedigree columns contain duplicates: {duplicates}")
    return requested


def strip_trait_to_outcomes(df: pd.DataFrame | pl.DataFrame, kind: TraitKind) -> pd.DataFrame | pl.DataFrame:
    """Return ``df`` restricted to the outcomes-only schema for ``kind``.

    Same-type dual-frame API (transitional, ADR 0015): returns the same frame
    library it was given.

    Args:
        df: Trait-like DataFrame, potentially carrying hydrated pedigree columns.
        kind: Trait file kind: ``"raw"`` or ``"censored"``.

    Returns:
        A copy with only the canonical outcomes-only columns for ``kind``, in
        schema order.

    Raises:
        ValueError: If any required outcomes-only column is missing.
    """
    columns = _columns_for_kind(kind)
    _require_columns(df, columns, where=f"{kind} trait")
    if isinstance(df, pl.DataFrame):
        return df.select(columns)
    return df.loc[:, list(columns)].copy()


def hydrate_trait(
    trait: pd.DataFrame | pl.DataFrame,
    pedigree: pd.DataFrame | pl.DataFrame,
    *,
    kind: TraitKind,
    columns: Sequence[str] | None = None,
    validate: bool = True,
) -> pd.DataFrame | pl.DataFrame:
    """Join pedigree columns onto an outcomes-only trait frame by ``id``.

    The returned frame preserves trait row order and places pedigree columns
    first, followed by trait outcome/audit columns. The join is strict: trait
    and pedigree IDs must be unique, every trait ID must exist in the pedigree,
    and trait columns must not collide with requested pedigree columns other
    than the shared ``id`` join key.

    Same-type dual-frame API (transitional, ADR 0015): polars in, polars out;
    pandas in, pandas out; mixed-library input raises ``TypeError``.

    Args:
        trait: Outcomes-only trait DataFrame.
        pedigree: Pedigree DataFrame containing ``id`` and requested columns.
            Must be the same frame library as ``trait``.
        kind: Trait file kind used to validate required outcome columns.
        columns: Pedigree columns to include. ``id`` is always included first
            even when omitted here. ``None`` includes all pedigree columns.
        validate: When true, require the minimum outcomes-only columns for
            ``kind``. Join-key and collision checks always run.

    Returns:
        Hydrated DataFrame with pedigree columns first, then all trait columns
        except the duplicate ``id``.

    Raises:
        TypeError: If ``trait`` and ``pedigree`` come from different frame
            libraries.
        ValueError: If required columns are missing, IDs are duplicated or
            missing, or requested pedigree columns collide with trait columns.
    """
    is_polars = _require_same_library(trait, pedigree)

    if validate:
        _require_columns(trait, _columns_for_kind(kind), where=f"{kind} trait")
    _require_columns(trait, [_ID], where="trait")
    _require_columns(pedigree, [_ID], where="pedigree")

    pedigree_cols = _normalize_pedigree_columns(columns, list(pedigree.columns))
    _require_columns(pedigree, pedigree_cols, where="pedigree")

    collisions = sorted((set(trait.columns) & set(pedigree_cols)) - {_ID})
    if collisions:
        raise ValueError(
            "trait columns collide with requested pedigree columns; "
            f"hydrate outcomes-only trait files or drop duplicate columns first: {collisions}"
        )

    _require_unique_id(trait, where="trait")
    _require_unique_id(pedigree, where="pedigree")

    if is_polars:
        in_pedigree = trait[_ID].is_in(pedigree[_ID].implode())
        if not bool(in_pedigree.all()):
            missing = trait.filter(~in_pedigree)[_ID].head(5).to_list()
            raise ValueError(f"trait ids missing from pedigree; examples: {missing}")

        ped_part = (
            trait.select(_ID)
            .join(pedigree.select(pedigree_cols), on=_ID, how="left", maintain_order="left")
            .select(pedigree_cols)
        )
        return ped_part.hstack(trait.drop(_ID))

    trait_ids = trait[_ID]
    in_pedigree = trait_ids.isin(pedigree[_ID])
    if not bool(in_pedigree.all()):
        missing = trait_ids.loc[~in_pedigree].head(5).tolist()
        raise ValueError(f"trait ids missing from pedigree; examples: {missing}")

    ped_part = (
        pedigree.loc[:, pedigree_cols].set_index(_ID, drop=False).loc[trait_ids.to_numpy()].reset_index(drop=True)
    )
    trait_part = trait.drop(columns=[_ID]).reset_index(drop=True)
    return ped_part.join(trait_part)
