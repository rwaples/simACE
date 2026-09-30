"""The cohort file, the views rebuilt from it, and the layout marker (ADR 0021).

A finished rep stores two Parquet files: ``pedigree.parquet``, the full
recorded pedigree, and ``cohort.parquet``, one row per member of the analysis
pedigree with the trait outcome columns (``COHORT_COLUMNS``). Readers rebuild
the analysis pedigree and the analysis sample with :func:`selected_views`.

Who is in the cohort, by route, and how a reader tells:

- **A. drawn.** Survived dropout, in the trailing ``G_pheno`` generations,
  kept by the case-weighted draw. Outcome columns non-null (``t1``, ``t2``
  may be null, ADR 0019). Told by ``affected1`` not null.
- **B. ancestor, never phenotyped.** An ancestor of a drawn person through
  intact parent links, older than the phenotyped window. Outcome columns all
  null. Told by ``affected1`` null and ``generation`` outside the window.
- **C. ancestor, phenotyped but not drawn.** An ancestor of a drawn person
  inside the phenotyped window. Outcome columns all null although outcomes
  were computed. Told by ``affected1`` null and ``generation`` inside the
  window.

A person who is both drawn and an ancestor is route A. Route C needs
``generation`` (from ``pedigree.parquet``) and ``G_pheno`` to tell apart from
B, and arises only when the draw is not pass-through and ``G_pheno >= 2``.

Out of the cohort: dropped individuals at any generation (the ancestor
closure cannot cross them); phenotyped people neither drawn nor ancestors of
anyone drawn; unphenotyped people who are not ancestors of anyone drawn.

In the analysis pedigree a ``mother``, ``father`` or ``twin`` of ``-1`` means
the referent is unknown in the recorded pedigree, was dropped (the only way a
parent link severs; a row may keep one parent and lose the other), or is
outside the closure (twins only). ``pedigree.parquet`` keeps the recorded
links, so comparing the two classifies each ``-1``.

Both files carry the Parquet key-value metadata ``simace_layout=2``;
:func:`read_pedigree` and :func:`read_cohort` refuse files without it, which
catches an older rep's selected ``pedigree.parquet`` read as recorded.
"""

from __future__ import annotations

__all__ = [
    "COHORT_COLUMNS",
    "LAYOUT",
    "SAMPLE_REQUIRED",
    "SelectedViews",
    "build_cohort",
    "check_cohort",
    "read_cohort",
    "read_pedigree",
    "selected_views",
    "sever_dangling_links",
    "write_cohort",
    "write_pedigree",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from simace.core.parquet import load_parquet, save_parquet
from simace.core.pedigree_arrays import PedigreeArrays
from simace.core.trait_schema import TRAIT_CENSORED_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The results layout this module reads and writes; ``simace.cli.stages.REP_LAYOUT`` mirrors it.
LAYOUT = 2
_LAYOUT_KEY = "simace_layout"

COHORT_COLUMNS: tuple[str, ...] = TRAIT_CENSORED_COLUMNS
_OUTCOME_COLUMNS: tuple[str, ...] = tuple(c for c in COHORT_COLUMNS if c != "id")
#: Columns non-null on every sample row; the raw onsets ``t1``, ``t2`` stay nullable (ADR 0019).
SAMPLE_REQUIRED: tuple[str, ...] = tuple(c for c in _OUTCOME_COLUMNS if c not in ("t1", "t2"))

_LINK_COLUMNS = ("mother", "father", "twin")


@dataclass(frozen=True)
class SelectedViews:
    """The analysis pedigree and analysis sample rebuilt from a pedigree and its cohort.

    Attributes:
        pedigree: Pedigree rows whose id is in the cohort, in pedigree order,
            with parent and twin links outside that set rewritten to ``-1``.
        trait: Cohort rows with ``affected1`` not null, ``COHORT_COLUMNS``
            only, in cohort order.
    """

    pedigree: pl.DataFrame
    trait: pl.DataFrame


def sever_dangling_links(df: pl.DataFrame, valid_ids: np.ndarray) -> pl.DataFrame:
    """Rewrite ``mother``/``father``/``twin`` references pointing outside ``valid_ids`` to -1.

    Columns absent from ``df`` are skipped, so any column subset holding ``id``
    works. ``valid_ids`` must be unique and non-negative.
    """
    valid = PedigreeArrays({"id": np.asarray(valid_ids)})
    result = df
    for col in _LINK_COLUMNS:
        if col not in result.columns:
            continue
        vals = result[col].to_numpy()
        dangling = (vals >= 0) & ~valid.contains(vals)
        if dangling.any():
            fixed = vals.copy()
            fixed[dangling] = -1
            result = result.with_columns(pl.Series(col, fixed))
    return result


def check_cohort(cohort: pl.DataFrame, pedigree_ids: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``cohort`` satisfies the cohort invariants against ``pedigree_ids``.

    1. ``id`` is unique and every id is in ``pedigree_ids``.
    2. On sample rows (``affected1`` not null) every ``SAMPLE_REQUIRED``
       column is non-null.
    3. On other rows every outcome column is null.
    4. Rows follow ``pedigree_ids`` order.

    Args:
        cohort: Frame with exactly ``COHORT_COLUMNS``.
        pedigree_ids: Ids of the pedigree the cohort was drawn from, in its row order; unique.
    """
    _checked_rows(cohort, PedigreeArrays({"id": np.asarray(pedigree_ids)}))


def _checked_rows(cohort: pl.DataFrame, pedigree: PedigreeArrays) -> np.ndarray:
    """Check the :func:`check_cohort` invariants; return the cohort's row positions in ``pedigree``."""
    if tuple(cohort.columns) != COHORT_COLUMNS:
        raise ValueError(f"cohort columns are {list(cohort.columns)}; expected {list(COHORT_COLUMNS)}")

    ids = cohort["id"].to_numpy()
    present = pedigree.contains(ids)
    if not present.all():
        raise ValueError(f"cohort ids missing from the pedigree; examples: {ids[~present][:5].tolist()}")
    rows = pedigree.positions(ids)

    sample = cohort["affected1"].is_not_null()
    null_on_sample = cohort.filter(sample).select(pl.col(SAMPLE_REQUIRED).null_count()).row(0, named=True)
    bad = [col for col, n in null_on_sample.items() if n]
    if bad:
        raise ValueError(f"cohort sample rows (affected1 not null) have nulls in {bad}")
    set_off_sample = cohort.filter(~sample).select(pl.col(_OUTCOME_COLUMNS).is_not_null().sum()).row(0, named=True)
    bad = [col for col, n in set_off_sample.items() if n]
    if bad:
        raise ValueError(f"cohort rows outside the sample (affected1 null) have values in {bad}")

    steps = np.diff(rows)
    if not bool(np.all(steps > 0)):
        duplicated = cohort["id"].is_duplicated()
        if duplicated.any():
            examples = cohort.filter(duplicated)["id"].unique().head(5).to_list()
            raise ValueError(f"cohort ids are not unique; examples: {examples}")
        first = int(np.flatnonzero(steps <= 0)[0]) + 1
        raise ValueError(
            f"cohort rows are not in pedigree order: id {ids[first]} at row {first} "
            f"comes before id {ids[first - 1]} in the pedigree"
        )
    return rows


def build_cohort(analysis_pedigree: pl.DataFrame, sample_trait: pl.DataFrame) -> pl.DataFrame:
    """Return the cohort: the analysis pedigree's ids in its order, with the sample's outcome columns.

    Args:
        analysis_pedigree: Ascertained pedigree (``run_ascertainment``'s first output).
        sample_trait: Ascertained censored trait rows (its second output), in
            pedigree order.

    Raises:
        ValueError: A sample id is missing from the analysis pedigree, the
            sample is not in pedigree order, or the result breaks an invariant
            of :func:`check_cohort`.
    """
    cohort = analysis_pedigree.select("id").join(
        sample_trait.select(COHORT_COLUMNS), on="id", how="left", maintain_order="left"
    )
    sample_ids = cohort.filter(pl.col("affected1").is_not_null())["id"]
    if not sample_ids.equals(sample_trait["id"]):
        raise ValueError(
            "sample trait ids are not the analysis pedigree's ids in its order: "
            f"{len(sample_trait)} sample rows, {len(sample_ids)} matched with affected1 set"
        )
    check_cohort(cohort, analysis_pedigree["id"].to_numpy())
    return cohort


def selected_views(pedigree: pl.DataFrame, cohort: pl.DataFrame) -> SelectedViews:
    """Rebuild the analysis pedigree and analysis sample from a pedigree and its cohort.

    Args:
        pedigree: The recorded pedigree, or any column subset of it that includes ``id``.
        cohort: The cohort drawn from that pedigree.

    Raises:
        ValueError: ``cohort`` breaks an invariant of :func:`check_cohort` against ``pedigree``.
    """
    rows = _checked_rows(cohort, PedigreeArrays({"id": pedigree["id"].to_numpy()}))
    keep = np.zeros(len(pedigree), dtype=bool)
    keep[rows] = True
    analysis_pedigree = sever_dangling_links(pedigree.filter(pl.Series(keep)), cohort["id"].to_numpy())
    return SelectedViews(analysis_pedigree, cohort.filter(pl.col("affected1").is_not_null()))


def _write(df: pl.DataFrame, path: Any) -> None:
    save_parquet(df, path, metadata={_LAYOUT_KEY: str(LAYOUT)})


def _require_layout(path: Any) -> None:
    found = pl.read_parquet_metadata(path).get(_LAYOUT_KEY)
    if found != str(LAYOUT):
        raise ValueError(
            f"{path} lacks the Parquet metadata {_LAYOUT_KEY}={LAYOUT} (found {found!r}): "
            f"it predates results layout {LAYOUT} (ADR 0021); recompute it"
        )


def write_pedigree(df: pl.DataFrame, path: Any) -> None:
    """Write the recorded pedigree through :func:`save_parquet`, marked with the layout."""
    _write(df, path)


def write_cohort(df: pl.DataFrame, path: Any) -> None:
    """Write a cohort built by :func:`build_cohort` through :func:`save_parquet`, marked with the layout."""
    _write(df, path)


def read_pedigree(path: Any, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read a recorded ``pedigree.parquet``, refusing a file without the layout marker."""
    _require_layout(path)
    return load_parquet(path, columns=columns)


def read_cohort(path: Any) -> pl.DataFrame:
    """Read a ``cohort.parquet``, refusing a file without the layout marker."""
    _require_layout(path)
    return load_parquet(path)
