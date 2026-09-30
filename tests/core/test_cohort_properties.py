"""Property tests for cohort reconstruction: ascertain, build the cohort, rebuild the views.

The composition law: for ``(ped_out, trait_out) = run_ascertainment(recorded,
trait, ...)`` and ``cohort = build_cohort(ped_out, trait_out)``,
``selected_views(recorded, cohort)`` returns exactly ``ped_out`` and
``trait_out``, before and after the marked Parquet round trip.  Independent
assertions check the cohort itself, so a builder and reader that share a
mistake cannot satisfy the law together.

Generated inputs come from ``ascertainment_inputs``; one constructed pedigree
guarantees every cohort route (``simace.core.cohort``) and link-severing case.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import polars.testing
import pytest
from hypothesis import event, given

from simace.ascertainment.runner import run_ascertainment
from simace.core.cohort import (
    COHORT_COLUMNS,
    build_cohort,
    read_cohort,
    read_pedigree,
    selected_views,
    write_cohort,
    write_pedigree,
)
from simace.core.parquet import normalize_for_parquet
from simace.core.schema import PEDIGREE
from tests.conftest import schema_pad
from tests.downstream_strategies import ascertainment_inputs

_OUTCOMES = [c for c in COHORT_COLUMNS if c != "id"]


def _assert_same(got: pl.DataFrame, want: pl.DataFrame) -> None:
    polars.testing.assert_frame_equal(got, want, check_row_order=True, check_dtypes=True, check_exact=True)


def _rows(frame: pl.DataFrame, ids) -> pl.DataFrame:
    return frame.filter(pl.col("id").is_in(pl.Series(list(ids), dtype=frame.schema["id"]).implode()))


def _assert_reconstructs(recorded, trait, ped_out, trait_out) -> pl.DataFrame:
    """Build the cohort, check it independently of the reader, check the law, and return it."""
    cohort = build_cohort(ped_out, trait_out)

    assert cohort.columns == list(COHORT_COLUMNS)
    assert cohort["id"].equals(ped_out["id"])
    in_sample = pl.col("id").is_in(trait_out["id"].implode())
    _assert_same(cohort.filter(in_sample), _rows(trait, trait_out["id"]).select(COHORT_COLUMNS))
    # Not ``is_not_null().any()``: polars 1.44.2 returns True for it on a frame emptied by filter(~is_in(...)).
    off_sample = cohort.filter(~in_sample)
    assert off_sample.select(_OUTCOMES).null_count().row(0) == (off_sample.height,) * len(_OUTCOMES)

    views = selected_views(recorded, cohort)
    _assert_same(views.pedigree, ped_out)
    _assert_same(views.trait, trait_out)
    return cohort


def _assert_round_trip(recorded, ped_out, trait_out, cohort) -> None:
    """Write the marked files to a fresh directory, read them back, and rebuild the views."""
    with tempfile.TemporaryDirectory() as tmp:
        write_pedigree(recorded, Path(tmp) / "pedigree.parquet")
        write_cohort(cohort, Path(tmp) / "cohort.parquet")
        stored_pedigree = read_pedigree(Path(tmp) / "pedigree.parquet")
        stored_cohort = read_cohort(Path(tmp) / "cohort.parquet")
    _assert_same(stored_pedigree, recorded)
    _assert_same(stored_cohort, cohort)
    views = selected_views(stored_pedigree, stored_cohort)
    _assert_same(views.pedigree, ped_out)
    _assert_same(views.trait, trait_out)


@given(inp=ascertainment_inputs())
def test_selected_views_rebuild_generated_ascertainments(inp):
    """``selected_views(recorded, build_cohort(*run_ascertainment(...)))`` returns the ascertainment outputs.

    Preconditions: a successful ascertainment whose trait frame has exactly
    ``COHORT_COLUMNS`` and the pedigree's id dtype.  Rejects treating null
    outcomes as controls, putting outcomes on ancestors, adding or losing
    sample members, and rebuilding severed links differently from
    ascertainment.
    """
    ped_out, trait_out = inp.run()
    cohort = _assert_reconstructs(inp.pedigree, inp.trait, ped_out, trait_out)
    off_sample = cohort.filter(pl.col("affected1").is_null())["id"]
    phenotyped = off_sample.is_in(inp.trait["id"].implode())
    event(f"sample empty: {trait_out.is_empty()}")
    event(f"ancestor never phenotyped: {bool((~phenotyped).any())}")
    event(f"ancestor phenotyped, not drawn: {bool(phenotyped.any())}")


@pytest.mark.slow
@given(inp=ascertainment_inputs())
def test_selected_views_survive_the_parquet_round_trip(inp):
    """After the marked write and read, the stored files equal their inputs and rebuild the same views.

    Inputs are normalized first because storage narrows ids, sex, components,
    and times.  Rejects a reader or writer that drops nulls, reorders rows,
    changes a value, or loses the layout marker.
    """
    inp = inp._replace(pedigree=normalize_for_parquet(inp.pedigree), trait=normalize_for_parquet(inp.trait))
    ped_out, trait_out = inp.run()
    cohort = _assert_reconstructs(inp.pedigree, inp.trait, ped_out, trait_out)
    _assert_round_trip(inp.pedigree, ped_out, trait_out, cohort)


# G_pheno = 2 over generations 0-2.  Gapped ids, so no id equals its row.
#   3 x 5 -> 12 (drawn, and mother of 20, 25, 26, 28)
#   8 x 9 -> 14 (phenotyped, not drawn; father of 20, 23, 25, 26, 28), 17 (dropped)
#   12 x 14 -> 20 (drawn, null t1), 25 = 26 MZ twins (25 drawn), 28 (not drawn)
#   17 x 14 -> 23 (drawn; its mother is dropped)
_RECORDED = [
    # id, generation, sex, mother, father, twin
    (3, 0, 0, -1, -1, -1),
    (5, 0, 1, -1, -1, -1),
    (8, 0, 0, -1, -1, -1),
    (9, 0, 1, -1, -1, -1),
    (12, 1, 0, 3, 5, -1),
    (14, 1, 1, 8, 9, -1),
    (17, 1, 0, 8, 9, -1),
    (20, 2, 0, 12, 14, -1),
    (23, 2, 1, 17, 14, -1),
    (25, 2, 0, 12, 14, 26),
    (26, 2, 0, 12, 14, 25),
    (28, 2, 1, 12, 14, -1),
]
_DROPPED = 17
_DRAWN = (12, 20, 23, 25)
_ANALYSIS_IDS = [3, 5, 8, 9, 12, 14, 20, 23, 25]


def _recorded() -> pl.DataFrame:
    frame = pl.DataFrame(_RECORDED, schema=["id", "generation", "sex", "mother", "father", "twin"], orient="row")
    frame = frame.with_columns(pl.all().cast(pl.Int32), household_id=pl.col("mother").rank("dense").cast(pl.Int32))
    tag = pl.col("id").cast(pl.Float64)
    frame = frame.with_columns(
        (tag + 0.25 * k).alias(col)
        for k, col in enumerate(["A1", "C1", "E1", "liability1", "A2", "C2", "E2", "liability2"])
    )
    return normalize_for_parquet(schema_pad(frame, PEDIGREE))


def _phenotyped_trait(recorded: pl.DataFrame) -> pl.DataFrame:
    """Outcomes for every generation 1-2 person, distinct per column; 20 has a null ``t1``."""
    tag = pl.col("id").cast(pl.Float64)
    affected1 = pl.col("id") % 2 == 0
    affected2 = pl.col("id") % 3 == 0
    frame = recorded.filter(pl.col("generation") >= 1).select(
        "id",
        t1=pl.when(pl.col("id") != 20).then(tag + 0.5),
        t2=tag + 100.5,
        death_age=tag + 200.5,
        age_censored1=~affected1,
        t_observed1=tag + 300.5,
        death_censored1=pl.lit(False),
        affected1=affected1,
        age_censored2=~affected2,
        t_observed2=tag + 400.5,
        death_censored2=pl.lit(False),
        affected2=affected2,
    )
    return normalize_for_parquet(frame)


def test_constructed_cohort_covers_every_route():
    """A fixed pedigree reaches each cohort route and each severing case, and the law holds through storage.

    Dropout is fixed by removing 17 before ascertainment and the draw by
    passing only the drawn rows (``N_sample=0`` passes the pool through).
    Covers a drawn ancestor (12), ancestors outside the window (3, 5, 8, 9),
    an in-window ancestor not drawn (14), a drawn row with null ``t1`` (20), one
    surviving parent (23), an excluded twin partner (26), and gapped ids.
    Rejects nulling a drawn ancestor's outcomes, giving an undrawn phenotyped
    ancestor its outcomes, and restoring a dropped parent or unsampled twin.
    """
    recorded = _recorded()
    phenotyped = _phenotyped_trait(recorded)
    ped_out, trait_out = run_ascertainment(
        recorded.filter(pl.col("id") != _DROPPED), _rows(phenotyped, _DRAWN), N_sample=0
    )
    assert ped_out["id"].to_list() == _ANALYSIS_IDS

    cohort = _assert_reconstructs(recorded, phenotyped, ped_out, trait_out)
    _assert_round_trip(recorded, ped_out, trait_out, cohort)

    outcomes = dict(zip(cohort["id"].to_list(), cohort.select(_OUTCOMES).rows(), strict=True))
    assert None not in outcomes[12]
    for ancestor in (3, 5, 8, 9, 14):
        assert set(outcomes[ancestor]) == {None}
    assert _rows(phenotyped, [14])["affected1"].item() is not None
    null_onset = dict(zip(_OUTCOMES, outcomes[20], strict=True))
    assert null_onset["t1"] is None
    assert null_onset["affected1"] is not None

    links = {row["id"]: row for row in selected_views(recorded, cohort).pedigree.iter_rows(named=True)}
    assert (links[20]["mother"], links[20]["father"]) == (12, 14)
    assert (links[23]["mother"], links[23]["father"]) == (-1, 14)
    assert links[25]["twin"] == -1


def test_empty_sample_gives_an_empty_cohort():
    """An empty sample builds an empty cohort whose views are the empty ascertainment outputs.

    Rejects a builder or reader that refuses, or changes columns or dtypes, when
    nobody is drawn.
    """
    recorded = _recorded()
    ped_out, trait_out = run_ascertainment(recorded, _phenotyped_trait(recorded).head(0), N_sample=0)
    assert ped_out.is_empty()
    assert trait_out.is_empty()
    cohort = _assert_reconstructs(recorded, trait_out, ped_out, trait_out)
    _assert_round_trip(recorded, ped_out, trait_out, cohort)
