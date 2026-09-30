"""Ascertainment inputs shared by the ascertainment and cohort property tests.

:func:`ascertainment_inputs` draws a recorded pedigree, an informative
censored trait frame over its phenotyped window, and ``run_ascertainment``
arguments that always succeed.  It is a structural contract for the stages
that only select rows, not a sample from the simulation or censoring models.

Pedigree: ``pedigree_frame(twins=True, liabilities=True)`` through
``relabel_ids``, so ids are gapped and never equal row positions.

Trait: one row per pedigree row in the trailing ``G_pheno`` generations, in
pedigree order, with exactly ``TRAIT_CENSORED_COLUMNS`` in that order.

- ``id`` has the pedigree's id dtype (int32).
- ``t1``, ``t2``, ``death_age``, ``t_observed1``, ``t_observed2`` are Float64.
  Each column has its own disjoint value range and unique values within it, so
  a swap between columns or rows changes a value.  Values are multiples of 0.5
  below 5000, exact in the Float32 that storage narrows them to.  The raw
  onsets ``t1`` and ``t2`` are null on a drawn subset of rows; the others are
  never null.
- ``affected``, ``age_censored``, ``death_censored`` are Boolean, never null,
  with ``affected = NOT (age_censored OR death_censored)`` and exactly one of
  the three set per row and trait.  The onset and censoring values are not
  otherwise consistent with each other.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl
from hypothesis import strategies as st

from simace.ascertainment.runner import run_ascertainment
from simace.core.trait_schema import TRAIT_CENSORED_COLUMNS
from tests.conftest import pedigree_frame, relabel_ids

__all__ = ["AscertainmentInput", "ascertainment_inputs"]

_TIMES = ("t1", "t2", "death_age", "t_observed1", "t_observed2")
_RAW_ONSETS = ("t1", "t2")
_STATUSES = (
    # (affected, age_censored, death_censored)
    (True, False, False),
    (False, True, False),
    (False, False, True),
)


class AscertainmentInput(NamedTuple):
    """A recorded pedigree, its censored trait rows, and ``run_ascertainment`` keyword arguments."""

    pedigree: pl.DataFrame
    trait: pl.DataFrame
    kwargs: dict[str, float | int]

    def run(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return ``run_ascertainment``'s (analysis pedigree, sample trait) for this input."""
        return run_ascertainment(self.pedigree, self.trait, **self.kwargs)


@st.composite
def _informative_trait(draw, ids: pl.Series) -> pl.DataFrame:
    n = len(ids)
    columns = {"id": ids}
    for index, name in enumerate(_TIMES):
        gaps = draw(st.lists(st.integers(min_value=1, max_value=3), min_size=n, max_size=n))
        steps = np.cumsum(gaps)[draw(st.permutations(range(n)))]
        values = (1000.0 * index + 0.5 * steps).tolist()
        if name in _RAW_ONSETS:
            nulls = draw(st.lists(st.booleans(), min_size=n, max_size=n))
            values = [None if null else value for value, null in zip(values, nulls, strict=True)]
        columns[name] = pl.Series(name, values, dtype=pl.Float64)
    for trait in (1, 2):
        rows = draw(st.lists(st.sampled_from(_STATUSES), min_size=n, max_size=n))
        for prefix, flags in zip(("affected", "age_censored", "death_censored"), zip(*rows, strict=True), strict=True):
            columns[f"{prefix}{trait}"] = pl.Series(f"{prefix}{trait}", flags, dtype=pl.Boolean)
    return pl.DataFrame([columns[name] for name in TRAIT_CENSORED_COLUMNS])


@st.composite
def ascertainment_inputs(draw) -> AscertainmentInput:
    """Draw an input on which ``run_ascertainment`` succeeds, with a draw whenever ``0 < N_sample < pool``.

    ``case_ascertainment_ratio`` is positive, so the zero-weight refusal
    cannot fire; that refusal is covered by the existing ascertainment tests.
    ``dropout_rate`` is an exact drop count in ``[0, n - 1]`` over ``n``, so
    dropout never refuses either.
    """
    pedigree = relabel_ids(draw(pedigree_frame(twins=True, liabilities=True)), draw(st.data()))
    generations = sorted(set(pedigree["generation"].to_list()))
    g_pheno = draw(st.integers(min_value=1, max_value=len(generations)))
    ids = pedigree.filter(pl.col("generation").is_in(generations[-g_pheno:]))["id"]
    trait = draw(_informative_trait(ids))
    n_drop = draw(st.integers(min_value=0, max_value=len(pedigree) - 1))
    kwargs = {
        "dropout_rate": n_drop / len(pedigree),
        "case_ascertainment_ratio": draw(st.sampled_from([0.25, 1.0, 4.0])),
        "N_sample": draw(st.integers(min_value=-2, max_value=len(trait) + 2)),
        "seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
    }
    return AscertainmentInput(pedigree, trait, kwargs)
