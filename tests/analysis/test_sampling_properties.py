"""Property-based tests for :mod:`simace.analysis.stats.sampling`.

``create_sample`` draws per generation and then adds back the drawn rows'
direct parents.  The invariants below are chosen so each is observable from the
public result alone — none replays NumPy's RNG or asserts anything about which
rows the draw happened to pick.
"""

import numpy as np
import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from simace.analysis.stats.sampling import create_sample
from tests.conftest import pedigree_frame, relabel_ids

_ROW_MARKER = "_source_row"


def _with_row_marker(frame: pl.DataFrame) -> pl.DataFrame:
    """Tag each row with its original position so ordering can be asserted directly.

    Increasing ``id`` is not a valid proxy: ``create_sample`` is also expected
    to preserve *id* order, and using one to check the other would make the two
    properties circular.
    """
    return frame.with_columns(pl.Series(_ROW_MARKER, np.arange(len(frame), dtype=np.int64)))


def _generation_counts(frame: pl.DataFrame) -> dict[int, int]:
    """Rows per generation, via ``np.unique`` rather than ``value_counts``.

    ``Series.value_counts`` is a hash aggregation with no row-order guarantee,
    so reading its two columns is not reproducible across calls — which makes a
    Hypothesis strategy built on it non-deterministic.
    """
    values, counts = np.unique(frame["generation"].to_numpy(), return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


@st.composite
def _capped_pedigree(draw):
    """Draw ``(frame, n_per_gen)`` where every generation is at or below the cap."""
    frame = draw(pedigree_frame(twins=True))
    largest = max(_generation_counts(frame).values())
    return frame, draw(st.integers(min_value=largest, max_value=largest + 5))


@st.composite
def _parentless_frame(draw):
    """Draw a frame whose parent pointers are all ``-1``, plus a cap.

    With no parents to add back, the output *is* the draw, so the per-generation
    count is exactly observable without replaying the RNG.
    """
    frame = draw(pedigree_frame())
    frame = frame.with_columns(
        pl.Series("mother", np.full(len(frame), -1, dtype=np.int32)),
        pl.Series("father", np.full(len(frame), -1, dtype=np.int32)),
        pl.Series("twin", np.full(len(frame), -1, dtype=np.int32)),
    )
    return frame, draw(st.integers(min_value=1, max_value=8))


@st.composite
def _extended_last_generation(draw, frame: pl.DataFrame, generation: int, extra: int) -> pl.DataFrame:
    """Append ``extra`` valid children to ``generation``, preserving every invariant.

    Parents come from the previous generation with the usual female/mother,
    male/father roles, and each new child joins its mother's household —
    reusing hers when she already has children, otherwise opening a new one.
    """
    ids = frame["id"].to_numpy()
    households = frame["household_id"].to_numpy()
    parents = frame["generation"].to_numpy() == generation - 1
    females = ids[parents & (frame["sex"].to_numpy() == 0)].tolist()
    males = ids[parents & (frame["sex"].to_numpy() == 1)].tolist()

    household_of_mother = {
        int(mother): int(household)
        for mother, household in zip(frame["mother"].to_numpy(), households, strict=True)
        if mother != -1
    }
    next_household = int(households.max()) + 1

    new_mothers = [draw(st.sampled_from(females)) for _ in range(extra)]
    new_households = []
    for mother in new_mothers:
        if mother not in household_of_mother:
            household_of_mother[mother] = next_household
            next_household += 1
        new_households.append(household_of_mother[mother])

    next_id = int(ids.max()) + 1
    addition = pl.DataFrame(
        {
            "id": np.arange(next_id, next_id + extra, dtype=np.int32),
            "generation": np.full(extra, generation, dtype=np.int32),
            "sex": np.asarray([draw(st.integers(min_value=0, max_value=1)) for _ in range(extra)], dtype=np.int32),
            "mother": np.asarray(new_mothers, dtype=np.int32),
            "father": np.asarray([draw(st.sampled_from(males)) for _ in range(extra)], dtype=np.int32),
            "twin": np.full(extra, -1, dtype=np.int32),
            "household_id": np.asarray(new_households, dtype=np.int32),
        }
    )
    # The ACE and liability columns carry no information for create_sample, but
    # must match the frame's dtypes for the concat.
    addition = addition.with_columns(
        pl.lit(0.0).cast(frame.schema[column]).alias(column)
        for column in frame.columns
        if column not in addition.columns
    )
    return pl.concat([frame, addition.select(frame.columns)], how="vertical")


@st.composite
def _only_last_generation_over_cap(draw):
    """Draw ``(frame, n_per_gen)`` where only the final generation exceeds the cap.

    Every earlier generation is then retained whole, so each output row of the
    final generation is necessarily a *drawn* row — making one-level parent
    retention observable without knowing which rows were drawn.  The shape is
    built rather than filtered for: when the drawn frame's last generation is
    not already the largest, it is extended until it is.
    """
    frame = draw(pedigree_frame(twins=True))
    counts = _generation_counts(frame)
    if len(counts) == 1:
        return None  # a single generation has no in-frame parents to retain
    last = max(counts)
    n_per_gen = max(count for generation, count in counts.items() if generation != last)
    shortfall = n_per_gen + 1 - counts[last]
    if shortfall > 0:
        frame = draw(_extended_last_generation(frame, last, shortfall))
    return frame, n_per_gen


class TestCreateSample:
    """Identity, subsetting, per-generation cap, and one-level parent retention."""

    @given(case=_capped_pedigree(), seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_identity_below_the_cap(self, case, seed):
        """When no generation exceeds the cap the frame is returned unchanged."""
        frame, n_per_gen = case
        assert create_sample(frame, seed=seed, n_per_gen=n_per_gen).equals(frame)

    @given(
        pedigree=pedigree_frame(twins=True),
        n_per_gen=st.integers(min_value=1, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        data=st.data(),
    )
    def test_subset_deduplicated_and_row_order_preserving(self, pedigree, n_per_gen, seed, data):
        """Output rows are a deduplicated, order-preserving subset of the input.

        Run on relabelled ids as well as dense ones: ``create_sample`` builds a
        direct-address ``id -> row`` table sized by ``max_id + 1``, so gapped
        ids exercise a different path through it.
        """
        frame = _with_row_marker(relabel_ids(pedigree, data) if data.draw(st.booleans()) else pedigree)
        result = create_sample(frame, seed=seed, n_per_gen=n_per_gen)

        markers = result[_ROW_MARKER].to_numpy()
        assert len(set(markers.tolist())) == len(markers)
        assert set(markers.tolist()) <= set(range(len(frame)))
        assert np.all(np.diff(markers) > 0)
        assert result.columns == frame.columns

    @given(
        case=_parentless_frame(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_per_generation_count_is_the_cap(self, case, seed):
        """With no parents to add back, each generation contributes ``min(count, cap)``."""
        frame, n_per_gen = case
        result = create_sample(frame, seed=seed, n_per_gen=n_per_gen)

        for generation in frame["generation"].unique().to_list():
            available = int((frame["generation"] == generation).sum())
            kept = int((result["generation"] == generation).sum())
            assert kept == min(available, n_per_gen)

    @given(case=_only_last_generation_over_cap(), seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_drawn_rows_keep_their_in_frame_parents(self, case, seed):
        """Every drawn row's in-frame mother and father are present in the output.

        Deliberately *not* generalized to transitive ancestor closure: the added
        parent rows are not themselves re-closed, so in an unconstrained
        multi-generation frame the output legitimately carries parent references
        pointing outside itself (26 such references measured on a
        3-generation, 300-row frame at ``n_per_gen=10``).
        """
        if case is None:
            return
        frame, n_per_gen = case
        result = create_sample(frame, seed=seed, n_per_gen=n_per_gen)

        last = int(frame["generation"].max())
        present = set(result["id"].to_list())
        in_frame = set(frame["id"].to_list())

        drawn = result.filter(pl.col("generation") == last)
        for mother, father in zip(drawn["mother"].to_list(), drawn["father"].to_list(), strict=True):
            for parent in (mother, father):
                if parent != -1 and parent in in_frame:
                    assert parent in present
