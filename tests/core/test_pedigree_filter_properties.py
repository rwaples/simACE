"""Property-based tests for the observed-pedigree ancestor closure.

``filter_pedigree_to_observed`` returns the observed set plus every ancestor
reachable through in-pedigree parent pointers. Over random acyclic pedigrees
this pins the closure's set semantics: subset, ancestor-closedness,
idempotence, and minimality (no orphan rows) — properties a handful of
hand-built topologies cannot exercise.
"""

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from simace.core.pedigree_filter import filter_pedigree_to_observed


@st.composite
def _pedigree_and_observed(draw):
    """A random acyclic pedigree (parents are earlier ids, -1, or absent) + observed subset."""
    n = draw(st.integers(min_value=0, max_value=30))
    mothers, fathers = [], []
    absent = n + 1  # an id never present in the frame: exercises the dropped-parent branch
    for i in range(n):
        choices = st.sampled_from([-1, absent, *range(i)])
        mothers.append(draw(choices))
        fathers.append(draw(choices))
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "mother": np.array(mothers, dtype=np.int64),
            "father": np.array(fathers, dtype=np.int64),
        }
    )
    if n == 0:
        observed = np.array([], dtype=np.int64)
    else:
        observed = np.array(draw(st.lists(st.sampled_from(range(n)), unique=True)), dtype=np.int64)
    return df, observed


def _parent_map(df: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Map each id to its ``(mother, father)`` pair."""
    return dict(zip(df["id"].tolist(), zip(df["mother"].tolist(), df["father"].tolist(), strict=True), strict=True))


@given(_pedigree_and_observed())
def test_closure_subset_and_ancestor_closed(case):
    df, observed = case
    out = filter_pedigree_to_observed(df, observed)

    kept = set(out["id"].tolist())
    present = set(df["id"].tolist())
    parents = _parent_map(df)

    # subset + every observed id retained
    assert kept <= present
    assert set(observed.tolist()) <= kept

    # ancestor-closed: any in-pedigree parent of a kept id is also kept
    for i in kept:
        for p in parents[i]:
            if p >= 0 and p in present:
                assert p in kept


@given(_pedigree_and_observed())
def test_closure_is_idempotent(case):
    df, observed = case
    out1 = filter_pedigree_to_observed(df, observed)
    # observed is a subset of out1 by construction, so re-filtering is well-defined
    out2 = filter_pedigree_to_observed(out1, observed)
    np.testing.assert_array_equal(out1["id"].to_numpy(), out2["id"].to_numpy())


@given(_pedigree_and_observed())
def test_closure_is_minimal(case):
    df, observed = case
    out = filter_pedigree_to_observed(df, observed)

    kept = set(out["id"].tolist())
    parents = _parent_map(df)
    obs = set(observed.tolist())

    referenced_parents: set[int] = set()
    for i in kept:
        for p in parents[i]:
            if p >= 0:
                referenced_parents.add(int(p))

    # no orphans: every kept id is either observed or a parent of some kept id
    for i in kept:
        assert i in obs or i in referenced_parents
