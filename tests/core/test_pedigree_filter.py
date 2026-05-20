"""Unit tests for :func:`simace.core.pedigree_filter.filter_pedigree_to_observed`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simace.core.pedigree_filter import filter_pedigree_to_observed


def _ped(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """Build a minimal pedigree DataFrame from ``(id, mother, father)`` tuples."""
    return pd.DataFrame(rows, columns=["id", "mother", "father"])


def test_empty_observed_returns_empty():
    """No seeds → no closure."""
    df_ped = _ped([(0, -1, -1), (1, -1, -1), (2, 0, 1)])
    out = filter_pedigree_to_observed(df_ped, np.array([], dtype=np.int64))
    assert len(out) == 0
    assert list(out.columns) == ["id", "mother", "father"]


def test_single_observed_recovers_full_ancestry():
    """Closure walks back through both parents recursively."""
    # 0,1 founders → 2 (child of 0,1); 3,4 founders → 5 (child of 3,4); 6 = child of 2,5
    df_ped = _ped(
        [
            (0, -1, -1),
            (1, -1, -1),
            (2, 1, 0),
            (3, -1, -1),
            (4, -1, -1),
            (5, 4, 3),
            (6, 5, 2),
        ]
    )
    out = filter_pedigree_to_observed(df_ped, np.array([6]))
    assert sorted(out["id"].tolist()) == [0, 1, 2, 3, 4, 5, 6]


def test_late_gen_only_recovers_full_pedigree():
    """Observing only the last generation should still pull all ancestors."""
    # Two-gen pedigree: 0,1 founders, 2,3 are kids
    df_ped = _ped([(0, -1, -1), (1, -1, -1), (2, 1, 0), (3, 1, 0)])
    out = filter_pedigree_to_observed(df_ped, np.array([2, 3]))
    assert sorted(out["id"].tolist()) == [0, 1, 2, 3]


def test_founder_observed_returns_just_self():
    """A founder has no ancestors to add."""
    df_ped = _ped([(0, -1, -1), (1, -1, -1), (2, 1, 0)])
    out = filter_pedigree_to_observed(df_ped, np.array([0]))
    assert out["id"].tolist() == [0]


def test_parent_absent_from_df_ped_halts_traversal():
    """If a parent ID is not present in df_ped (e.g. dropped), traversal halts."""
    # Individual 2's mother is 99 — not in df_ped (treat as dropped)
    df_ped = _ped([(0, -1, -1), (2, 99, 0)])
    out = filter_pedigree_to_observed(df_ped, np.array([2]))
    # 2 + father (0) survive; mother 99 is omitted (not present)
    assert sorted(out["id"].tolist()) == [0, 2]


def test_unobserved_sibling_not_included():
    """Siblings of observed individuals are NOT ancestors and must not enter the closure."""
    # 0,1 founders; 2 and 3 are full sibs
    df_ped = _ped([(0, -1, -1), (1, -1, -1), (2, 1, 0), (3, 1, 0)])
    out = filter_pedigree_to_observed(df_ped, np.array([2]))
    # 3 is a sibling, not an ancestor — must be excluded
    assert sorted(out["id"].tolist()) == [0, 1, 2]


def test_raises_when_observed_id_missing():
    """Observed IDs not in df_ped are an error, not silent skip."""
    df_ped = _ped([(0, -1, -1), (1, -1, -1)])
    with pytest.raises(ValueError, match="not in df_ped"):
        filter_pedigree_to_observed(df_ped, np.array([0, 42]))


def test_preserves_row_order_and_extra_columns():
    """Filter preserves original row order and any additional columns."""
    df_ped = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "mother": [-1, -1, 1, 1],
            "father": [-1, -1, 0, 0],
            "sex": [1, 0, 1, 0],
            "generation": [0, 0, 1, 1],
        }
    )
    out = filter_pedigree_to_observed(df_ped, np.array([2]))
    # Order: 0, 1, 2 — same as input order
    assert out["id"].tolist() == [0, 1, 2]
    assert out["sex"].tolist() == [1, 0, 1]
    assert out["generation"].tolist() == [0, 0, 1]
