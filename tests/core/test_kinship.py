"""Smoke tests for the moved kinship module (sim_ace.core.kinship).

Comprehensive coverage lives in ``tests/fit_ace/pafgrs/test_pafgrs.py``
(which also imports from the new canonical path).  These tests only
verify the public symbols are importable from ``sim_ace.core.kinship``
and that values match expectations on a small pedigree — protecting
against accidental regressions if anyone refactors back into fit_ace.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_ace.core.kinship import build_kinship_from_pairs, build_sparse_kinship


def _toy_pedigree_arrays():
    """3-gen, 6-individual pedigree (mirrors _make_simple_pedigree in fit_ace tests)."""
    ids = np.array([0, 1, 2, 3, 4, 5])
    mothers = np.array([-1, -1, 0, 0, 2, -1])
    fathers = np.array([-1, -1, 1, 1, 5, -1])
    twins = np.array([-1, -1, -1, -1, -1, -1])
    return ids, mothers, fathers, twins


def test_build_sparse_kinship_self_and_parent_offspring():
    ids, mothers, fathers, twins = _toy_pedigree_arrays()
    K = build_sparse_kinship(ids, mothers, fathers, twins).toarray()
    # Non-inbred self-kinship
    assert np.allclose(np.diag(K), 0.5)
    # Parent-offspring kinship = 0.25
    assert K[0, 2] == pytest.approx(0.25)
    assert K[1, 2] == pytest.approx(0.25)
    # Full sibs
    assert K[2, 3] == pytest.approx(0.25)
    # Grandparent-grandchild
    assert K[0, 4] == pytest.approx(0.125)
    # Unrelated founders
    assert K[0, 1] == pytest.approx(0.0)


def test_build_kinship_from_pairs_matches_sparse():
    """Pair-based construction agrees with DP on a non-consanguineous pedigree."""
    import pandas as pd

    ids, mothers, fathers, twins = _toy_pedigree_arrays()
    ped = pd.DataFrame(
        {
            "id": ids,
            "mother": mothers,
            "father": fathers,
            "twin": twins,
            "sex": np.array([0, 1, 0, 0, 0, 1]),
            "generation": np.array([0, 0, 1, 1, 2, 1]),
        }
    )
    K_dp = build_sparse_kinship(ids, mothers, fathers, twins).toarray()
    K_pairs = build_kinship_from_pairs(ped, ndegree=2).toarray()
    # Pair-based capped at 1st-cousin degree; the 3-gen toy only has
    # relationships within that cutoff, so both should match on the
    # upper-triangular off-diagonal pairs it captures.
    # Compare only non-zero entries of K_pairs (which is the subset it emits).
    nonzero = K_pairs != 0
    np.testing.assert_allclose(K_pairs[nonzero], K_dp[nonzero])
