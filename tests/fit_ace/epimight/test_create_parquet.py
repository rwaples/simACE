"""Tests for EPIMIGHT create_parquet helper functions."""

import numpy as np

from fit_ace.epimight.create_parquet import (
    _orient_pairs_by_generation,
    count_affected_relatives,
    count_total_relatives,
)

# ---------------------------------------------------------------------------
# _orient_pairs_by_generation
# ---------------------------------------------------------------------------


class TestOrientPairsByGeneration:
    def test_already_correct(self):
        # idx1 has higher gen (younger) — no swap needed
        idx1 = np.array([2, 3])
        idx2 = np.array([0, 1])
        generations = np.array([0, 0, 1, 1])
        result = _orient_pairs_by_generation([(idx1, idx2)], generations)
        np.testing.assert_array_equal(result[0][0], idx1)
        np.testing.assert_array_equal(result[0][1], idx2)

    def test_swaps_when_reversed(self):
        # idx1 has lower gen (older) — should swap
        idx1 = np.array([0, 1])
        idx2 = np.array([2, 3])
        generations = np.array([0, 0, 1, 1])
        result = _orient_pairs_by_generation([(idx1, idx2)], generations)
        np.testing.assert_array_equal(result[0][0], np.array([2, 3]))
        np.testing.assert_array_equal(result[0][1], np.array([0, 1]))

    def test_empty_pair_list(self):
        result = _orient_pairs_by_generation(
            [(np.array([], dtype=int), np.array([], dtype=int))],
            np.array([0, 1, 2]),
        )
        assert len(result[0][0]) == 0

    def test_mixed_orientations(self):
        # Pair 0: correct (2->0), Pair 1: reversed (0->2)
        idx1 = np.array([2, 0])
        idx2 = np.array([0, 2])
        generations = np.array([0, 0, 1])
        result = _orient_pairs_by_generation([(idx1, idx2)], generations)
        np.testing.assert_array_equal(result[0][0], np.array([2, 2]))  # both young
        np.testing.assert_array_equal(result[0][1], np.array([0, 0]))  # both old


# ---------------------------------------------------------------------------
# count_affected_relatives
# ---------------------------------------------------------------------------


class TestCountAffectedRelatives:
    def test_bidirectional(self):
        # 3 people: pair (0,1). Person 1 affected.
        idx1 = np.array([0])
        idx2 = np.array([1])
        affected = np.array([False, True, False])
        counts = count_affected_relatives([(idx1, idx2)], affected, n=3)
        assert counts[0] == 1  # 0's relative 1 is affected
        assert counts[1] == 0  # 1's relative 0 is not affected
        assert counts[2] == 0

    def test_unidirectional(self):
        # Only idx1 gets count of affected idx2
        idx1 = np.array([0])
        idx2 = np.array([1])
        affected = np.array([True, True, False])
        counts = count_affected_relatives([(idx1, idx2)], affected, n=3, unidirectional=True)
        assert counts[0] == 1  # idx1=0 counts affected idx2=1
        assert counts[1] == 0  # idx2 not counted in unidirectional

    def test_no_pairs_returns_zeros(self):
        affected = np.array([True, True])
        counts = count_affected_relatives(
            [(np.array([], dtype=int), np.array([], dtype=int))],
            affected,
            n=2,
        )
        np.testing.assert_array_equal(counts, [0, 0])

    def test_multiple_pair_types(self):
        # Two pair lists, both referencing person 0
        p1 = (np.array([0]), np.array([1]))
        p2 = (np.array([0]), np.array([2]))
        affected = np.array([False, True, True])
        counts = count_affected_relatives([p1, p2], affected, n=3)
        assert counts[0] == 2  # both relatives affected

    def test_hand_computed(self):
        # 4 people: pairs (0,1), (1,2), (2,3). All affected.
        idx1 = np.array([0, 1, 2])
        idx2 = np.array([1, 2, 3])
        affected = np.array([True, True, True, True])
        counts = count_affected_relatives([(idx1, idx2)], affected, n=4)
        # Person 0: rel=1 -> 1 affected
        # Person 1: rel=0,2 -> 2 affected
        # Person 2: rel=1,3 -> 2 affected
        # Person 3: rel=2 -> 1 affected
        np.testing.assert_array_equal(counts, [1, 2, 2, 1])


# ---------------------------------------------------------------------------
# count_total_relatives
# ---------------------------------------------------------------------------


class TestCountTotalRelatives:
    def test_bidirectional(self):
        idx1 = np.array([0, 0])
        idx2 = np.array([1, 2])
        counts = count_total_relatives([(idx1, idx2)], n=3)
        assert counts[0] == 2
        assert counts[1] == 1
        assert counts[2] == 1

    def test_unidirectional(self):
        idx1 = np.array([0])
        idx2 = np.array([1])
        counts = count_total_relatives([(idx1, idx2)], n=2, unidirectional=True)
        assert counts[0] == 1
        assert counts[1] == 0  # idx2 not counted

    def test_empty_returns_zeros(self):
        counts = count_total_relatives([(np.array([], dtype=int), np.array([], dtype=int))], n=3)
        np.testing.assert_array_equal(counts, [0, 0, 0])


