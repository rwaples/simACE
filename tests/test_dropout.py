"""Unit tests for sim_ace.dropout and dropout-aware sibling detection."""

import numpy as np
import pandas as pd
import pytest

from sim_ace.dropout import run_dropout
from sim_ace.pedigree_graph import PedigreeGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_pedigree():
    """A small 3-generation pedigree (20 individuals)."""
    # Gen 0: founders (IDs 0-5), no parents, no twins
    # Gen 1: children of gen-0 couples (IDs 6-11)
    # Gen 2: children of gen-1 couples (IDs 12-19)
    ids = list(range(20))
    mothers = [-1] * 6 + [0, 0, 2, 2, 4, 4] + [6, 6, 8, 8, 10, 10, 10, 10]
    fathers = [-1] * 6 + [1, 1, 3, 3, 5, 5] + [9, 9, 7, 7, 11, 11, 11, 11]
    twins = [-1] * 20
    sex = [0, 1] * 10
    gens = [0] * 6 + [1] * 6 + [2] * 8
    return pd.DataFrame(
        {
            "id": ids,
            "mother": mothers,
            "father": fathers,
            "twin": twins,
            "sex": sex,
            "generation": gens,
        }
    )


@pytest.fixture
def twin_pedigree():
    """A pedigree with one twin pair (IDs 4 and 5)."""
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "mother": [-1, -1, 0, 0, 0, 0],
            "father": [-1, -1, 1, 1, 1, 1],
            "twin": [-1, -1, -1, -1, 5, 4],
            "sex": [0, 1, 0, 1, 0, 0],
            "generation": [0, 0, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def large_pedigree():
    """A larger pedigree for statistical tests (N=1500)."""
    rng = np.random.default_rng(99)
    n_founders = 500
    n_children = 1000
    n = n_founders + n_children

    ids = np.arange(n)
    mothers = np.full(n, -1, dtype=np.int64)
    fathers = np.full(n, -1, dtype=np.int64)
    twins = np.full(n, -1, dtype=np.int64)
    sex = rng.integers(0, 2, n)
    generation = np.zeros(n, dtype=np.int32)

    # Assign parents to children
    for i in range(n_founders, n):
        mothers[i] = rng.integers(0, n_founders)
        fathers[i] = rng.integers(0, n_founders)
        generation[i] = 1

    return pd.DataFrame(
        {
            "id": ids,
            "mother": mothers,
            "father": fathers,
            "twin": twins,
            "sex": sex,
            "generation": generation,
        }
    )


# ---------------------------------------------------------------------------
# TestDropoutBasic
# ---------------------------------------------------------------------------


class TestDropoutBasic:
    def test_zero_rate_is_noop(self, small_pedigree):
        """rate=0 returns identical DataFrame."""
        result = run_dropout(small_pedigree, {"pedigree_dropout_rate": 0, "seed": 42})
        pd.testing.assert_frame_equal(result, small_pedigree)

    def test_dropped_individuals_absent(self, small_pedigree):
        """Correct count removed, remaining IDs are a subset."""
        result = run_dropout(small_pedigree, {"pedigree_dropout_rate": 0.3, "seed": 42})
        n_expected = len(small_pedigree) - int(round(len(small_pedigree) * 0.3))
        assert len(result) == n_expected
        assert set(result["id"].values).issubset(set(small_pedigree["id"].values))

    def test_actual_fraction_matches_configured(self, small_pedigree):
        """n_dropped == round(n * rate)."""
        rate = 0.25
        n = len(small_pedigree)
        result = run_dropout(small_pedigree, {"pedigree_dropout_rate": rate, "seed": 42})
        n_dropped = n - len(result)
        assert n_dropped == int(round(n * rate))

    def test_parent_links_rewritten(self, small_pedigree):
        """No surviving row references a dropped ID as mother/father."""
        result = run_dropout(small_pedigree, {"pedigree_dropout_rate": 0.3, "seed": 42})
        surviving_ids = set(result["id"].values.tolist())
        surviving_ids.add(-1)  # -1 is valid (unknown)
        for col in ("mother", "father"):
            vals = result[col].values
            assert all(v in surviving_ids for v in vals), f"Dangling {col} reference found"

    def test_twin_links_rewritten(self, twin_pedigree):
        """If one twin is dropped, the surviving twin's twin field is set to -1."""
        # Force drop individual 4 (twin of 5) by using a specific seed
        # Instead, drop at 50% rate repeatedly until we get a case
        for seed in range(100):
            result = run_dropout(twin_pedigree, {"pedigree_dropout_rate": 0.5, "seed": seed})
            surviving_ids = set(result["id"].values.tolist())
            # Check: if 4 dropped but 5 survives
            if 4 not in surviving_ids and 5 in surviving_ids:
                row5 = result[result["id"] == 5].iloc[0]
                assert row5["twin"] == -1, "Twin link not severed when partner dropped"
                return
            # Or 5 dropped but 4 survives
            if 5 not in surviving_ids and 4 in surviving_ids:
                row4 = result[result["id"] == 4].iloc[0]
                assert row4["twin"] == -1, "Twin link not severed when partner dropped"
                return
        pytest.fail("Could not find seed that drops exactly one twin")

    def test_deterministic_with_same_seed(self, small_pedigree):
        """Same seed → same result."""
        params = {"pedigree_dropout_rate": 0.3, "seed": 77}
        r1 = run_dropout(small_pedigree, params)
        r2 = run_dropout(small_pedigree, params)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seed_different_result(self, small_pedigree):
        """Different seed → different drop set."""
        r1 = run_dropout(small_pedigree, {"pedigree_dropout_rate": 0.3, "seed": 77})
        r2 = run_dropout(small_pedigree, {"pedigree_dropout_rate": 0.3, "seed": 78})
        assert not r1["id"].equals(r2["id"])


# ---------------------------------------------------------------------------
# TestDropoutRelationships
# ---------------------------------------------------------------------------


class TestDropoutRelationships:
    def test_grandparent_grandchild_severed(self, small_pedigree):
        """Drop the parent between a GP and GC, verify the pair is undetectable."""
        # Individual 6 (gen 1) connects grandparent 0 to grandchild 12.
        # Drop individual 6: grandchild 12's mother link is severed.
        drop_ids = {6}
        keep = small_pedigree[~small_pedigree["id"].isin(drop_ids)].copy()
        for col in ("mother", "father", "twin"):
            vals = keep[col].values
            dangling = np.isin(vals, list(drop_ids)) & (vals >= 0)
            keep.loc[keep.index[dangling], col] = -1
        keep = keep.reset_index(drop=True)

        pg = PedigreeGraph(keep)
        pairs = pg.extract_pairs(skip_2nd_cousins=True)
        gp_gc = pairs["Grandparent-grandchild"]
        # 12 should NOT appear as grandchild of 0 or 1
        if len(gp_gc[0]) > 0:
            ids = keep["id"].values
            gc_ids = ids[gp_gc[0]]
            gp_ids = ids[gp_gc[1]]
            pair_set = set(zip(gc_ids.tolist(), gp_ids.tolist(), strict=True))
            assert (12, 0) not in pair_set, "GP-GC pair should be severed"
            assert (12, 1) not in pair_set, "GP-GC pair should be severed"

    def test_former_full_sibs_become_half_sibs(self):
        """Drop one parent of a full-sib pair → reclassified as half-sibs."""
        # Two children (2, 3) share mother 0 and father 1.
        # After dropping father 1, they share known mother but father is -1.
        df = pd.DataFrame(
            {
                "id": [0, 2, 3],
                "mother": [-1, 0, 0],
                "father": [-1, -1, -1],  # father 1 dropped, links severed
                "twin": [-1, -1, -1],
                "sex": [0, 0, 1],
                "generation": [0, 1, 1],
            }
        )
        pg = PedigreeGraph(df)
        full_sib, mat_hs, pat_hs = pg._sibling_pairs()

        # Should NOT be full sibs (father unknown for both)
        assert len(full_sib[0]) == 0, "Should not be full sibs with unknown father"
        # Should be maternal half sibs
        assert len(mat_hs[0]) == 1, "Should be detected as maternal half sibs"

    def test_half_sib_detection_through_surviving_parent(self):
        """Two children with one parent dropped are detected as half-sibs."""
        # Child 2: mother=0, father=-1 (dropped)
        # Child 3: mother=0, father=-1 (different dad, also dropped)
        df = pd.DataFrame(
            {
                "id": [0, 2, 3],
                "mother": [-1, 0, 0],
                "father": [-1, -1, -1],
                "twin": [-1, -1, -1],
                "sex": [0, 1, 0],
                "generation": [0, 1, 1],
            }
        )
        pg = PedigreeGraph(df)
        full_sib, mat_hs, pat_hs = pg._sibling_pairs()
        assert len(mat_hs[0]) == 1, "Should detect half-sibs through surviving parent"


# ---------------------------------------------------------------------------
# TestDropoutStatistical
# ---------------------------------------------------------------------------


class TestDropoutStatistical:
    def test_dropout_fraction_large_pedigree(self, large_pedigree):
        """Verify exact fraction on a large pedigree."""
        rate = 0.2
        n = len(large_pedigree)
        result = run_dropout(large_pedigree, {"pedigree_dropout_rate": rate, "seed": 42})
        n_dropped = n - len(result)
        assert n_dropped == int(round(n * rate))

    def test_no_dangling_parent_refs(self, large_pedigree):
        """All mother/father/twin values are either -1 or a surviving ID."""
        result = run_dropout(large_pedigree, {"pedigree_dropout_rate": 0.3, "seed": 42})
        surviving = set(result["id"].values.tolist())
        surviving.add(-1)
        for col in ("mother", "father", "twin"):
            vals = result[col].values
            assert all(v in surviving for v in vals), f"Dangling {col} ref"

    def test_index_reset(self, large_pedigree):
        """Output has clean 0-based integer index."""
        result = run_dropout(large_pedigree, {"pedigree_dropout_rate": 0.3, "seed": 42})
        assert list(result.index) == list(range(len(result)))
