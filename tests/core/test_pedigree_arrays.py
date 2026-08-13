import numpy as np
import pandas as pd
import pytest

from simace.ascertainment import run_ascertainment
from simace.core.pedigree_arrays import PedigreeArrays
from simace.simulation.simulate import run_simulation


def _frame(ids):
    ids = np.asarray(ids, dtype=np.int32)
    n = len(ids)
    return pd.DataFrame(
        {
            "id": ids,
            "mother": np.full(n, -1, dtype=np.int32),
            "father": np.full(n, -1, dtype=np.int32),
            "sex": (np.arange(n) % 2).astype(np.int8),
            "A1": np.arange(n, dtype=np.float32) * 0.5,
        }
    )


DENSE = _frame(np.arange(12))
GAPPED = _frame([0, 3, 4, 9, 10, 25])


class TestConstruction:
    def test_dense_and_gapped_both_build(self):
        assert len(PedigreeArrays.from_frame(DENSE)) == 12
        assert len(PedigreeArrays.from_frame(GAPPED)) == 6

    def test_columns_are_views_not_copies(self):
        ped = PedigreeArrays.from_frame(DENSE)
        assert np.shares_memory(ped["A1"], DENSE["A1"].to_numpy())

    def test_row_order_is_preserved(self):
        ped = PedigreeArrays.from_frame(GAPPED)
        np.testing.assert_array_equal(ped.ids, GAPPED["id"].to_numpy())
        np.testing.assert_array_equal(ped.positions(GAPPED["id"].to_numpy()), np.arange(6))

    def test_missing_id_column_rejected(self):
        with pytest.raises(ValueError, match="requires an 'id' column"):
            PedigreeArrays({"sex": np.arange(3)})

    def test_from_frame_accepts_polars(self):
        import polars as pl

        ped_pd = PedigreeArrays.from_frame(GAPPED)
        ped_pl = PedigreeArrays.from_frame(pl.from_pandas(GAPPED))
        assert ped_pl.columns == ped_pd.columns
        np.testing.assert_array_equal(ped_pl.ids, ped_pd.ids)
        np.testing.assert_array_equal(ped_pl.gather("A1", GAPPED["id"].to_numpy()), ped_pd["A1"])

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicated id"):
            PedigreeArrays.from_frame(_frame([0, 1, 1, 2]))

    def test_negative_ids_rejected_at_construction(self):
        with pytest.raises(ValueError, match="negative id"):
            PedigreeArrays.from_frame(_frame([0, -1, 2]))

    def test_column_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            PedigreeArrays({"id": np.arange(4), "sex": np.arange(3)})

    def test_sparse_id_space_rejected(self):
        with pytest.raises(ValueError, match="too sparse"):
            PedigreeArrays({"id": np.array([0, 2_000_000_000], dtype=np.int64)})

    def test_empty_pedigree(self):
        ped = PedigreeArrays.from_frame(_frame([]))
        assert len(ped) == 0
        assert ped.contains(np.array([0, 1])).tolist() == [False, False]
        assert ped.positions(np.array([], dtype=np.int64)).shape == (0,)


class TestGather:
    @pytest.mark.parametrize("df", [DENSE, GAPPED], ids=["dense", "gapped"])
    def test_matches_the_pandas_loc_it_replaces(self, df):
        ped = PedigreeArrays.from_frame(df)
        indexed = df.set_index("id")
        ids = df["id"].to_numpy()[::-1]  # reversed, to prove order is honoured
        for col in ("sex", "A1", "mother"):
            np.testing.assert_array_equal(ped.gather(col, ids), indexed.loc[ids, col].to_numpy())

    def test_gather_equals_column_indexed_by_positions(self):
        ped = PedigreeArrays.from_frame(GAPPED)
        ids = np.array([9, 0, 25])
        np.testing.assert_array_equal(ped.gather("A1", ids), ped["A1"][ped.positions(ids)])

    def test_repeated_ids_are_allowed(self):
        ped = PedigreeArrays.from_frame(DENSE)
        np.testing.assert_array_equal(ped.gather("A1", np.array([3, 3, 3])), np.full(3, 1.5, dtype=np.float32))

    def test_unknown_column_raises(self):
        ped = PedigreeArrays.from_frame(DENSE)
        with pytest.raises(KeyError, match="no column 'nope'"):
            ped.gather("nope", np.array([0]))


class TestStrictMisses:
    """The contract that has no equivalent in the idioms this replaces."""

    def test_unknown_id_raises_like_loc_did(self):
        ped = PedigreeArrays.from_frame(GAPPED)
        with pytest.raises(KeyError, match="not in this pedigree"):
            ped.positions(np.array([0, 5]))  # 5 is inside the range but absent

    def test_id_above_max_raises(self):
        ped = PedigreeArrays.from_frame(GAPPED)
        with pytest.raises(KeyError, match="above max_id"):
            ped.positions(np.array([0, 99]))

    def test_negative_id_raises_rather_than_wrapping(self):
        """A bare pos[-1] would silently return the LAST row; -1 is the sentinel."""
        ped = PedigreeArrays.from_frame(DENSE)
        with pytest.raises(ValueError, match="missing-parent sentinel"):
            ped.positions(np.array([0, -1]))

    def test_negative_id_raises_through_gather(self):
        ped = PedigreeArrays.from_frame(DENSE)
        with pytest.raises(ValueError, match="missing-parent sentinel"):
            ped.gather("A1", np.array([-1]))

    def test_error_names_the_offending_ids(self):
        ped = PedigreeArrays.from_frame(GAPPED)
        with pytest.raises(KeyError, match=r"\[1, 2\]"):
            ped.positions(np.array([0, 1, 2]))


class TestContains:
    def test_matches_the_isin_it_replaces(self):
        ped = PedigreeArrays.from_frame(GAPPED)
        probe = np.array([0, 1, 3, 25, 26, 100])
        expected = pd.Series(probe).isin(GAPPED["id"]).to_numpy()
        np.testing.assert_array_equal(ped.contains(probe), expected)

    def test_sentinel_is_absent_not_an_error(self):
        """Ascertainment severs mother/father independently, so -1 reaches this."""
        ped = PedigreeArrays.from_frame(GAPPED)
        np.testing.assert_array_equal(ped.contains(np.array([-1, 0, -1])), [False, True, False])

    def test_out_of_range_is_absent_not_an_error(self):
        ped = PedigreeArrays.from_frame(GAPPED)
        np.testing.assert_array_equal(ped.contains(np.array([25, 26, 10_000])), [True, False, False])

    def test_shape_is_preserved(self):
        ped = PedigreeArrays.from_frame(DENSE)
        assert ped.contains(np.array([[0, 1], [2, 99]])).shape == (2, 2)


@pytest.fixture(scope="module")
def severed_pedigree():
    """An ascertained pedigree carrying genuinely severed parent links.

    Dropout removes individuals before the ancestor closure is built, so a
    surviving child's parent may be absent from the closure and gets rewritten
    to -1 by ``_sever_dangling_links``. That produces rows with exactly one
    real parent -- the case that makes ``contains(-1)`` load-bearing.
    """
    ped = run_simulation(
        seed=42,
        N=200,
        G_ped=4,
        G_sim=4,
        mating_lambda=0.5,
        p_mztwin=0.02,
        A1=0.5,
        C1=0.2,
        E1=0.3,
        A2=0.5,
        C2=0.2,
        E2=0.3,
        rA=0.3,
        rC=0.5,
        assort1=0.0,
        assort2=0.0,
    )
    max_gen = int(ped["generation"].max())
    phenotyped = ped[ped["generation"] >= max_gen - 1].reset_index(drop=True)
    rng = np.random.default_rng(0)
    trait = pd.DataFrame(
        {
            "id": phenotyped["id"].to_numpy(),
            "generation": phenotyped["generation"].to_numpy(),
            "sex": phenotyped["sex"].to_numpy(),
            "affected1": rng.random(len(phenotyped)) < 0.2,
            "affected2": rng.random(len(phenotyped)) < 0.2,
        }
    )
    ascertained, _ = run_ascertainment(ped, trait, dropout_rate=0.2, seed=7)
    return ascertained


class TestSeveredParentPedigree:
    """Severed parents are real, not hypothetical: dropout produces them in bulk.

    ``run_ascertainment``'s docstring said parent links were "safe by
    construction"; that holds only at ``dropout_rate=0``.
    """

    def test_fixture_actually_has_one_parent_rows(self, severed_pedigree):
        """Guard against a vacuous suite below if severing behaviour ever changes."""
        mother = severed_pedigree["mother"].to_numpy()
        father = severed_pedigree["father"].to_numpy()
        one_parent = int(((mother == -1) ^ (father == -1)).sum())
        assert one_parent > 0, "fixture no longer exercises severed parents"

    @pytest.mark.parametrize("col", ["mother", "father"])
    def test_contains_matches_isin_with_sentinels_present(self, severed_pedigree, col):
        ped = PedigreeArrays.from_frame(severed_pedigree)
        parents = severed_pedigree[col].to_numpy()
        assert (parents == -1).any(), f"no severed {col} in fixture"
        expected = pd.Series(parents).isin(severed_pedigree["id"]).to_numpy()
        np.testing.assert_array_equal(ped.contains(parents), expected)

    def test_reproduces_the_am_relatedness_parent_filter(self, severed_pedigree):
        """The exact idiom at am_relatedness.py:84, which sees these -1s."""
        ped = PedigreeArrays.from_frame(severed_pedigree)
        indexed = severed_pedigree.set_index("id")
        non_founders = severed_pedigree[severed_pedigree["mother"] != -1]
        pairs = non_founders[["mother", "father"]].drop_duplicates()
        mother, father = pairs["mother"].to_numpy(), pairs["father"].to_numpy()
        assert (father == -1).any(), "fixture should retain a mother-only row"

        expected = pairs["mother"].isin(indexed.index).to_numpy() & pairs["father"].isin(indexed.index).to_numpy()
        np.testing.assert_array_equal(ped.contains(mother) & ped.contains(father), expected)

        # And the surviving pairs still gather identically to the .loc they replace.
        kept_m, kept_f = mother[expected], father[expected]
        np.testing.assert_array_equal(ped.gather("A1", kept_m), indexed.loc[kept_m, "A1"].to_numpy())
        np.testing.assert_array_equal(ped.gather("A1", kept_f), indexed.loc[kept_f, "A1"].to_numpy())

    def test_gathering_severed_parents_raises_rather_than_wrapping(self, severed_pedigree):
        """Without the guard, pos[-1] would return the last row's values."""
        ped = PedigreeArrays.from_frame(severed_pedigree)
        father = severed_pedigree["father"].to_numpy()
        with pytest.raises(ValueError, match="missing-parent sentinel"):
            ped.gather("A1", father)

    def test_severed_twin_links_also_handled(self, severed_pedigree):
        ped = PedigreeArrays.from_frame(severed_pedigree)
        twin = severed_pedigree["twin"].to_numpy()
        expected = pd.Series(twin).isin(severed_pedigree["id"]).to_numpy()
        np.testing.assert_array_equal(ped.contains(twin), expected)


class TestColumnAccess:
    def test_membership_and_listing(self):
        ped = PedigreeArrays.from_frame(DENSE)
        assert "sex" in ped
        assert "nope" not in ped
        assert ped.columns == ["A1", "father", "id", "mother", "sex"]

    def test_whole_column_is_row_ordered(self):
        ped = PedigreeArrays.from_frame(GAPPED)
        np.testing.assert_array_equal(ped["A1"], GAPPED["A1"].to_numpy())
