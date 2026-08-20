"""Tests for outcomes-only trait schemas and hydration."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from simace.core.trait_schema import (
    TRAIT_CENSORED_COLUMNS,
    TRAIT_RAW_COLUMNS,
    hydrate_trait,
    strip_trait_to_outcomes,
)


def _pedigree() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": np.array([3, 1, 2], dtype=np.int32),
            "generation": np.array([1, 0, 0], dtype=np.int32),
            "sex": np.array([0, 1, 0], dtype=np.int8),
            "mother": np.array([1, -1, -1], dtype=np.int32),
            "father": np.array([2, -1, -1], dtype=np.int32),
            "liability1": np.array([0.3, 0.1, 0.2], dtype=np.float64),
        }
    )


def _raw_trait() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": np.array([2, 3], dtype=np.int32),
            "t1": np.array([10.0, 20.0], dtype=np.float32),
            "t2": np.array([30.0, 40.0], dtype=np.float32),
        }
    )


def _censored_trait() -> pl.DataFrame:
    df = _raw_trait()
    new_cols = [pl.Series("death_age", np.array([80.0, 70.0], dtype=np.float32))]
    for trait in (1, 2):
        new_cols.append(pl.Series(f"age_censored{trait}", np.array([False, True])))
        new_cols.append(pl.Series(f"t_observed{trait}", np.array([10.0, 60.0], dtype=np.float32)))
        new_cols.append(pl.Series(f"death_censored{trait}", np.array([False, False])))
        new_cols.append(pl.Series(f"affected{trait}", np.array([True, False])))
    return df.with_columns(new_cols)


class TestStripTraitToOutcomes:
    def test_strips_raw_trait_to_ordered_schema(self):
        df = _raw_trait().with_columns(
            pl.Series("generation", [0, 1]),
            pl.lit("drop").alias("extra"),
        )

        out = strip_trait_to_outcomes(df, "raw")

        assert list(out.columns) == list(TRAIT_RAW_COLUMNS)
        assert list(out["id"]) == [2, 3]

    def test_strips_censored_trait_to_ordered_schema(self):
        df = _censored_trait().with_columns(pl.Series("liability1", [0.2, 0.3]))

        out = strip_trait_to_outcomes(df, "censored")

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == list(TRAIT_CENSORED_COLUMNS)

    def test_rejects_missing_required_outcome_column(self):
        df = _raw_trait().drop("t2")

        with pytest.raises(ValueError, match=r"missing required columns.*t2"):
            strip_trait_to_outcomes(df, "raw")

    def test_rejects_pandas_input(self):
        df = pd.DataFrame({"id": [2, 3], "t1": [10.0, 20.0], "t2": [30.0, 40.0]})

        with pytest.raises(TypeError, match=r"polars DataFrame since the polars migration"):
            strip_trait_to_outcomes(df, "raw")


class TestHydrateTrait:
    def test_hydrates_with_all_pedigree_columns_first_and_preserves_trait_order(self):
        trait = _raw_trait()
        pedigree = _pedigree()

        out = hydrate_trait(trait, pedigree, kind="raw")

        assert list(out.columns) == ["id", "generation", "sex", "mother", "father", "liability1", "t1", "t2"]
        assert list(out["id"]) == [2, 3]
        assert list(out["generation"]) == [0, 1]
        assert list(out["t1"]) == [10.0, 20.0]

    def test_hydrates_requested_pedigree_columns_with_id_added_first(self):
        trait = _raw_trait()
        pedigree = _pedigree()

        out = hydrate_trait(trait, pedigree, kind="raw", columns=["generation", "sex"])

        assert list(out.columns) == ["id", "generation", "sex", "t1", "t2"]

    def test_allows_extra_trait_columns_that_do_not_collide(self):
        trait = _raw_trait().with_columns(pl.Series("model_note", ["a", "b"]))
        pedigree = _pedigree()

        out = hydrate_trait(trait, pedigree, kind="raw", columns=["generation"])

        assert list(out.columns) == ["id", "generation", "t1", "t2", "model_note"]

    def test_rejects_trait_columns_that_collide_with_requested_pedigree_columns(self):
        trait = _raw_trait().with_columns(pl.Series("generation", [0, 1]))
        pedigree = _pedigree()

        with pytest.raises(ValueError, match=r"collide.*generation"):
            hydrate_trait(trait, pedigree, kind="raw", columns=["generation"])

    def test_rejects_missing_trait_id_in_pedigree(self):
        trait = pl.DataFrame({"id": [99], "t1": [1.0], "t2": [2.0]})
        pedigree = _pedigree()

        with pytest.raises(ValueError, match=r"trait ids missing from pedigree.*99"):
            hydrate_trait(trait, pedigree, kind="raw")

    def test_rejects_duplicate_trait_ids(self):
        trait = pl.DataFrame({"id": [2, 2], "t1": [1.0, 2.0], "t2": [3.0, 4.0]})
        pedigree = _pedigree()

        with pytest.raises(ValueError, match=r"trait.*duplicate id"):
            hydrate_trait(trait, pedigree, kind="raw")

    def test_rejects_duplicate_pedigree_ids(self):
        trait = _raw_trait()
        pedigree = pl.concat([_pedigree(), _pedigree().head(1)])

        with pytest.raises(ValueError, match=r"pedigree.*duplicate id"):
            hydrate_trait(trait, pedigree, kind="raw")

    def test_rejects_missing_required_outcome_column_when_validating(self):
        trait = _raw_trait().drop("t2")
        pedigree = _pedigree()

        with pytest.raises(ValueError, match=r"raw trait.*t2"):
            hydrate_trait(trait, pedigree, kind="raw")

    def test_hydrate_preserves_shuffled_trait_row_order(self):
        rng = np.random.default_rng(3)
        n = 500
        ped = pl.DataFrame(
            {
                "id": rng.permutation(n).astype(np.int32),
                "generation": np.zeros(n, dtype=np.int32),
                "liability1": rng.normal(size=n),
            }
        )
        trait_ids = rng.permutation(n)[: n // 2].astype(np.int32)
        trait = pl.DataFrame(
            {
                "id": trait_ids,
                "t1": rng.normal(size=n // 2),
                "t2": rng.normal(size=n // 2),
            }
        )

        out = hydrate_trait(trait, ped, kind="raw")

        assert out["id"].to_list() == trait_ids.tolist()
        assert out["t1"].to_list() == trait["t1"].to_list()
        lookup = dict(zip(ped["id"].to_list(), ped["liability1"].to_list(), strict=True))
        assert out["liability1"].to_list() == [lookup[i] for i in trait_ids.tolist()]

    def test_rejects_pandas_input(self):
        trait_pd = pd.DataFrame({"id": [2, 3], "t1": [10.0, 20.0], "t2": [30.0, 40.0]})

        with pytest.raises(TypeError, match=r"polars DataFrame since the polars migration"):
            hydrate_trait(trait_pd, _pedigree(), kind="raw")
        with pytest.raises(TypeError, match=r"polars DataFrame since the polars migration"):
            hydrate_trait(_raw_trait(), trait_pd, kind="raw")
