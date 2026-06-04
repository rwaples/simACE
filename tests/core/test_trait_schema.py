"""Tests for outcomes-only trait schemas and hydration."""

import numpy as np
import pandas as pd
import pytest

from simace.core.trait_schema import (
    TRAIT_CENSORED_COLUMNS,
    TRAIT_RAW_COLUMNS,
    TRAIT_SIMPLE_LTM_COLUMNS,
    hydrate_trait,
    strip_trait_to_outcomes,
)


def _pedigree() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.array([3, 1, 2], dtype=np.int32),
            "generation": np.array([1, 0, 0], dtype=np.int32),
            "sex": np.array([0, 1, 0], dtype=np.int8),
            "mother": np.array([1, -1, -1], dtype=np.int32),
            "father": np.array([2, -1, -1], dtype=np.int32),
            "liability1": np.array([0.3, 0.1, 0.2], dtype=np.float64),
        }
    )


def _raw_trait() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.array([2, 3], dtype=np.int32),
            "t1": np.array([10.0, 20.0], dtype=np.float32),
            "t2": np.array([30.0, 40.0], dtype=np.float32),
        }
    )


def _censored_trait() -> pd.DataFrame:
    df = _raw_trait()
    df["death_age"] = np.array([80.0, 70.0], dtype=np.float32)
    for trait in (1, 2):
        df[f"age_censored{trait}"] = np.array([False, True])
        df[f"t_observed{trait}"] = np.array([10.0, 60.0], dtype=np.float32)
        df[f"death_censored{trait}"] = np.array([False, False])
        df[f"affected{trait}"] = np.array([True, False])
    return df


def _simple_ltm_trait() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.array([2, 3], dtype=np.int32),
            "affected1": np.array([True, False]),
            "affected2": np.array([False, True]),
        }
    )


class TestStripTraitToOutcomes:
    def test_strips_raw_trait_to_ordered_schema(self):
        df = _raw_trait().assign(generation=[0, 1], extra="drop")

        out = strip_trait_to_outcomes(df, "raw")

        assert list(out.columns) == list(TRAIT_RAW_COLUMNS)
        assert list(out["id"]) == [2, 3]

    def test_strips_censored_trait_to_ordered_schema(self):
        df = _censored_trait().assign(liability1=[0.2, 0.3])

        out = strip_trait_to_outcomes(df, "censored")

        assert list(out.columns) == list(TRAIT_CENSORED_COLUMNS)

    def test_strips_simple_ltm_trait_to_ordered_schema(self):
        df = _simple_ltm_trait().assign(generation=[0, 1])

        out = strip_trait_to_outcomes(df, "simple_ltm")

        assert list(out.columns) == list(TRAIT_SIMPLE_LTM_COLUMNS)

    def test_rejects_missing_required_outcome_column(self):
        df = _raw_trait().drop(columns=["t2"])

        with pytest.raises(ValueError, match=r"missing required columns.*t2"):
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
        trait = _raw_trait().assign(model_note=["a", "b"])
        pedigree = _pedigree()

        out = hydrate_trait(trait, pedigree, kind="raw", columns=["generation"])

        assert list(out.columns) == ["id", "generation", "t1", "t2", "model_note"]

    def test_rejects_trait_columns_that_collide_with_requested_pedigree_columns(self):
        trait = _raw_trait().assign(generation=[0, 1])
        pedigree = _pedigree()

        with pytest.raises(ValueError, match=r"collide.*generation"):
            hydrate_trait(trait, pedigree, kind="raw", columns=["generation"])

    def test_rejects_missing_trait_id_in_pedigree(self):
        trait = pd.DataFrame({"id": [99], "t1": [1.0], "t2": [2.0]})
        pedigree = _pedigree()

        with pytest.raises(ValueError, match=r"trait ids missing from pedigree.*99"):
            hydrate_trait(trait, pedigree, kind="raw")

    def test_rejects_duplicate_trait_ids(self):
        trait = pd.DataFrame({"id": [2, 2], "t1": [1.0, 2.0], "t2": [3.0, 4.0]})
        pedigree = _pedigree()

        with pytest.raises(ValueError, match=r"trait.*duplicate id"):
            hydrate_trait(trait, pedigree, kind="raw")

    def test_rejects_duplicate_pedigree_ids(self):
        trait = _raw_trait()
        pedigree = pd.concat([_pedigree(), _pedigree().iloc[[0]]], ignore_index=True)

        with pytest.raises(ValueError, match=r"pedigree.*duplicate id"):
            hydrate_trait(trait, pedigree, kind="raw")

    def test_rejects_missing_required_outcome_column_when_validating(self):
        trait = _raw_trait().drop(columns=["t2"])
        pedigree = _pedigree()

        with pytest.raises(ValueError, match=r"raw trait.*t2"):
            hydrate_trait(trait, pedigree, kind="raw")
