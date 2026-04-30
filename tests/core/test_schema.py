"""Tests for the pipeline-stage schema contracts in ``simace.core.schema``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simace.core.schema import CENSORED, PEDIGREE, PHENOTYPE, assert_schema


def _make_pedigree_df() -> pd.DataFrame:
    n = 4
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int32),
            "generation": np.zeros(n, dtype=np.int32),
            "sex": np.zeros(n, dtype=np.int8),
            "mother": -np.ones(n, dtype=np.int32),
            "father": -np.ones(n, dtype=np.int32),
            "twin": -np.ones(n, dtype=np.int32),
            "household_id": np.arange(n, dtype=np.int32),
            "A1": np.zeros(n, dtype=np.float32),
            "C1": np.zeros(n, dtype=np.float32),
            "E1": np.zeros(n, dtype=np.float32),
            "liability1": np.zeros(n, dtype=np.float64),
            "A2": np.zeros(n, dtype=np.float32),
            "C2": np.zeros(n, dtype=np.float32),
            "E2": np.zeros(n, dtype=np.float32),
            "liability2": np.zeros(n, dtype=np.float64),
        }
    )


def _make_phenotype_df() -> pd.DataFrame:
    df = _make_pedigree_df()
    df["t1"] = np.ones(len(df), dtype=np.float32)
    df["t2"] = np.ones(len(df), dtype=np.float32)
    return df


def _make_censored_df() -> pd.DataFrame:
    df = _make_phenotype_df()
    n = len(df)
    df["death_age"] = np.full(n, 80.0, dtype=np.float32)
    for trait in (1, 2):
        df[f"age_censored{trait}"] = np.zeros(n, dtype=bool)
        df[f"t_observed{trait}"] = np.ones(n, dtype=np.float32)
        df[f"death_censored{trait}"] = np.zeros(n, dtype=bool)
        df[f"affected{trait}"] = np.ones(n, dtype=bool)
    return df


class TestAssertSchemaAccepts:
    def test_pedigree_passes(self):
        assert_schema(_make_pedigree_df(), PEDIGREE, where="test")

    def test_phenotype_passes(self):
        assert_schema(_make_phenotype_df(), PHENOTYPE, where="test")

    def test_censored_passes(self):
        assert_schema(_make_censored_df(), CENSORED, where="test")

    def test_extra_columns_allowed(self):
        df = _make_pedigree_df()
        df["extra"] = "ignored"
        assert_schema(df, PEDIGREE, where="test")

    def test_float64_satisfies_float_kind(self):
        df = _make_pedigree_df()
        df["A1"] = df["A1"].astype(np.float64)
        assert_schema(df, PEDIGREE, where="test")


class TestAssertSchemaRejects:
    def test_missing_column_message_names_column_and_stage(self):
        df = _make_pedigree_df().drop(columns=["liability1"])
        with pytest.raises(ValueError, match=r"phenotype input.*liability1"):
            assert_schema(df, PEDIGREE, where="phenotype input")

    def test_string_dtype_in_numeric_column_rejected(self):
        df = _make_pedigree_df()
        df["id"] = df["id"].astype(str)
        with pytest.raises(ValueError, match=r"dtype mismatch"):
            assert_schema(df, PEDIGREE, where="test")

    def test_int_in_bool_column_rejected(self):
        df = _make_censored_df()
        df["affected1"] = df["affected1"].astype(np.int8)
        with pytest.raises(ValueError, match=r"affected1"):
            assert_schema(df, CENSORED, where="test")

    def test_float_in_int_column_rejected(self):
        df = _make_pedigree_df()
        df["generation"] = df["generation"].astype(np.float64)
        with pytest.raises(ValueError, match=r"generation"):
            assert_schema(df, PEDIGREE, where="test")
