"""Tests for the pipeline-stage schema contracts in ``simace.core.schema``."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from simace.core.schema import CENSORED, PEDIGREE, PHENOTYPE, assert_schema


def _make_pedigree_df() -> pl.DataFrame:
    n = 4
    return pl.DataFrame(
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


def _make_phenotype_df() -> pl.DataFrame:
    df = _make_pedigree_df()
    n = len(df)
    return df.with_columns(
        pl.Series("t1", np.ones(n, dtype=np.float32)),
        pl.Series("t2", np.ones(n, dtype=np.float32)),
    )


def _make_censored_df() -> pl.DataFrame:
    df = _make_phenotype_df()
    n = len(df)
    new_cols = [pl.Series("death_age", np.full(n, 80.0, dtype=np.float32))]
    for trait in (1, 2):
        new_cols.append(pl.Series(f"age_censored{trait}", np.zeros(n, dtype=bool)))
        new_cols.append(pl.Series(f"t_observed{trait}", np.ones(n, dtype=np.float32)))
        new_cols.append(pl.Series(f"death_censored{trait}", np.zeros(n, dtype=bool)))
        new_cols.append(pl.Series(f"affected{trait}", np.ones(n, dtype=bool)))
    return df.with_columns(new_cols)


class TestAssertSchemaAccepts:
    def test_pedigree_passes(self):
        assert_schema(_make_pedigree_df(), PEDIGREE, where="test")

    def test_phenotype_passes(self):
        assert_schema(_make_phenotype_df(), PHENOTYPE, where="test")

    def test_censored_passes(self):
        assert_schema(_make_censored_df(), CENSORED, where="test")

    def test_extra_columns_allowed(self):
        df = _make_pedigree_df().with_columns(pl.lit("ignored").alias("extra"))
        assert_schema(df, PEDIGREE, where="test")

    def test_float64_satisfies_float_kind(self):
        df = _make_pedigree_df().with_columns(pl.col("A1").cast(pl.Float64))
        assert_schema(df, PEDIGREE, where="test")

    def test_nulls_do_not_change_dtype_kind(self):
        df = _make_phenotype_df().with_columns(
            pl.when(pl.col("id") == 0).then(None).otherwise(pl.col("t1")).alias("t1")
        )
        assert_schema(df, PHENOTYPE, where="test")


class TestAssertSchemaRejects:
    def test_missing_column_message_names_column_and_stage(self):
        df = _make_pedigree_df().drop("liability1")
        with pytest.raises(ValueError, match=r"phenotype input.*liability1"):
            assert_schema(df, PEDIGREE, where="phenotype input")

    def test_string_dtype_in_numeric_column_rejected(self):
        df = _make_pedigree_df().with_columns(pl.col("id").cast(pl.String))
        with pytest.raises(ValueError, match=r"dtype mismatch"):
            assert_schema(df, PEDIGREE, where="test")

    def test_int_in_bool_column_rejected(self):
        df = _make_censored_df().with_columns(pl.col("affected1").cast(pl.Int8))
        with pytest.raises(ValueError, match=r"affected1"):
            assert_schema(df, CENSORED, where="test")

    def test_float_in_int_column_rejected(self):
        df = _make_pedigree_df().with_columns(pl.col("generation").cast(pl.Float64))
        with pytest.raises(ValueError, match=r"generation"):
            assert_schema(df, PEDIGREE, where="test")

    def test_lazyframe_rejected_with_actionable_error(self):
        lf = _make_pedigree_df().lazy()
        with pytest.raises(TypeError, match=r"never LazyFrame.*collect"):
            assert_schema(lf, PEDIGREE, where="test")

    def test_pandas_rejected_with_actionable_error(self):
        df = pd.DataFrame({"id": np.arange(4, dtype=np.int32)})
        with pytest.raises(TypeError, match=r"polars DataFrames since the polars migration"):
            assert_schema(df, PEDIGREE, where="test")
