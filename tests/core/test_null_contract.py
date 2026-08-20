"""Regression tests for the on-disk null contract (ADR 0015, decision 4c).

Missing values are parquet **null** on disk — never literal NaN — regardless of
how missingness reaches ``save_parquet`` (polars null or transient NaN from a
NumPy compute boundary). These tests inspect the written file through pyarrow
because a pandas round-trip re-conflates null and NaN on read and therefore
cannot catch a contract regression (the exact gap ADR 0014 documented).

Cross-library compat is retained here: pandas-written files (pandas' own
``to_parquet``) must load as null-carrying polars frames.

The EPIMIGHT emitter's int8/int16 schema freeze is covered where that emitter
lives (fitACE_epimight, Wave 2); the narrow-integer mechanics it relies on are
covered here via the writer's int32/int8 mappings.
"""

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from simace.core.parquet import load_parquet, save_parquet

_STRUCTURAL_INT_COLS = ["id", "mother", "father", "twin", "household_id", "generation"]


def _pedigree_data(n: int = 6) -> dict[str, np.ndarray]:
    return {
        "id": np.arange(n, dtype="int64"),
        "mother": np.array([-1, -1, 0, 0, 2, 2], dtype="int64")[:n],
        "father": np.array([-1, -1, 1, 1, 3, 3], dtype="int64")[:n],
        "twin": -np.ones(n, dtype="int64"),
        "household_id": np.arange(n, dtype="int64"),
        "generation": np.zeros(n, dtype="int64"),
        "sex": (np.arange(n) % 2).astype("int64"),
        "A1": np.linspace(0.0, 1.0, n),
        "liability1": np.linspace(-1.0, 1.0, n),
    }


def _pedigree_polars(n: int = 6) -> pl.DataFrame:
    return pl.DataFrame(_pedigree_data(n))


def _trait_data(n: int = 6) -> dict[str, np.ndarray]:
    t1 = np.linspace(20.0, 70.0, n)
    t1[1] = np.nan  # missing onset — must land as parquet null
    t2 = np.full(n, np.nan)  # entirely-missing column
    return {"id": np.arange(n, dtype="int64"), "t1": t1, "t2": t2}


def _trait_polars(n: int = 6) -> pl.DataFrame:
    # numpy-backed construction keeps NaN as NaN (no null conversion), so the
    # writer's NaN-to-null normalization is what these tests exercise
    return pl.DataFrame(_trait_data(n))


class TestOnDiskNullMask:
    def test_nan_written_as_parquet_null(self, tmp_path):
        out = tmp_path / "trait.parquet"
        save_parquet(_trait_polars(), out)
        tbl = pq.read_table(out)
        assert tbl.column("t1").null_count == 1
        assert tbl.column("t2").null_count == len(tbl)
        for name in ("t1", "t2"):
            vals = tbl.column(name).to_numpy(zero_copy_only=False)
            nulls = tbl.column(name).null_count
            assert int(np.isnan(vals).sum()) == nulls, f"{name}: literal NaN on disk"

    def test_polars_nan_normalized_to_null_at_write_edge(self, tmp_path):
        out = tmp_path / "trait.parquet"
        df = pl.DataFrame({"id": [0, 1, 2], "t1": [1.0, float("nan"), None], "liability1": [0.5, float("nan"), 1.5]})
        save_parquet(df, out)
        tbl = pq.read_table(out)
        assert tbl.column("t1").null_count == 2  # NaN and null both land as null
        assert tbl.column("liability1").null_count == 1
        for name in ("t1", "liability1"):
            vals = tbl.column(name).to_numpy(zero_copy_only=False)
            assert int(np.isnan(vals).sum()) == tbl.column(name).null_count

    def test_polars_nulls_preserved(self, tmp_path):
        out = tmp_path / "t.parquet"
        save_parquet(pl.DataFrame({"id": [0, 1], "t1": [None, 30.0]}), out)
        assert pq.read_table(out).column("t1").null_count == 1

    def test_structural_columns_have_no_nulls_and_frozen_dtypes(self, tmp_path):
        out = tmp_path / "pedigree.parquet"
        save_parquet(_pedigree_polars(), out)
        tbl = pq.read_table(out)
        for name in _STRUCTURAL_INT_COLS:
            col = tbl.column(name)
            assert col.null_count == 0, f"{name}: structural column carries nulls"
            assert str(tbl.schema.field(name).type) == "int32"
        assert str(tbl.schema.field("sex").type) == "int8"
        assert tbl.column("sex").null_count == 0
        assert str(tbl.schema.field("A1").type) == "float"
        assert str(tbl.schema.field("liability1").type) == "double"


class TestCrossLibraryRoundTrip:
    def test_pandas_write_polars_read(self, tmp_path):
        # pandas' own writer (the historical producer): NaN lands as null on
        # disk, and loads back as a null-carrying polars frame
        out = tmp_path / "trait.parquet"
        pd.DataFrame(_trait_data()).to_parquet(out)
        back = load_parquet(out)
        assert isinstance(back, pl.DataFrame)
        assert back["t1"].null_count() == 1
        assert back["t1"].is_nan().sum() == 0  # nulls, never NaN, in-frame
        assert back["t2"].null_count() == len(back)

    def test_polars_write_pandas_read(self, tmp_path):
        out = tmp_path / "trait.parquet"
        save_parquet(pl.DataFrame({"id": [0, 1], "t1": [None, 30.0]}), out)
        back = pd.read_parquet(out)
        # pandas re-conflates null as NaN — the transitional consumers' view
        assert np.isnan(back["t1"].to_numpy()[0])
        assert back["t1"].to_numpy()[1] == np.float32(30.0)

    def test_same_null_mask_whether_missing_arrives_as_nan_or_null(self, tmp_path):
        nan_out, null_out = tmp_path / "a.parquet", tmp_path / "b.parquet"
        pdf = pd.DataFrame(_trait_data())
        save_parquet(_trait_polars(), nan_out)  # NaN-carrying polars frame
        save_parquet(pl.from_pandas(pdf), null_out)  # from_pandas converts NaN to null
        a, b = pq.read_table(nan_out), pq.read_table(null_out)
        assert a.schema == b.schema
        for name in a.schema.names:
            assert a.column(name).null_count == b.column(name).null_count


class TestNarrowingAndOverflow:
    def test_float32_rounding_is_expected(self, tmp_path):
        out = tmp_path / "f.parquet"
        precise = 0.1234567890123456789
        save_parquet(pl.DataFrame({"id": [0], "A1": [precise], "liability1": [precise]}), out)
        back = load_parquet(out)
        assert back["A1"][0] == np.float32(precise)  # narrowed
        assert back["liability1"][0] == precise  # full precision

    @pytest.mark.parametrize(
        ("col", "bad"),
        [("id", 2_147_483_648), ("mother", -2_147_483_649), ("sex", 128)],
    )
    def test_integer_overflow_raises_instead_of_wrapping(self, tmp_path, col, bad):
        data = _pedigree_data(2)
        data[col][0] = bad
        with pytest.raises(ValueError, match=col):
            save_parquet(pl.DataFrame(data), tmp_path / "o.parquet")

    def test_integer_overflow_raises_for_polars_input(self, tmp_path):
        df = pl.DataFrame({"id": [0, 2_147_483_648], "t1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="id"):
            save_parquet(df, tmp_path / "o.parquet")


class TestCallerNotMutated:
    def test_polars_caller_keeps_nan_and_dtypes(self, tmp_path):
        df = pl.DataFrame({"id": [0, 1], "t1": [float("nan"), 30.0]})
        save_parquet(df, tmp_path / "t.parquet")
        assert df["id"].dtype == pl.Int64  # not narrowed in place
        assert df["t1"].is_nan().sum() == 1  # NaN not filled in place


class TestLoadParquet:
    def test_returns_eager_polars_frame(self, tmp_path):
        out = tmp_path / "p.parquet"
        save_parquet(_pedigree_polars(), out)
        back = load_parquet(out)
        assert isinstance(back, pl.DataFrame)
        assert back.height == 6

    def test_column_subset(self, tmp_path):
        out = tmp_path / "p.parquet"
        save_parquet(_pedigree_polars(), out)
        back = load_parquet(out, columns=["id", "generation"])
        assert back.columns == ["id", "generation"]
