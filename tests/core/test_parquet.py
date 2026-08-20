"""Tests for simace.core.parquet.save_parquet (dtype narrowing + non-mutation)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from simace.core.parquet import load_parquet, save_parquet


def _pedigree_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": np.arange(4, dtype="int64"),
            "mother": np.array([-1, -1, 0, 0], dtype="int64"),
            "father": np.array([-1, -1, 1, 1], dtype="int64"),
            "sex": np.array([0, 1, 0, 1], dtype="int64"),
            "generation": np.zeros(4, dtype="int64"),
            "A1": np.ones(4, dtype="float64"),
            "liability1": np.ones(4, dtype="float64"),
        }
    )


def test_save_parquet_does_not_mutate_caller(tmp_path):
    df = _pedigree_frame()
    before = dict(df.schema)

    save_parquet(df, tmp_path / "out.parquet")

    # Caller's frame is untouched: dtypes are exactly as they were before
    # the write.
    assert dict(df.schema) == before
    assert df.schema["id"] == pl.Int64
    assert df.schema["A1"] == pl.Float64


def test_save_parquet_narrows_written_output(tmp_path):
    df = _pedigree_frame()
    out = tmp_path / "out.parquet"

    save_parquet(df, out)
    written = load_parquet(out)

    assert written.schema["id"] == pl.Int32
    assert written.schema["sex"] == pl.Int8
    assert written.schema["A1"] == pl.Float32
    # Liabilities keep full precision.
    assert written.schema["liability1"] == pl.Float64
    # Values round-trip unchanged.
    np.testing.assert_array_equal(written["id"].to_numpy(), df["id"].to_numpy())


def test_save_parquet_rejects_pandas(tmp_path):
    df = pd.DataFrame({"id": np.arange(4, dtype="int64")})
    with pytest.raises(TypeError, match=r"polars DataFrame since the polars migration"):
        save_parquet(df, tmp_path / "out.parquet")
