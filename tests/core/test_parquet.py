"""Tests for simace.core.parquet.save_parquet (dtype narrowing + non-mutation)."""

import numpy as np
import pandas as pd

from simace.core.parquet import save_parquet


def _pedigree_frame() -> pd.DataFrame:
    return pd.DataFrame(
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
    before = df.dtypes.copy()

    save_parquet(df, tmp_path / "out.parquet")

    # Caller's frame is untouched: dtypes (and the object identity of columns)
    # are exactly as they were before the write.
    pd.testing.assert_series_equal(df.dtypes, before)
    assert df["id"].dtype == np.int64
    assert df["A1"].dtype == np.float64


def test_save_parquet_narrows_written_output(tmp_path):
    df = _pedigree_frame()
    out = tmp_path / "out.parquet"

    save_parquet(df, out)
    written = pd.read_parquet(out)

    assert written["id"].dtype == np.int32
    assert written["sex"].dtype == np.int8
    assert written["A1"].dtype == np.float32
    # Liabilities keep full precision.
    assert written["liability1"].dtype == np.float64
    # Values round-trip unchanged.
    np.testing.assert_array_equal(written["id"].to_numpy(), df["id"].to_numpy())
