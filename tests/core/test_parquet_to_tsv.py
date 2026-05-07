"""Round-trip tests for simace.core.parquet_to_tsv."""

import gzip
import sys

import numpy as np
import pandas as pd
import pytest

from simace.core.parquet_to_tsv import cli, convert


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": np.arange(5, dtype=np.int64),
            "x": np.array([0.123456789, 1.0, -0.5, 1e-3, 99.9], dtype=np.float64),
            "flag": np.array([True, False, True, False, True]),
            "name": ["a", "b", "c", "d", "e"],
        }
    )


def test_round_trip_gzipped(tmp_path, sample_df):
    pq = tmp_path / "in.parquet"
    sample_df.to_parquet(pq)

    convert(str(pq))

    out = tmp_path / "in.tsv.gz"
    assert out.exists()
    # File is actually gzip-compressed
    with gzip.open(out, "rt") as fh:
        first = fh.readline()
    assert first.startswith("id\tx\tflag\tname")

    back = pd.read_csv(out, sep="\t", compression="gzip")
    assert list(back.columns) == ["id", "x", "flag", "name"]
    assert back["id"].tolist() == sample_df["id"].tolist()
    # Default precision=4
    assert back["x"].iloc[0] == pytest.approx(0.1235, abs=1e-9)


def test_round_trip_uncompressed(tmp_path, sample_df):
    pq = tmp_path / "in.parquet"
    sample_df.to_parquet(pq)

    convert(str(pq), gzip=False)

    out = tmp_path / "in.tsv"
    assert out.exists()
    back = pd.read_csv(out, sep="\t")
    assert back["id"].tolist() == sample_df["id"].tolist()


def test_explicit_output_path(tmp_path, sample_df):
    pq = tmp_path / "in.parquet"
    sample_df.to_parquet(pq)

    out = tmp_path / "custom.tsv"
    convert(str(pq), output_path=str(out), gzip=False)

    assert out.exists()
    back = pd.read_csv(out, sep="\t")
    assert len(back) == len(sample_df)


def test_precision_setting(tmp_path, sample_df):
    pq = tmp_path / "in.parquet"
    sample_df.to_parquet(pq)

    out = tmp_path / "p7.tsv"
    convert(str(pq), output_path=str(out), float_precision=7, gzip=False)
    back = pd.read_csv(out, sep="\t")
    # 7 decimal places preserves .123456789 → .1234568
    assert back["x"].iloc[0] == pytest.approx(0.1234568, abs=1e-9)


def test_cli_writes_default_output(tmp_path, sample_df, monkeypatch):
    pq = tmp_path / "in.parquet"
    sample_df.to_parquet(pq)
    monkeypatch.setattr(sys, "argv", ["parquet-to-tsv", str(pq)])
    cli()
    assert (tmp_path / "in.tsv.gz").exists()


def test_cli_rejects_output_with_multiple_inputs(tmp_path, sample_df, monkeypatch):
    pq1 = tmp_path / "a.parquet"
    pq2 = tmp_path / "b.parquet"
    sample_df.to_parquet(pq1)
    sample_df.to_parquet(pq2)
    monkeypatch.setattr(sys, "argv", ["parquet-to-tsv", str(pq1), str(pq2), "-o", str(tmp_path / "out.tsv")])
    # argparse calls parser.error → SystemExit
    with pytest.raises(SystemExit):
        cli()


def test_cli_no_gzip_flag(tmp_path, sample_df, monkeypatch):
    pq = tmp_path / "in.parquet"
    sample_df.to_parquet(pq)
    monkeypatch.setattr(sys, "argv", ["parquet-to-tsv", str(pq), "--no-gzip"])
    cli()
    assert (tmp_path / "in.tsv").exists()
    assert not (tmp_path / "in.tsv.gz").exists()
