"""The gene-drop augmented pedigree keeps its input's results-layout marker (ADR 0021)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from simace.core.cohort import read_pedigree, write_pedigree
from simace.core.parquet import save_parquet

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gene_drop" / "tstrait_augment_pedigree.py"


@pytest.fixture(scope="module")
def augment():
    spec = importlib.util.spec_from_file_location("simace_tstrait_augment_layout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pedigree() -> pl.DataFrame:
    return pl.DataFrame({"id": [0, 1, 2], "mother": [-1, -1, 0], "father": [-1, -1, 1], "A1": [0.1, -0.2, 0.3]})


def test_marked_input_gives_a_marked_output(augment, tmp_path):
    source, out = tmp_path / "pedigree.parquet", tmp_path / "pedigree.full.tstrait.parquet"
    write_pedigree(_pedigree(), source)

    augment.write_like(pd.read_parquet(source), out, source)

    assert read_pedigree(out).equals(read_pedigree(source))


def test_unmarked_input_gives_an_unmarked_output(augment, tmp_path):
    source, out = tmp_path / "pedigree.parquet", tmp_path / "pedigree.full.tstrait.parquet"
    save_parquet(_pedigree(), source)

    augment.write_like(pd.read_parquet(source), out, source)

    with pytest.raises(ValueError, match="simace_layout"):
        read_pedigree(out)
