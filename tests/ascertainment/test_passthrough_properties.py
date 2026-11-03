"""The ``simace ascertain`` copy fast path writes what ``run_ascertainment`` would.

``copy_passthrough_if_possible`` copies its inputs when no individual can be
dropped. Whenever it returns True, the copied files must equal
``save_parquet(run_ascertainment(...))`` on the same inputs, including the
API's severing of links that point outside the pedigree.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import polars.testing
from hypothesis import event, given
from hypothesis import strategies as st

from simace.ascertainment.runner import copy_passthrough_if_possible, run_ascertainment
from simace.core.parquet import load_parquet, save_parquet
from tests.conftest import pedigree_frame, relabel_ids


def _with_dangling_links(pedigree: pl.DataFrame, data) -> pl.DataFrame:
    """Point some links at ids absent from the pedigree, as an externally built file can."""
    if len(pedigree) == 0 or not data.draw(st.booleans(), label="dangle"):
        return pedigree
    absent = int(pedigree["id"].max()) + 1
    rows = data.draw(st.sets(st.integers(0, len(pedigree) - 1), min_size=1), label="rows")
    column = data.draw(st.sampled_from(["mother", "father", "twin"]), label="column")
    hit = pl.int_range(pl.len()).is_in(sorted(rows))
    return pedigree.with_columns(pl.when(hit).then(pl.lit(absent)).otherwise(pl.col(column)).alias(column))


@given(
    pedigree=pedigree_frame(twins=True, liabilities=True),
    data=st.data(),
    n_sample_offset=st.sampled_from([None, 0, 2]),
)
def test_fast_path_output_equals_run_ascertainment(pedigree, data, n_sample_offset):
    pedigree = _with_dangling_links(relabel_ids(pedigree, data), data)
    n_sample = 0 if n_sample_offset is None else len(pedigree) + n_sample_offset
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        save_parquet(pedigree, d / "ped.parquet")
        save_parquet(pedigree, d / "trait.parquet")
        copied = copy_passthrough_if_possible(
            d / "ped.parquet",
            d / "trait.parquet",
            d / "fast_ped.parquet",
            d / "fast_trait.parquet",
            dropout_rate=0.0,
            N_sample=n_sample,
        )
        event(f"fast path taken: {copied}")
        if not copied:
            return
        api_ped, api_trait = run_ascertainment(
            load_parquet(d / "ped.parquet"), load_parquet(d / "trait.parquet"), N_sample=n_sample
        )
        save_parquet(api_ped, d / "api_ped.parquet")
        save_parquet(api_trait, d / "api_trait.parquet")
        for name in ("ped", "trait"):
            polars.testing.assert_frame_equal(
                load_parquet(d / f"fast_{name}.parquet"), load_parquet(d / f"api_{name}.parquet")
            )
