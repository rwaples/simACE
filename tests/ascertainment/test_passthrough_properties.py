"""The ``simace ascertain`` copy fast path writes what ``run_ascertainment`` would.

``copy_passthrough_if_possible`` copies its inputs when no individual can be
dropped. Whenever it returns True, the copied files must equal
``save_parquet(run_ascertainment(...))`` on the same inputs, including the
API's severing of links that point outside the pedigree. It must also take the
copy whenever nothing would change, or the fast path silently disappears.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import polars.testing
import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from simace.ascertainment.runner import copy_passthrough_if_possible, run_ascertainment
from simace.core.parquet import load_parquet, save_parquet
from tests.conftest import pedigree_frame, relabel_ids


def _with_dangling_links(pedigree: pl.DataFrame, data) -> tuple[pl.DataFrame, bool]:
    """Point some links at ids absent from the pedigree, as an externally built file can.

    The absent id is drawn from the gaps ``relabel_ids`` leaves as well as just
    past the largest id, so an in-range hole is exercised, not only an
    out-of-range value.
    """
    if len(pedigree) == 0 or not data.draw(st.booleans(), label="dangle"):
        return pedigree, False
    ids = set(pedigree["id"].to_list())
    absent = data.draw(st.sampled_from(sorted(set(range(max(ids) + 2)) - ids)), label="absent")
    rows = data.draw(st.sets(st.integers(0, len(pedigree) - 1), min_size=1), label="rows")
    column = data.draw(st.sampled_from(["mother", "father", "twin"]), label="column")
    hit = pl.int_range(pl.len()).is_in(sorted(rows))
    dangled = pedigree.with_columns(pl.when(hit).then(pl.lit(absent)).otherwise(pl.col(column)).alias(column))
    return dangled, True


def _run_both(pedigree: pl.DataFrame, d: Path, n_sample: int) -> bool:
    """Run the fast path into ``fast_*`` and, when it copied, the API into ``api_*``; return whether it copied."""
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
    if copied:
        api_ped, api_trait = run_ascertainment(
            load_parquet(d / "ped.parquet"), load_parquet(d / "trait.parquet"), N_sample=n_sample
        )
        save_parquet(api_ped, d / "api_ped.parquet")
        save_parquet(api_trait, d / "api_trait.parquet")
    return copied


def _assert_outputs_equal(d: Path) -> None:
    for name in ("ped", "trait"):
        polars.testing.assert_frame_equal(
            load_parquet(d / f"fast_{name}.parquet"), load_parquet(d / f"api_{name}.parquet")
        )


@given(
    pedigree=pedigree_frame(twins=True, liabilities=True),
    data=st.data(),
    n_sample_offset=st.sampled_from([None, 0, 2]),
)
def test_fast_path_copies_exactly_when_the_api_is_the_identity(pedigree, data, n_sample_offset):
    pedigree, dangled = _with_dangling_links(relabel_ids(pedigree, data), data)
    n_sample = 0 if n_sample_offset is None else len(pedigree) + n_sample_offset
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        copied = _run_both(pedigree, d, n_sample)
        event(f"dangling links: {dangled}")
        assert copied == (not dangled)
        if copied:
            _assert_outputs_equal(d)


def _external_pedigree(**links) -> pl.DataFrame:
    """A three-row founder/founder/child pedigree; ``links`` overrides link columns."""
    columns = {"id": [10, 11, 12], "mother": [-1, -1, 10], "father": [-1, -1, 11], "twin": [-1, -1, -1]}
    return pl.DataFrame({**columns, **links})


def _copy(pedigree: pl.DataFrame, d: Path) -> bool:
    """The fast path on ``pedigree`` written raw (no dtype narrowing), as an external tool would."""
    pedigree.write_parquet(d / "ped.parquet")
    pedigree.select("id").write_parquet(d / "trait.parquet")
    return copy_passthrough_if_possible(
        d / "ped.parquet", d / "trait.parquet", d / "o_ped.parquet", d / "o_trait.parquet"
    )


@pytest.mark.parametrize(
    ("pedigree", "copies"),
    [
        pytest.param(_external_pedigree(), True, id="clean"),
        pytest.param(_external_pedigree(mother=[-1, -1, 16]), False, id="mother-absent-id"),
        pytest.param(_external_pedigree(twin=[-1, 13, -1]), False, id="twin-absent-id"),
        pytest.param(_external_pedigree(mother=[None, None, 10]), False, id="null-mother"),
        pytest.param(_external_pedigree().drop("father"), False, id="no-father-column"),
        pytest.param(_external_pedigree().drop("twin"), True, id="no-twin-column"),
        pytest.param(
            pl.DataFrame(
                {
                    "id": [900_000_001, 900_000_002, 900_000_003],
                    "mother": [-1, -1, 900_000_001],
                    "father": [-1, -1, 900_000_002],
                }
            ),
            True,
            id="sparse-registry-ids",
        ),
    ],
)
def test_fast_path_on_external_pedigrees(tmp_path, pedigree, copies):
    """Declines, never raises, on inputs the API would rewrite or cannot read."""
    assert _copy(pedigree, tmp_path) is copies
