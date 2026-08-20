"""Property-based round-trip test for the pedigree parquet writer.

``load_parquet(save_parquet(df))`` must preserve the column set and row count,
reproduce values modulo the documented int32/int8/float32 narrowing (with
float64 liabilities value-exact, NaN normalized to null per the ADR 0015 null
contract), still satisfy the ``PEDIGREE`` schema contract, and leave the
caller's frame unmutated. This ties the writer's narrowing to the schema
validator that every downstream stage relies on — a space a single fixed
fixture cannot sweep (int32 boundary, NaN/inf in narrowed floats, the
float64/float32 precision split).
"""

import io

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.core.parquet import load_parquet, save_parquet
from simace.core.schema import PEDIGREE, assert_schema

_INT32_MAX = 2_147_483_647

# int columns hold ids/sentinels within the documented int32 domain
_INT_COLS = ["id", "mother", "father", "twin", "household_id", "generation"]
# narrowed to float32 by the writer
_F32_COLS = ["A1", "C1", "E1", "A2", "C2", "E2"]
# kept at full float64 precision
_F64_COLS = ["liability1", "liability2"]

_int_elems = st.integers(min_value=-1, max_value=_INT32_MAX)
_float_elems = st.floats(allow_nan=True, allow_infinity=True)


@st.composite
def _pedigree_frame(draw):
    n = draw(st.integers(min_value=1, max_value=30))
    data: dict[str, np.ndarray] = {}
    for c in _INT_COLS:
        data[c] = draw(hnp.arrays(np.int64, n, elements=_int_elems))
    data["sex"] = draw(hnp.arrays(np.int8, n, elements=st.integers(0, 1)))
    for c in _F32_COLS:
        data[c] = draw(hnp.arrays(np.float64, n, elements=_float_elems))
    for c in _F64_COLS:
        data[c] = draw(hnp.arrays(np.float64, n, elements=_float_elems))
    return pl.DataFrame(data)


@settings(deadline=None, max_examples=75)
@given(_pedigree_frame())
def test_parquet_roundtrip_modulo_narrowing(df):
    orig_schema = dict(df.schema)

    buf = io.BytesIO()
    save_parquet(df, buf)
    buf.seek(0)
    back = load_parquet(buf)

    # structure preserved
    assert set(back.columns) == set(df.columns)
    assert len(back) == len(df)

    # integer ids/sentinels are exact within the int32 domain
    for c in [*_INT_COLS, "sex"]:
        np.testing.assert_array_equal(back[c].to_numpy(), df[c].to_numpy())

    # float32-narrowed columns equal a float32 cast of the original
    # (NaN is normalized to null on disk; to_numpy materializes null as NaN)
    for c in _F32_COLS:
        assert np.array_equal(
            back[c].to_numpy(),
            df[c].to_numpy().astype(np.float32),
            equal_nan=True,
        )

    # float64 liabilities round-trip value-exact, including inf; NaN comes
    # back as null (ADR 0015), which to_numpy materializes as NaN again
    for c in _F64_COLS:
        assert np.array_equal(back[c].to_numpy(), df[c].to_numpy(), equal_nan=True)
        # the null contract: every NaN in the input is null after read-back
        assert back[c].is_null().sum() == int(np.isnan(df[c].to_numpy()).sum())
        assert not back[c].is_nan().fill_null(False).any()

    # the narrowed read-back still honors the schema contract
    assert_schema(back, PEDIGREE, where="parquet roundtrip")

    # the caller's frame was not mutated
    assert dict(df.schema) == orig_schema
