"""Tests for simace.analysis.stats.sampling.create_sample."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from simace.analysis.stats.sampling import create_sample


def _make_df(n_per_gen: int, n_gens: int = 3, seed: int = 0) -> pl.DataFrame:
    """Build a tiny pedigree DF with parent links pointing within the generation above."""
    rng = np.random.default_rng(seed)
    rows = []
    next_id = 0
    prev_gen_ids: list[int] = []
    for g in range(n_gens):
        gen_ids = list(range(next_id, next_id + n_per_gen))
        for i in gen_ids:
            if g == 0 or not prev_gen_ids:
                mother, father = -1, -1
            else:
                mother = int(rng.choice(prev_gen_ids))
                # Pick a different parent for father; OK if same here, just for plumbing
                father = int(rng.choice(prev_gen_ids))
            rows.append({"id": i, "mother": mother, "father": father, "generation": g})
        prev_gen_ids = gen_ids
        next_id += n_per_gen
    return pl.DataFrame(rows)


class TestCreateSample:
    def test_below_cap_returns_full_copy(self):
        df = _make_df(n_per_gen=10, n_gens=2)
        out = create_sample(df, n_per_gen=50_000)
        assert len(out) == len(df)
        # It's a copy, not the same object
        assert out is not df

    def test_above_cap_truncates(self):
        df = _make_df(n_per_gen=200, n_gens=2)
        out = create_sample(df, n_per_gen=50, seed=1)
        # Within cap per gen + parent-row preservation; parent rows live in earlier gens
        # so the sampled-from-G1 set is bounded by the cap.
        last_gen = df["generation"].max()
        n_last = (out["generation"] == last_gen).sum()
        assert n_last <= 50

    def test_parent_rows_preserved_for_last_gen(self):
        # Two-gen pedigree: gen-0 founders (no parents), gen-1 children.
        # create_sample adds parents of the initially-sampled rows (one pass,
        # not recursive), so a 2-gen layout exercises the invariant cleanly.
        df = _make_df(n_per_gen=200, n_gens=2, seed=2)
        out = create_sample(df, n_per_gen=20, seed=3)
        out_ids = set(out["id"].to_list())
        # All non-(-1) parent references in the output must resolve to
        # rows that are themselves in the output.
        for parent_col in ("mother", "father"):
            referenced = set(out[parent_col].to_list()) - {-1}
            missing = referenced - out_ids
            assert not missing, f"missing parent rows in output: {sorted(missing)[:5]}"

    def test_deterministic_under_same_seed(self):
        df = _make_df(n_per_gen=100, n_gens=2, seed=4)
        a = create_sample(df, n_per_gen=20, seed=42)
        b = create_sample(df, n_per_gen=20, seed=42)
        assert_frame_equal(a, b)

    def test_seed_changes_sample(self):
        df = _make_df(n_per_gen=300, n_gens=2, seed=5)
        a = create_sample(df, n_per_gen=20, seed=1)
        b = create_sample(df, n_per_gen=20, seed=2)
        # Highly unlikely to be identical given 300 candidates per gen
        assert not a["id"].equals(b["id"])

    def test_cap_per_gen_respected_when_some_gens_below(self):
        """A small early gen + large late gen still triggers downsampling."""
        df_small = _make_df(n_per_gen=5, n_gens=1, seed=6)
        df_large = pl.DataFrame(
            {
                "id": np.arange(5, 5 + 200),
                "mother": np.full(200, -1, dtype=np.int64),
                "father": np.full(200, -1, dtype=np.int64),
                "generation": np.ones(200, dtype=np.int64),
            }
        )
        df = pl.concat([df_small, df_large])
        out = create_sample(df, n_per_gen=10, seed=7)
        # gen 0 has 5 rows ≤ 10, all kept; gen 1 capped at 10
        assert (out["generation"] == 0).sum() == 5
        assert (out["generation"] == 1).sum() == 10

    def test_row_order_preserved_in_output(self):
        df = _make_df(n_per_gen=100, n_gens=2, seed=8)
        out = create_sample(df, n_per_gen=20, seed=9)
        # create_sample uses np.unique (sorted) on the row positions, so the
        # output preserves the input row order.  _make_df assigns ids equal to
        # row positions, so the output ids are monotonically non-decreasing.
        idx = out["id"].to_numpy()
        assert np.all(np.diff(idx) >= 0)

    def test_pandas_input_rejected_with_typeerror(self):
        """The public boundary is polars-only (ADR 0015 Wave 2)."""
        pdf = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "mother": [-1, -1, -1],
                "father": [-1, -1, -1],
                "generation": [0, 0, 0],
            }
        )
        with pytest.raises(TypeError, match="polars"):
            create_sample(pdf)


@pytest.mark.parametrize("n_gens", [1, 2, 4])
def test_returns_dataframe_for_various_pedigree_depths(n_gens):
    df = _make_df(n_per_gen=20, n_gens=n_gens, seed=10)
    out = create_sample(df, n_per_gen=5, seed=10)
    assert isinstance(out, pl.DataFrame)
    assert {"id", "mother", "father", "generation"}.issubset(out.columns)
