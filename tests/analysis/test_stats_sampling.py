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

    def test_pandas_input_rejected_with_typeerror(self):
        """The public boundary is polars-only (ADR 0015)."""
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
