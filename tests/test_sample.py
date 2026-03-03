"""Unit tests for sim_ace.sample functions."""

import numpy as np
import pandas as pd
import pytest

from sim_ace.sample import run_sample


@pytest.fixture
def phenotype_df():
    """Create a minimal phenotype DataFrame for sampling tests."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "id": np.arange(n),
        "generation": np.repeat([3, 4, 5], [60, 70, 70]),
        "sex": rng.integers(0, 2, n),
        "liability1": rng.standard_normal(n),
        "liability2": rng.standard_normal(n),
        "t_observed1": rng.uniform(10, 80, n),
        "t_observed2": rng.uniform(10, 80, n),
        "affected1": rng.choice([True, False], n),
        "affected2": rng.choice([True, False], n),
    })


class TestPassThrough:

    def test_n_sample_zero(self, phenotype_df):
        """N_sample=0 returns all rows unchanged."""
        result = run_sample(phenotype_df, {"N_sample": 0, "seed": 42})
        pd.testing.assert_frame_equal(result, phenotype_df)

    def test_n_sample_negative(self, phenotype_df):
        """Negative N_sample returns all rows unchanged."""
        result = run_sample(phenotype_df, {"N_sample": -1, "seed": 42})
        pd.testing.assert_frame_equal(result, phenotype_df)

    def test_n_sample_greater_than_total(self, phenotype_df):
        """N_sample >= total returns all rows unchanged."""
        result = run_sample(phenotype_df, {"N_sample": 500, "seed": 42})
        pd.testing.assert_frame_equal(result, phenotype_df)

    def test_n_sample_equal_to_total(self, phenotype_df):
        """N_sample == total returns all rows unchanged."""
        result = run_sample(phenotype_df, {"N_sample": len(phenotype_df), "seed": 42})
        pd.testing.assert_frame_equal(result, phenotype_df)


class TestSampling:

    def test_exact_count(self, phenotype_df):
        """Sampled output has exactly N_sample rows."""
        result = run_sample(phenotype_df, {"N_sample": 50, "seed": 42})
        assert len(result) == 50

    def test_columns_preserved(self, phenotype_df):
        """All columns from input are present in output."""
        result = run_sample(phenotype_df, {"N_sample": 50, "seed": 42})
        assert list(result.columns) == list(phenotype_df.columns)

    def test_sampled_ids_are_subset(self, phenotype_df):
        """Sampled IDs are a subset of original IDs."""
        result = run_sample(phenotype_df, {"N_sample": 50, "seed": 42})
        assert set(result["id"].values).issubset(set(phenotype_df["id"].values))

    def test_no_duplicate_ids(self, phenotype_df):
        """Sampled output has no duplicate IDs."""
        result = run_sample(phenotype_df, {"N_sample": 50, "seed": 42})
        assert result["id"].nunique() == len(result)

    def test_preserved_order(self, phenotype_df):
        """Sampled IDs are in ascending order (sorted indices)."""
        result = run_sample(phenotype_df, {"N_sample": 50, "seed": 42})
        ids = result["id"].values
        assert np.all(ids[:-1] <= ids[1:])

    def test_reset_index(self, phenotype_df):
        """Output index is reset to 0..N_sample-1."""
        result = run_sample(phenotype_df, {"N_sample": 50, "seed": 42})
        assert list(result.index) == list(range(50))


class TestDeterminism:

    def test_same_seed_same_result(self, phenotype_df):
        """Same seed produces identical samples."""
        r1 = run_sample(phenotype_df, {"N_sample": 50, "seed": 99})
        r2 = run_sample(phenotype_df, {"N_sample": 50, "seed": 99})
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seed_different_result(self, phenotype_df):
        """Different seeds produce different samples."""
        r1 = run_sample(phenotype_df, {"N_sample": 50, "seed": 99})
        r2 = run_sample(phenotype_df, {"N_sample": 50, "seed": 100})
        assert not r1["id"].equals(r2["id"])
