"""Unit tests for simace.sample functions."""

import numpy as np
import pandas as pd
import pytest

from simace.sampling.sample import run_sample


@pytest.fixture
def phenotype_df():
    """Create a minimal phenotype DataFrame for sampling tests."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "generation": np.repeat([3, 4, 5], [60, 70, 70]),
            "sex": rng.integers(0, 2, n),
            "liability1": rng.standard_normal(n),
            "liability2": rng.standard_normal(n),
            "t_observed1": rng.uniform(10, 80, n),
            "t_observed2": rng.uniform(10, 80, n),
            "affected1": rng.choice([True, False], n),
            "affected2": rng.choice([True, False], n),
        }
    )


@pytest.fixture
def phenotype_df_low_prevalence():
    """Create a phenotype DataFrame with ~10% case prevalence for ascertainment tests."""
    rng = np.random.default_rng(99)
    n = 1000
    # ~10% cases
    affected1 = rng.random(n) < 0.10
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "generation": np.repeat([3, 4, 5], [300, 350, 350]),
            "sex": rng.integers(0, 2, n),
            "liability1": rng.standard_normal(n),
            "liability2": rng.standard_normal(n),
            "t_observed1": rng.uniform(10, 80, n),
            "t_observed2": rng.uniform(10, 80, n),
            "affected1": affected1,
            "affected2": rng.choice([True, False], n),
        }
    )


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


class TestCaseAscertainment:
    def test_ratio_1_is_uniform(self, phenotype_df):
        """ratio=1.0 produces identical result to no-ratio call."""
        r1 = run_sample(phenotype_df, {"N_sample": 50, "seed": 42})
        r2 = run_sample(phenotype_df, {"N_sample": 50, "seed": 42, "case_ascertainment_ratio": 1.0})
        pd.testing.assert_frame_equal(r1, r2)

    def test_high_ratio_enriches_cases(self, phenotype_df_low_prevalence):
        """ratio=10 with ~10% prevalence yields >20% cases in sample."""
        result = run_sample(
            phenotype_df_low_prevalence,
            {"N_sample": 200, "seed": 42, "case_ascertainment_ratio": 10.0},
        )
        case_frac = result["affected1"].mean()
        # With 10% prevalence and ratio=10, expected ~53% cases in sample
        assert case_frac > 0.20, f"Expected >20% cases but got {case_frac:.1%}"

    def test_ratio_0_excludes_cases(self, phenotype_df_low_prevalence):
        """ratio=0 samples only controls."""
        result = run_sample(
            phenotype_df_low_prevalence,
            {"N_sample": 100, "seed": 42, "case_ascertainment_ratio": 0},
        )
        assert result["affected1"].sum() == 0, "Expected no cases with ratio=0"

    def test_exact_count_with_ratio(self, phenotype_df_low_prevalence):
        """Weighted sampling still returns exactly N_sample rows."""
        result = run_sample(
            phenotype_df_low_prevalence,
            {"N_sample": 200, "seed": 42, "case_ascertainment_ratio": 5.0},
        )
        assert len(result) == 200

    def test_no_duplicates_with_ratio(self, phenotype_df_low_prevalence):
        """No duplicate IDs with weighted sampling."""
        result = run_sample(
            phenotype_df_low_prevalence,
            {"N_sample": 200, "seed": 42, "case_ascertainment_ratio": 5.0},
        )
        assert result["id"].nunique() == len(result)

    def test_deterministic_with_ratio(self, phenotype_df_low_prevalence):
        """Same seed + ratio = same result."""
        params = {"N_sample": 200, "seed": 77, "case_ascertainment_ratio": 5.0}
        r1 = run_sample(phenotype_df_low_prevalence, params)
        r2 = run_sample(phenotype_df_low_prevalence, params)
        pd.testing.assert_frame_equal(r1, r2)

    def test_passthrough_warns_on_ratio(self, phenotype_df, caplog):
        """N_sample=0 with ratio!=1 logs warning, returns all."""
        import logging

        with caplog.at_level(logging.WARNING):
            result = run_sample(
                phenotype_df,
                {"N_sample": 0, "seed": 42, "case_ascertainment_ratio": 5.0},
            )
        assert len(result) == len(phenotype_df)
        assert "no effect" in caplog.text.lower()

    def test_negative_ratio_raises(self, phenotype_df):
        """Negative ratio raises ValueError."""
        with pytest.raises(ValueError, match="case_ascertainment_ratio must be >= 0"):
            run_sample(phenotype_df, {"N_sample": 50, "seed": 42, "case_ascertainment_ratio": -1.0})

    def test_all_cases_fallback(self, phenotype_df):
        """All affected → uniform fallback."""
        df = phenotype_df.copy()
        df["affected1"] = True
        result = run_sample(df, {"N_sample": 50, "seed": 42, "case_ascertainment_ratio": 5.0})
        assert len(result) == 50

    def test_no_cases_fallback(self, phenotype_df):
        """None affected → uniform fallback."""
        df = phenotype_df.copy()
        df["affected1"] = False
        result = run_sample(df, {"N_sample": 50, "seed": 42, "case_ascertainment_ratio": 5.0})
        assert len(result) == 50

    def test_ratio_0_clamps(self, phenotype_df_low_prevalence):
        """ratio=0 with N_sample > n_controls clamps to n_controls."""
        df = phenotype_df_low_prevalence
        n_controls = int((~df["affected1"]).sum())
        # Request more than available controls
        result = run_sample(
            df,
            {"N_sample": n_controls + 50, "seed": 42, "case_ascertainment_ratio": 0},
        )
        assert len(result) == n_controls
        assert result["affected1"].sum() == 0
