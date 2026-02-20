"""Unit tests for sim_ace.phenotype functions."""

import numpy as np
import pytest

from sim_ace.phenotype import simulate_phenotype, age_censor, death_censor


# ---------------------------------------------------------------------------
# simulate_phenotype
# ---------------------------------------------------------------------------

class TestSimulatePhenotype:

    def test_output_shape(self):
        liability = np.random.default_rng(0).standard_normal(500)
        t = simulate_phenotype(liability, beta=1.0, scale=316.228, rho=2.0, seed=42)
        assert t.shape == (500,)

    def test_all_positive_times(self):
        liability = np.random.default_rng(0).standard_normal(1000)
        t = simulate_phenotype(liability, beta=1.0, scale=316.228, rho=2.0, seed=42)
        assert np.all(t > 0)

    def test_all_finite_times(self):
        liability = np.random.default_rng(0).standard_normal(1000)
        t = simulate_phenotype(liability, beta=1.0, scale=316.228, rho=2.0, seed=42)
        assert np.all(np.isfinite(t))

    def test_higher_liability_earlier_onset(self):
        """Higher liability should produce earlier onset on average."""
        rng = np.random.default_rng(99)
        n = 10000
        liability = rng.standard_normal(n)
        t = simulate_phenotype(liability, beta=2.0, scale=464.159, rho=1.5, seed=42,
                               standardize=False)
        high = liability > 1.0
        low = liability < -1.0
        assert t[high].mean() < t[low].mean()

    def test_deterministic_with_same_seed(self):
        liability = np.array([0.5, -0.3, 1.2, -1.0])
        t1 = simulate_phenotype(liability, beta=1.0, scale=316.228, rho=2.0, seed=42)
        t2 = simulate_phenotype(liability, beta=1.0, scale=316.228, rho=2.0, seed=42)
        np.testing.assert_array_equal(t1, t2)

    def test_zero_beta_no_liability_effect(self):
        """With beta=0, frailty=1 for all, so times are independent of liability."""
        rng = np.random.default_rng(0)
        liability = np.concatenate([np.full(5000, -5.0), np.full(5000, 5.0)])
        t = simulate_phenotype(liability, beta=0.0, scale=1000.0, rho=1.0, seed=42,
                               standardize=False)
        # With beta=0, high and low liability groups should have similar means
        assert abs(t[:5000].mean() - t[5000:].mean()) / t.mean() < 0.1

    def test_standardize_centers_liability(self):
        """When standardize=True, output should not depend on liability shift."""
        liability1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        liability2 = liability1 + 100  # shifted
        t1 = simulate_phenotype(liability1, beta=1.0, scale=316.228, rho=2.0, seed=42,
                                standardize=True)
        t2 = simulate_phenotype(liability2, beta=1.0, scale=316.228, rho=2.0, seed=42,
                                standardize=True)
        np.testing.assert_allclose(t1, t2)

    # --- Validation error tests ---

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            simulate_phenotype(np.array([1.0]), beta=1.0, scale=0.0, rho=2.0, seed=42)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            simulate_phenotype(np.array([1.0]), beta=1.0, scale=-1.0, rho=2.0, seed=42)

    def test_zero_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            simulate_phenotype(np.array([1.0]), beta=1.0, scale=316.228, rho=0.0, seed=42)

    def test_inf_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            simulate_phenotype(np.array([1.0]), beta=float("inf"), scale=316.228, rho=2.0, seed=42)

    def test_nan_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            simulate_phenotype(np.array([1.0]), beta=float("nan"), scale=316.228, rho=2.0, seed=42)


# ---------------------------------------------------------------------------
# age_censor
# ---------------------------------------------------------------------------

class TestAgeCensor:

    def test_no_censoring_within_window(self):
        t = np.array([30.0, 40.0, 50.0])
        left = np.array([20.0, 20.0, 20.0])
        right = np.array([60.0, 60.0, 60.0])
        t_out, censored = age_censor(t, left, right)
        np.testing.assert_array_equal(t_out, t)
        assert not censored.any()

    def test_left_censoring(self):
        t = np.array([10.0, 5.0, 30.0])
        left = np.array([20.0, 20.0, 20.0])
        right = np.array([60.0, 60.0, 60.0])
        t_out, censored = age_censor(t, left, right)
        assert t_out[0] == 20.0
        assert t_out[1] == 20.0
        assert t_out[2] == 30.0
        assert censored[0] and censored[1]
        assert not censored[2]

    def test_right_censoring(self):
        t = np.array([70.0, 80.0, 50.0])
        left = np.array([20.0, 20.0, 20.0])
        right = np.array([60.0, 60.0, 60.0])
        t_out, censored = age_censor(t, left, right)
        assert t_out[0] == 60.0
        assert t_out[1] == 60.0
        assert t_out[2] == 50.0
        assert censored[0] and censored[1]
        assert not censored[2]

    def test_per_individual_windows(self):
        t = np.array([15.0, 55.0, 90.0])
        left = np.array([10.0, 20.0, 30.0])
        right = np.array([50.0, 60.0, 70.0])
        t_out, censored = age_censor(t, left, right)
        assert t_out[0] == 15.0 and not censored[0]
        assert t_out[1] == 55.0 and not censored[1]
        assert t_out[2] == 70.0 and censored[2]

    def test_output_shapes(self):
        n = 100
        t = np.random.default_rng(0).uniform(0, 100, n)
        left = np.full(n, 20.0)
        right = np.full(n, 80.0)
        t_out, censored = age_censor(t, left, right)
        assert t_out.shape == (n,)
        assert censored.shape == (n,)
        assert censored.dtype == bool


# ---------------------------------------------------------------------------
# death_censor
# ---------------------------------------------------------------------------

class TestDeathCensor:

    def test_output_shapes(self):
        t = np.random.default_rng(0).uniform(10, 100, 200)
        t_out, censored = death_censor(t.copy(), seed=42)
        assert t_out.shape == (200,)
        assert censored.shape == (200,)
        assert censored.dtype == bool

    def test_deterministic_with_same_seed(self):
        t = np.array([50.0, 60.0, 70.0, 80.0])
        t1, c1 = death_censor(t.copy(), seed=42)
        t2, c2 = death_censor(t.copy(), seed=42)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(c1, c2)

    def test_censored_times_are_death_ages(self):
        """For censored individuals, observed time should be <= original time."""
        rng = np.random.default_rng(0)
        t_original = rng.uniform(10, 100, 1000)
        t_copy = t_original.copy()
        t_out, censored = death_censor(t_copy, seed=42, scale=100.0, rho=5)
        # Censored individuals: observed time should be less than original
        assert np.all(t_out[censored] <= t_original[censored])

    def test_uncensored_times_unchanged(self):
        """For uncensored individuals, time should remain the same."""
        rng = np.random.default_rng(0)
        t_original = rng.uniform(10, 100, 1000)
        t_copy = t_original.copy()
        t_out, censored = death_censor(t_copy, seed=42)
        np.testing.assert_array_equal(t_out[~censored], t_original[~censored])
