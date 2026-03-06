"""Unit tests for sim_ace.phenotype functions."""

import numpy as np
import pytest

from sim_ace.phenotype import simulate_phenotype


# ---------------------------------------------------------------------------
# Default Weibull params for tests
# ---------------------------------------------------------------------------
WEIBULL_PARAMS = {"scale": 316.228, "rho": 2.0}


# ---------------------------------------------------------------------------
# simulate_phenotype
# ---------------------------------------------------------------------------

class TestSimulatePhenotype:

    def test_output_shape(self):
        liability = np.random.default_rng(0).standard_normal(500)
        t = simulate_phenotype(liability, beta=1.0, hazard_model="weibull",
                               hazard_params=WEIBULL_PARAMS, seed=42)
        assert t.shape == (500,)

    def test_all_positive_times(self):
        liability = np.random.default_rng(0).standard_normal(1000)
        t = simulate_phenotype(liability, beta=1.0, hazard_model="weibull",
                               hazard_params=WEIBULL_PARAMS, seed=42)
        assert np.all(t > 0)

    def test_all_finite_times(self):
        liability = np.random.default_rng(0).standard_normal(1000)
        t = simulate_phenotype(liability, beta=1.0, hazard_model="weibull",
                               hazard_params=WEIBULL_PARAMS, seed=42)
        assert np.all(np.isfinite(t))

    def test_higher_liability_earlier_onset(self):
        """Higher liability should produce earlier onset on average."""
        rng = np.random.default_rng(99)
        n = 10000
        liability = rng.standard_normal(n)
        t = simulate_phenotype(liability, beta=2.0, hazard_model="weibull",
                               hazard_params={"scale": 464.159, "rho": 1.5},
                               seed=42, standardize=False)
        high = liability > 1.0
        low = liability < -1.0
        assert t[high].mean() < t[low].mean()

    def test_deterministic_with_same_seed(self):
        liability = np.array([0.5, -0.3, 1.2, -1.0])
        t1 = simulate_phenotype(liability, beta=1.0, hazard_model="weibull",
                                hazard_params=WEIBULL_PARAMS, seed=42)
        t2 = simulate_phenotype(liability, beta=1.0, hazard_model="weibull",
                                hazard_params=WEIBULL_PARAMS, seed=42)
        np.testing.assert_array_equal(t1, t2)

    def test_zero_beta_no_liability_effect(self):
        """With beta=0, frailty=1 for all, so times are independent of liability."""
        rng = np.random.default_rng(0)
        liability = np.concatenate([np.full(5000, -5.0), np.full(5000, 5.0)])
        t = simulate_phenotype(liability, beta=0.0, hazard_model="weibull",
                               hazard_params={"scale": 1000.0, "rho": 1.0},
                               seed=42, standardize=False)
        # With beta=0, high and low liability groups should have similar means
        assert abs(t[:5000].mean() - t[5000:].mean()) / t.mean() < 0.1

    def test_standardize_centers_liability(self):
        """When standardize=True, output should not depend on liability shift."""
        liability1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        liability2 = liability1 + 100  # shifted
        t1 = simulate_phenotype(liability1, beta=1.0, hazard_model="weibull",
                                hazard_params=WEIBULL_PARAMS, seed=42,
                                standardize=True)
        t2 = simulate_phenotype(liability2, beta=1.0, hazard_model="weibull",
                                hazard_params=WEIBULL_PARAMS, seed=42,
                                standardize=True)
        np.testing.assert_allclose(t1, t2)

    # --- Validation error tests ---

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown hazard model"):
            simulate_phenotype(np.array([1.0]), beta=1.0,
                               hazard_model="unknown",
                               hazard_params=WEIBULL_PARAMS, seed=42)

    def test_missing_scale_raises(self):
        with pytest.raises(KeyError):
            simulate_phenotype(np.array([1.0]), beta=1.0,
                               hazard_model="weibull",
                               hazard_params={"rho": 2.0}, seed=42)

    def test_missing_rho_raises(self):
        with pytest.raises(KeyError):
            simulate_phenotype(np.array([1.0]), beta=1.0,
                               hazard_model="weibull",
                               hazard_params={"scale": 316.228}, seed=42)

    def test_inf_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            simulate_phenotype(np.array([1.0]), beta=float("inf"),
                               hazard_model="weibull",
                               hazard_params=WEIBULL_PARAMS, seed=42)

    def test_nan_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            simulate_phenotype(np.array([1.0]), beta=float("nan"),
                               hazard_model="weibull",
                               hazard_params=WEIBULL_PARAMS, seed=42)


# ---------------------------------------------------------------------------
# Sex covariate tests
# ---------------------------------------------------------------------------

class TestBetaSex:

    def test_beta_sex_zero_same_as_no_sex(self):
        """beta_sex=0 should produce identical results to omitting sex."""
        rng = np.random.default_rng(0)
        liability = rng.standard_normal(500)
        sex = rng.integers(0, 2, size=500).astype(float)

        t_no_sex = simulate_phenotype(liability, beta=1.0, hazard_model="weibull",
                                      hazard_params=WEIBULL_PARAMS, seed=42)
        t_zero = simulate_phenotype(liability, beta=1.0, hazard_model="weibull",
                                    hazard_params=WEIBULL_PARAMS, seed=42,
                                    sex=sex, beta_sex=0.0)
        np.testing.assert_array_equal(t_no_sex, t_zero)

    def test_positive_beta_sex_males_earlier(self):
        """beta_sex > 0 should make males (sex=1) have earlier onset."""
        n = 10000
        liability = np.zeros(n)
        sex = np.array([0.0] * (n // 2) + [1.0] * (n // 2))

        t = simulate_phenotype(liability, beta=0.0, hazard_model="weibull",
                               hazard_params=WEIBULL_PARAMS, seed=42,
                               standardize=False, sex=sex, beta_sex=0.5)
        female_mean = t[:n // 2].mean()
        male_mean = t[n // 2:].mean()
        assert male_mean < female_mean

    def test_negative_beta_sex_females_earlier(self):
        """beta_sex < 0 should make females (sex=0) have earlier onset (males delayed)."""
        n = 10000
        liability = np.zeros(n)
        sex = np.array([0.0] * (n // 2) + [1.0] * (n // 2))

        t = simulate_phenotype(liability, beta=0.0, hazard_model="weibull",
                               hazard_params=WEIBULL_PARAMS, seed=42,
                               standardize=False, sex=sex, beta_sex=-0.5)
        female_mean = t[:n // 2].mean()
        male_mean = t[n // 2:].mean()
        assert female_mean < male_mean
