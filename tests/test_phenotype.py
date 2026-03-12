"""Unit tests for sim_ace.phenotype functions."""

import numpy as np
import pytest

from sim_ace.phenotype import simulate_phenotype, phenotype_adult_ltm, phenotype_adult_cox


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


# ---------------------------------------------------------------------------
# ADuLT Liability Threshold Model tests
# ---------------------------------------------------------------------------

class TestAdultLtm:

    def test_output_shape(self):
        liability = np.random.default_rng(0).standard_normal(500)
        t = phenotype_adult_ltm(liability, prevalence=0.10, seed=42)
        assert t.shape == (500,)

    def test_all_positive_times(self):
        liability = np.random.default_rng(0).standard_normal(1000)
        t = phenotype_adult_ltm(liability, prevalence=0.10, seed=42)
        assert np.all(t > 0)

    def test_case_rate_matches_prevalence(self):
        """Fraction of cases should approximate the prevalence."""
        n = 50000
        liability = np.random.default_rng(0).standard_normal(n)
        prevalence = 0.10
        t = phenotype_adult_ltm(liability, prevalence=prevalence, seed=42)
        case_rate = np.mean(t < 1e6)
        assert abs(case_rate - prevalence) < 0.02

    def test_controls_are_large(self):
        """Controls should have t = 1e6."""
        liability = np.random.default_rng(0).standard_normal(5000)
        t = phenotype_adult_ltm(liability, prevalence=0.10, seed=42)
        controls = t[t >= 1e6 - 1]
        assert len(controls) > 0

    def test_case_ages_centered_on_x0(self):
        """Case onset ages should be centered around cip_x0."""
        n = 50000
        liability = np.random.default_rng(0).standard_normal(n)
        cip_x0 = 60.0
        t = phenotype_adult_ltm(liability, prevalence=0.20, cip_x0=cip_x0, seed=42)
        case_ages = t[t < 1e6]
        assert abs(case_ages.mean() - cip_x0) < 3.0

    def test_deterministic(self):
        """Age-of-onset is a deterministic function of liability (no randomness)."""
        liability = np.array([0.5, -0.3, 1.2, -1.0, 2.0])
        t1 = phenotype_adult_ltm(liability, prevalence=0.10, seed=42)
        t2 = phenotype_adult_ltm(liability, prevalence=0.10, seed=99)
        np.testing.assert_array_equal(t1, t2)

    def test_higher_liability_more_cases(self):
        """Individuals with higher liability should be more likely to be cases."""
        n = 10000
        rng = np.random.default_rng(99)
        liability = rng.standard_normal(n)
        t = phenotype_adult_ltm(liability, prevalence=0.10, seed=42)
        high = liability > 1.0
        low = liability < -1.0
        assert np.mean(t[high] < 1e6) > np.mean(t[low] < 1e6)

    def test_higher_liability_earlier_onset(self):
        """Among cases, higher liability should map to younger onset age."""
        n = 50000
        liability = np.random.default_rng(0).standard_normal(n)
        t = phenotype_adult_ltm(liability, prevalence=0.20, seed=42)
        cases = t < 1e6
        case_L = liability[cases]
        case_t = t[cases]
        high = case_L > np.percentile(case_L, 75)
        low = case_L < np.percentile(case_L, 25)
        assert case_t[high].mean() < case_t[low].mean()


# ---------------------------------------------------------------------------
# ADuLT Cox Model tests
# ---------------------------------------------------------------------------

class TestAdultCox:

    def test_output_shape(self):
        liability = np.random.default_rng(0).standard_normal(500)
        t = phenotype_adult_cox(liability, prevalence=0.10, seed=42)
        assert t.shape == (500,)

    def test_all_positive_times(self):
        liability = np.random.default_rng(0).standard_normal(1000)
        t = phenotype_adult_cox(liability, prevalence=0.10, seed=42)
        assert np.all(t > 0)

    def test_case_rate_matches_prevalence(self):
        """Fraction of cases should approximate the prevalence."""
        n = 50000
        liability = np.random.default_rng(0).standard_normal(n)
        prevalence = 0.10
        t = phenotype_adult_cox(liability, prevalence=prevalence, seed=42)
        case_rate = np.mean(t < 1e6)
        assert abs(case_rate - prevalence) < 0.02

    def test_case_ages_centered_on_x0(self):
        """Case onset ages should be centered around cip_x0."""
        n = 50000
        liability = np.random.default_rng(0).standard_normal(n)
        cip_x0 = 55.0
        t = phenotype_adult_cox(liability, prevalence=0.20, cip_x0=cip_x0, seed=42)
        case_ages = t[t < 1e6]
        assert abs(np.median(case_ages) - cip_x0) < 2.0

    def test_higher_liability_more_cases(self):
        """Individuals with higher liability should be more likely to be cases."""
        n = 50000
        liability = np.random.default_rng(0).standard_normal(n)
        t = phenotype_adult_cox(liability, prevalence=0.10, seed=42)
        high = liability > 1.0
        low = liability < -1.0
        assert np.mean(t[high] < 1e6) > np.mean(t[low] < 1e6)

    def test_deterministic_with_same_seed(self):
        liability = np.array([0.5, -0.3, 1.2, -1.0, 2.0])
        t1 = phenotype_adult_cox(liability, prevalence=0.10, seed=42)
        t2 = phenotype_adult_cox(liability, prevalence=0.10, seed=42)
        np.testing.assert_array_equal(t1, t2)

    def test_different_seed_changes_result(self):
        """Different seeds should produce different age assignments."""
        n = 5000
        liability = np.random.default_rng(0).standard_normal(n)
        t1 = phenotype_adult_cox(liability, prevalence=0.10, seed=42)
        t2 = phenotype_adult_cox(liability, prevalence=0.10, seed=99)
        assert not np.allclose(t1, t2)
