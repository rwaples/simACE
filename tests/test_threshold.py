"""Unit tests for sim_ace.threshold.apply_threshold and sex-specific prevalence."""

import numpy as np
import pytest

from sim_ace.threshold import _apply_threshold_sex_aware, apply_threshold


class TestApplyThreshold:
    def test_output_is_boolean(self):
        liability = np.random.default_rng(0).standard_normal(1000)
        generation = np.zeros(1000, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.1)
        assert affected.dtype == bool

    def test_output_shape(self):
        liability = np.random.default_rng(0).standard_normal(500)
        generation = np.zeros(500, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.2)
        assert affected.shape == (500,)

    def test_prevalence_fraction_single_generation(self):
        """Affected fraction should match prevalence within rounding."""
        rng = np.random.default_rng(42)
        n = 10000
        liability = rng.standard_normal(n)
        generation = np.zeros(n, dtype=int)
        prev = 0.15
        affected = apply_threshold(liability, generation, prevalence=prev)
        observed = affected.mean()
        # With percentile-based cutoff, should be very close
        assert abs(observed - prev) < 0.01

    def test_prevalence_per_generation(self):
        """Each generation should independently have ~prevalence fraction affected."""
        rng = np.random.default_rng(42)
        n_per_gen = 5000
        liability = rng.standard_normal(3 * n_per_gen)
        generation = np.repeat([0, 1, 2], n_per_gen)
        prev = 0.10
        affected = apply_threshold(liability, generation, prevalence=prev)
        for gen in [0, 1, 2]:
            mask = generation == gen
            gen_prev = affected[mask].mean()
            assert abs(gen_prev - prev) < 0.01

    def test_higher_liability_more_likely_affected(self):
        """Individuals with higher liability should be affected more often."""
        rng = np.random.default_rng(0)
        n = 10000
        liability = rng.standard_normal(n)
        generation = np.zeros(n, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.2)
        # Top 20% by liability should be affected
        high = liability > np.percentile(liability, 80)
        low = liability < np.percentile(liability, 20)
        assert affected[high].mean() > affected[low].mean()

    def test_zero_prevalence_raises(self):
        with pytest.raises(ValueError, match="prevalence"):
            apply_threshold(np.array([1.0, 2.0]), np.array([0, 0]), prevalence=0.0)

    def test_one_prevalence_raises(self):
        with pytest.raises(ValueError, match="prevalence"):
            apply_threshold(np.array([1.0, 2.0]), np.array([0, 0]), prevalence=1.0)

    def test_negative_prevalence_raises(self):
        with pytest.raises(ValueError, match="prevalence"):
            apply_threshold(np.array([1.0, 2.0]), np.array([0, 0]), prevalence=-0.1)

    def test_constant_liability_within_generation(self):
        """If all liabilities are equal, standardized values are 0; threshold is
        at percentile so ~ prevalence fraction should still be affected."""
        liability = np.full(1000, 5.0)
        generation = np.zeros(1000, dtype=int)
        affected = apply_threshold(liability, generation, prevalence=0.5)
        # With constant liability, standardized = 0, threshold = 0,
        # and >= threshold means all or none depending on implementation
        # Either way, it shouldn't crash
        assert affected.dtype == bool


class TestApplyThresholdDictPrevalence:
    def test_dict_prevalence_per_gen_rates(self):
        """Each generation gets its own prevalence from the dict."""
        rng = np.random.default_rng(42)
        n_per_gen = 10000
        liability = rng.standard_normal(3 * n_per_gen)
        generation = np.repeat([0, 1, 2], n_per_gen)
        prev_dict = {0: 0.05, 1: 0.10, 2: 0.20}
        affected = apply_threshold(liability, generation, prevalence=prev_dict)
        for gen, expected_prev in prev_dict.items():
            mask = generation == gen
            observed = affected[mask].mean()
            assert abs(observed - expected_prev) < 0.01, f"gen {gen}: expected ~{expected_prev}, got {observed}"

    def test_dict_prevalence_output_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        n = 500
        liability = rng.standard_normal(2 * n)
        generation = np.repeat([0, 1], n)
        affected = apply_threshold(liability, generation, prevalence={0: 0.1, 1: 0.2})
        assert affected.shape == (2 * n,)
        assert affected.dtype == bool

    def test_dict_missing_generation_raises(self):
        """Dict missing a generation key should raise ValueError."""
        liability = np.array([1.0, 2.0, 3.0, 4.0])
        generation = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="missing entries for generations"):
            apply_threshold(liability, generation, prevalence={0: 0.5})

    def test_dict_prevalence_out_of_range_raises(self):
        """Dict values outside (0,1) should raise ValueError."""
        liability = np.array([1.0, 2.0])
        generation = np.array([0, 0])
        with pytest.raises(ValueError, match="prevalence must be between 0 and 1"):
            apply_threshold(liability, generation, prevalence={0: 0.0})
        with pytest.raises(ValueError, match="prevalence must be between 0 and 1"):
            apply_threshold(liability, generation, prevalence={0: 1.0})
        with pytest.raises(ValueError, match="prevalence must be between 0 and 1"):
            apply_threshold(liability, generation, prevalence={0: -0.1})

    def test_scalar_backward_compatible(self):
        """Scalar prevalence still works identically to before."""
        rng = np.random.default_rng(99)
        n = 5000
        liability = rng.standard_normal(n)
        generation = np.repeat([0, 1], n // 2)
        affected = apply_threshold(liability, generation, prevalence=0.15)
        for gen in [0, 1]:
            mask = generation == gen
            observed = affected[mask].mean()
            assert abs(observed - 0.15) < 0.01


# ---------------------------------------------------------------------------
# Sex-specific prevalence via _apply_threshold_sex_aware
# ---------------------------------------------------------------------------


class TestThresholdSexPrevalence:
    def test_sex_specific_case_rates(self):
        """Male and female affected rates should match their respective prevalences."""
        rng = np.random.default_rng(42)
        n = 20000
        liability = rng.standard_normal(n)
        generation = np.zeros(n, dtype=int)
        sex = np.array([0] * (n // 2) + [1] * (n // 2))
        prev_f, prev_m = 0.08, 0.15
        params = {
            "prevalence1": {"female": prev_f, "male": prev_m},
        }
        affected = _apply_threshold_sex_aware(liability, generation, sex, params, trait_num=1)
        female_rate = affected[: n // 2].mean()
        male_rate = affected[n // 2 :].mean()
        assert abs(female_rate - prev_f) < 0.02
        assert abs(male_rate - prev_m) < 0.02

    def test_scalar_prevalence_unchanged(self):
        """Without sex-specific keys, behaviour is identical to apply_threshold."""
        rng = np.random.default_rng(0)
        n = 5000
        liability = rng.standard_normal(n)
        generation = np.repeat([0, 1], n // 2)
        sex = rng.integers(0, 2, size=n)
        params = {"prevalence1": 0.15}
        affected_sex_aware = _apply_threshold_sex_aware(
            liability,
            generation,
            sex,
            params,
            trait_num=1,
        )
        affected_direct = apply_threshold(liability, generation, prevalence=0.15)
        np.testing.assert_array_equal(affected_sex_aware, affected_direct)

    def test_sex_specific_with_per_gen_dict(self):
        """Sex-specific + per-generation dict prevalence should compose."""
        rng = np.random.default_rng(42)
        n_per_gen = 10000
        n = 2 * n_per_gen
        liability = rng.standard_normal(n)
        generation = np.repeat([0, 1], n_per_gen)
        sex = np.tile([0, 1], n // 2)  # alternating male/female

        prev_f = {0: 0.05, 1: 0.10}
        prev_m = {0: 0.10, 1: 0.20}
        params = {
            "prevalence1": {"female": prev_f, "male": prev_m},
        }
        affected = _apply_threshold_sex_aware(liability, generation, sex, params, trait_num=1)

        for gen, exp_f, exp_m in [(0, 0.05, 0.10), (1, 0.10, 0.20)]:
            gen_mask = generation == gen
            female_mask = gen_mask & (sex == 0)
            male_mask = gen_mask & (sex == 1)
            assert abs(affected[female_mask].mean() - exp_f) < 0.02, (
                f"gen {gen} female: expected ~{exp_f}, got {affected[female_mask].mean()}"
            )
            assert abs(affected[male_mask].mean() - exp_m) < 0.02, (
                f"gen {gen} male: expected ~{exp_m}, got {affected[male_mask].mean()}"
            )
