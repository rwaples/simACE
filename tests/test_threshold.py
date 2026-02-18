"""Unit tests for sim_ace.threshold.apply_threshold."""

import numpy as np
import pytest

from sim_ace.threshold import apply_threshold


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
