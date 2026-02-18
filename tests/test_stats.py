"""Unit tests for sim_ace.stats.tetrachoric_corr_se."""

import numpy as np
import pytest
from scipy.stats import norm

from sim_ace.stats import tetrachoric_corr_se, tetrachoric_corr


class TestTetrachoricCorrSE:

    def test_positive_correlation(self):
        """Two positively correlated binary variables should give r > 0."""
        rng = np.random.default_rng(42)
        n = 5000
        # Generate correlated binary via thresholding bivariate normal
        z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
        a = z[:, 0] > 0
        b = z[:, 1] > 0
        r, se = tetrachoric_corr_se(a, b)
        assert 0.3 < r < 0.7  # should be close to 0.5
        assert se > 0
        assert se < 0.1

    def test_negative_correlation(self):
        """Two negatively correlated binary variables should give r < 0."""
        rng = np.random.default_rng(42)
        n = 5000
        z = rng.multivariate_normal([0, 0], [[1, -0.5], [-0.5, 1]], size=n)
        a = z[:, 0] > 0
        b = z[:, 1] > 0
        r, se = tetrachoric_corr_se(a, b)
        assert -0.7 < r < -0.3

    def test_zero_correlation(self):
        """Independent binary variables should give r ~ 0."""
        rng = np.random.default_rng(42)
        n = 5000
        a = rng.binomial(1, 0.3, size=n).astype(bool)
        b = rng.binomial(1, 0.3, size=n).astype(bool)
        r, se = tetrachoric_corr_se(a, b)
        assert abs(r) < 0.15

    def test_high_correlation(self):
        """Highly correlated binary variables."""
        rng = np.random.default_rng(42)
        n = 5000
        z = rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=n)
        a = z[:, 0] > 0
        b = z[:, 1] > 0
        r, se = tetrachoric_corr_se(a, b)
        assert r > 0.75

    def test_all_same_returns_nan(self):
        """If one variable has zero variance, should return NaN."""
        a = np.ones(100, dtype=bool)
        b = np.array([True, False] * 50)
        r, se = tetrachoric_corr_se(a, b)
        assert np.isnan(r)
        assert np.isnan(se)

    def test_known_value(self):
        """Test against a known tetrachoric correlation scenario.

        Generate from bivariate normal with r=0.6, threshold at median (p=0.5).
        The tetrachoric correlation should recover ~0.6.
        """
        rng = np.random.default_rng(123)
        n = 20000
        r_true = 0.6
        z = rng.multivariate_normal([0, 0], [[1, r_true], [r_true, 1]], size=n)
        a = z[:, 0] > 0  # threshold at median
        b = z[:, 1] > 0
        r, se = tetrachoric_corr_se(a, b)
        assert abs(r - r_true) < 0.05

    def test_tetrachoric_corr_wrapper(self):
        """tetrachoric_corr should return just the correlation."""
        rng = np.random.default_rng(42)
        n = 1000
        z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
        a = z[:, 0] > 0
        b = z[:, 1] > 0
        r = tetrachoric_corr(a, b)
        r2, _ = tetrachoric_corr_se(a, b)
        assert r == r2

    def test_asymmetric_prevalence(self):
        """Test with very different prevalences for a and b."""
        rng = np.random.default_rng(42)
        n = 10000
        r_true = 0.5
        z = rng.multivariate_normal([0, 0], [[1, r_true], [r_true, 1]], size=n)
        # Asymmetric thresholds: a ~ 10% prevalence, b ~ 40% prevalence
        a = z[:, 0] > norm.ppf(0.9)
        b = z[:, 1] > norm.ppf(0.6)
        r, se = tetrachoric_corr_se(a, b)
        assert abs(r - r_true) < 0.1
