"""Tests for pairwise Weibull frailty liability correlation estimator."""

import numpy as np
import pandas as pd

from sim_ace.analysis.survival_stats import compute_weibull_pair_corr
from sim_ace.core.weibull_mle import pairwise_weibull_corr_se


class TestPairwiseWeibullCorrSE:
    """Tests for the core estimator."""

    def _simulate_pair_data(self, r, n, scale, rho, beta, seed=42, censor_age=None):
        """Generate paired survival data from known BVN liabilities."""
        rng = np.random.default_rng(seed)
        cov = [[1, r], [r, 1]]
        z = rng.multivariate_normal([0, 0], cov, size=n)
        frailty_i = np.exp(beta * z[:, 0])
        frailty_j = np.exp(beta * z[:, 1])
        u_i = 1.0 - rng.uniform(size=n)
        u_j = 1.0 - rng.uniform(size=n)
        t_i = scale * ((-np.log(u_i)) / frailty_i) ** (1 / rho)
        t_j = scale * ((-np.log(u_j)) / frailty_j) ** (1 / rho)

        if censor_age is not None:
            delta_i = (t_i <= censor_age).astype(float)
            delta_j = (t_j <= censor_age).astype(float)
            t_i = np.minimum(t_i, censor_age)
            t_j = np.minimum(t_j, censor_age)
        else:
            delta_i = np.ones(n)
            delta_j = np.ones(n)

        return t_i, delta_i, t_j, delta_j

    def test_positive_correlation(self):
        """BVN with r=0.5, no censoring: should recover r within tolerance."""
        scale, rho, beta = 31.623, 2.0, 1.0
        t_i, d_i, t_j, d_j = self._simulate_pair_data(r=0.5, n=3000, scale=scale, rho=rho, beta=beta, seed=42)
        r_hat, se = pairwise_weibull_corr_se(t_i, d_i, t_j, d_j, scale, rho, beta)
        assert abs(r_hat - 0.5) < 0.1, f"r_hat={r_hat}, expected ~0.5"
        assert se > 0
        assert se < 0.1

    def test_zero_correlation(self):
        """Independent liabilities: r should be near zero."""
        scale, rho, beta = 31.623, 2.0, 1.0
        t_i, d_i, t_j, d_j = self._simulate_pair_data(r=0.0, n=2000, scale=scale, rho=rho, beta=beta, seed=99)
        r_hat, _se = pairwise_weibull_corr_se(t_i, d_i, t_j, d_j, scale, rho, beta)
        assert abs(r_hat) < 0.12, f"r_hat={r_hat}, expected ~0.0"

    def test_high_correlation(self):
        """BVN with r=0.8: should recover high correlation."""
        scale, rho, beta = 31.623, 2.0, 1.0
        t_i, d_i, t_j, d_j = self._simulate_pair_data(r=0.8, n=2000, scale=scale, rho=rho, beta=beta, seed=77)
        r_hat, se = pairwise_weibull_corr_se(t_i, d_i, t_j, d_j, scale, rho, beta)
        assert r_hat > 0.65, f"r_hat={r_hat}, expected > 0.65"
        assert se > 0

    def test_se_positive_and_finite(self):
        """SE should be positive and finite for reasonable data."""
        scale, rho, beta = 31.623, 2.0, 1.0
        t_i, d_i, t_j, d_j = self._simulate_pair_data(r=0.5, n=1000, scale=scale, rho=rho, beta=beta, seed=55)
        _r_hat, se = pairwise_weibull_corr_se(t_i, d_i, t_j, d_j, scale, rho, beta)
        assert np.isfinite(se)
        assert se > 0

    def test_with_censoring(self):
        """With censoring, should still give reasonable estimate."""
        scale, rho, beta = 31.623, 2.0, 1.0
        t_i, d_i, t_j, d_j = self._simulate_pair_data(
            r=0.5, n=3000, scale=scale, rho=rho, beta=beta, seed=42, censor_age=50
        )
        # Verify some censoring actually occurred
        frac_censored = 1 - (d_i.mean() + d_j.mean()) / 2
        assert frac_censored > 0.05, "Test setup: need some censoring"

        r_hat, _se = pairwise_weibull_corr_se(t_i, d_i, t_j, d_j, scale, rho, beta)
        assert abs(r_hat - 0.5) < 0.15, f"r_hat={r_hat}, expected ~0.5 (with censoring)"

    def test_too_few_pairs_returns_nan(self):
        """Fewer than 10 pairs should return (nan, nan)."""
        scale, rho, beta = 31.623, 2.0, 1.0
        r_hat, se = pairwise_weibull_corr_se(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([1.0, 0.0, 1.0]),
            scale,
            rho,
            beta,
        )
        assert np.isnan(r_hat)
        assert np.isnan(se)


class TestComputeWeibullPairCorr:
    """Tests for the wrapper that computes across relationship types."""

    def test_output_structure(self):
        """Verify output dict has correct structure for all pair types."""
        rng = np.random.default_rng(42)
        n = 200
        # Build a minimal DataFrame with required columns
        df = pd.DataFrame(
            {
                "id": np.arange(n),
                "mother": np.concatenate([[-1] * 50, rng.integers(0, 50, n - 50)]),
                "father": np.concatenate([[-1] * 50, rng.integers(0, 50, n - 50)]),
                "twin": np.full(n, -1),
                "t_observed1": rng.exponential(50, n),
                "affected1": rng.choice([True, False], n),
            }
        )

        # Use a simple pairs dict with only a few types populated
        pairs = {
            "MZ twin": (np.array([], dtype=int), np.array([], dtype=int)),
            "Full sib": (np.arange(50, 75), np.arange(75, 100)),
            "Mother-offspring": (np.arange(50, 70), np.arange(0, 20)),
            "Father-offspring": (np.arange(50, 70), np.arange(20, 40)),
            "Maternal half sib": (np.array([], dtype=int), np.array([], dtype=int)),
            "Paternal half sib": (np.array([], dtype=int), np.array([], dtype=int)),
            "1st cousin": (np.array([], dtype=int), np.array([], dtype=int)),
        }

        result = compute_weibull_pair_corr(df, trait_num=1, scale=31.623, rho=2.0, beta=1.0, pairs=pairs)

        # All 7 pair types should be present
        assert len(result) == 7
        for ptype in pairs:
            assert ptype in result
            entry = result[ptype]
            assert "r" in entry
            assert "se" in entry
            assert "n_pairs" in entry

        # Empty pair types should have r=None
        assert result["MZ twin"]["r"] is None
        assert result["MZ twin"]["n_pairs"] == 0

        # Non-empty pair types should have numeric r
        assert result["Full sib"]["n_pairs"] == 25
