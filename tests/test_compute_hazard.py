"""Tests for compute_hazard_terms across all supported hazard models."""

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm, weibull_min

from sim_ace.compute_hazard_terms import compute_hazard_terms


# ---------------------------------------------------------------------------
# Weibull
# ---------------------------------------------------------------------------
class TestWeibull:
    """Weibull model: h0(t) = (rho/s)*(t/s)^(rho-1), H0(t) = (t/s)^rho."""

    params = {"scale": 100.0, "rho": 2.0}
    t = np.array([10.0, 50.0])

    def _analytical_h0(self, t):
        s, rho = self.params["scale"], self.params["rho"]
        return (rho / s) * (t / s) ** (rho - 1)

    def _analytical_H0(self, t):
        s, rho = self.params["scale"], self.params["rho"]
        return (t / s) ** rho

    def test_weibull_hazard(self):
        const, _ = compute_hazard_terms("weibull", self.t, self.params)
        h0 = np.exp(const)
        expected = self._analytical_h0(self.t)
        assert h0 == pytest.approx(expected, rel=1e-6)

    def test_weibull_cumulative_hazard(self):
        _, H_base = compute_hazard_terms("weibull", self.t, self.params)
        expected = self._analytical_H0(self.t)
        assert H_base == pytest.approx(expected, rel=1e-6)

    def test_weibull_survival_identity(self):
        _, H_base = compute_hazard_terms("weibull", self.t, self.params)
        S0 = np.exp(-H_base)
        s, rho = self.params["scale"], self.params["rho"]
        expected = weibull_min.sf(self.t, c=rho, scale=s)
        assert pytest.approx(expected, rel=1e-6) == S0

    def test_weibull_density_identity(self):
        const, H_base = compute_hazard_terms("weibull", self.t, self.params)
        f0 = np.exp(const) * np.exp(-H_base)
        s, rho = self.params["scale"], self.params["rho"]
        expected = weibull_min.pdf(self.t, c=rho, scale=s)
        assert f0 == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------
class TestExponential:
    """Exponential model: h0(t) = lam, H0(t) = lam*t."""

    params = {"rate": 0.01}
    t = np.array([10.0, 100.0])

    def test_exponential_hazard(self):
        const, _ = compute_hazard_terms("exponential", self.t, self.params)
        h0 = np.exp(const)
        expected = np.full_like(self.t, self.params["rate"])
        assert h0 == pytest.approx(expected, rel=1e-6)

    def test_exponential_cumulative_hazard(self):
        _, H_base = compute_hazard_terms("exponential", self.t, self.params)
        expected = self.params["rate"] * self.t
        assert H_base == pytest.approx(expected, rel=1e-6)

    def test_exponential_survival_identity(self):
        _, H_base = compute_hazard_terms("exponential", self.t, self.params)
        S0 = np.exp(-H_base)
        lam = self.params["rate"]
        expected = np.exp(-lam * self.t)
        assert pytest.approx(expected, rel=1e-6) == S0

    def test_exponential_density_identity(self):
        const, H_base = compute_hazard_terms("exponential", self.t, self.params)
        f0 = np.exp(const) * np.exp(-H_base)
        lam = self.params["rate"]
        expected = lam * np.exp(-lam * self.t)
        assert f0 == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Gompertz
# ---------------------------------------------------------------------------
class TestGompertz:
    """Gompertz model: h0(t) = b*exp(g*t), H0(t) = (b/g)*(exp(g*t)-1)."""

    params = {"rate": 0.001, "gamma": 0.05}
    t = np.array([10.0, 50.0])

    def _analytical_h0(self, t, b, g):
        return b * np.exp(g * t)

    def _analytical_H0(self, t, b, g):
        return (b / g) * (np.exp(g * t) - 1)

    def test_gompertz_hazard(self):
        const, _ = compute_hazard_terms("gompertz", self.t, self.params)
        h0 = np.exp(const)
        b, g = self.params["rate"], self.params["gamma"]
        expected = self._analytical_h0(self.t, b, g)
        assert h0 == pytest.approx(expected, rel=1e-6)

    def test_gompertz_cumulative_hazard(self):
        _, H_base = compute_hazard_terms("gompertz", self.t, self.params)
        b, g = self.params["rate"], self.params["gamma"]
        expected = self._analytical_H0(self.t, b, g)
        assert H_base == pytest.approx(expected, rel=1e-6)

    def test_gompertz_survival_identity(self):
        _, H_base = compute_hazard_terms("gompertz", self.t, self.params)
        S0 = np.exp(-H_base)
        b, g = self.params["rate"], self.params["gamma"]
        expected = np.exp(-self._analytical_H0(self.t, b, g))
        assert pytest.approx(expected, rel=1e-6) == S0

    def test_gompertz_density_identity(self):
        const, H_base = compute_hazard_terms("gompertz", self.t, self.params)
        f0 = np.exp(const) * np.exp(-H_base)
        b, g = self.params["rate"], self.params["gamma"]
        expected = self._analytical_h0(self.t, b, g) * np.exp(
            -self._analytical_H0(self.t, b, g)
        )
        assert f0 == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Lognormal
# ---------------------------------------------------------------------------
class TestLognormal:
    """Lognormal model using standard normal CDF/PDF."""

    params = {"mu": 4.0, "sigma": 1.0}
    t = np.array([10.0, 50.0])

    def _z(self, t):
        return (np.log(t) - self.params["mu"]) / self.params["sigma"]

    def test_lognormal_hazard(self):
        const, _ = compute_hazard_terms("lognormal", self.t, self.params)
        h0 = np.exp(const)
        sigma = self.params["sigma"]
        z = self._z(self.t)
        S0 = 1 - norm.cdf(z)
        expected = norm.pdf(z) / (sigma * self.t * S0)
        assert h0 == pytest.approx(expected, rel=1e-6)

    def test_lognormal_cumulative_hazard(self):
        _, H_base = compute_hazard_terms("lognormal", self.t, self.params)
        z = self._z(self.t)
        S0 = 1 - norm.cdf(z)
        expected = -np.log(S0)
        assert H_base == pytest.approx(expected, rel=1e-6)

    def test_lognormal_survival_identity(self):
        _, H_base = compute_hazard_terms("lognormal", self.t, self.params)
        S0 = np.exp(-H_base)
        mu, sigma = self.params["mu"], self.params["sigma"]
        # scipy lognorm: shape=sigma, scale=exp(mu)
        expected = lognorm.sf(self.t, s=sigma, scale=np.exp(mu))
        assert pytest.approx(expected, rel=1e-6) == S0

    def test_lognormal_density_identity(self):
        const, H_base = compute_hazard_terms("lognormal", self.t, self.params)
        f0 = np.exp(const) * np.exp(-H_base)
        mu, sigma = self.params["mu"], self.params["sigma"]
        expected = lognorm.pdf(self.t, s=sigma, scale=np.exp(mu))
        assert f0 == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Log-logistic
# ---------------------------------------------------------------------------
class TestLoglogistic:
    """Log-logistic model: H0(t)=log(1+(t/alpha)^k)."""

    params = {"scale": 50.0, "shape": 3.0}
    t = np.array([10.0, 50.0])

    def _analytical_h0(self, t):
        alpha, k = self.params["scale"], self.params["shape"]
        return (k / alpha) * (t / alpha) ** (k - 1) / (1 + (t / alpha) ** k)

    def _analytical_H0(self, t):
        alpha, k = self.params["scale"], self.params["shape"]
        return np.log(1 + (t / alpha) ** k)

    def test_loglogistic_hazard(self):
        const, _ = compute_hazard_terms("loglogistic", self.t, self.params)
        h0 = np.exp(const)
        expected = self._analytical_h0(self.t)
        assert h0 == pytest.approx(expected, rel=1e-6)

    def test_loglogistic_cumulative_hazard(self):
        _, H_base = compute_hazard_terms("loglogistic", self.t, self.params)
        expected = self._analytical_H0(self.t)
        assert H_base == pytest.approx(expected, rel=1e-6)

    def test_loglogistic_survival_identity(self):
        _, H_base = compute_hazard_terms("loglogistic", self.t, self.params)
        S0 = np.exp(-H_base)
        alpha, k = self.params["scale"], self.params["shape"]
        expected = 1 / (1 + (self.t / alpha) ** k)
        assert pytest.approx(expected, rel=1e-6) == S0

    def test_loglogistic_density_identity(self):
        const, H_base = compute_hazard_terms("loglogistic", self.t, self.params)
        f0 = np.exp(const) * np.exp(-H_base)
        alpha, k = self.params["scale"], self.params["shape"]
        expected = self._analytical_h0(self.t) / (1 + (self.t / alpha) ** k)
        assert f0 == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------
class TestGamma:
    """Gamma model using scipy.stats.gamma for reference."""

    params = {"shape": 2.0, "scale": 50.0}
    t = np.array([10.0, 50.0])

    def test_gamma_hazard(self):
        const, _ = compute_hazard_terms("gamma", self.t, self.params)
        h0 = np.exp(const)
        k, theta = self.params["shape"], self.params["scale"]
        pdf = gamma_dist.pdf(self.t, a=k, scale=theta)
        sf = gamma_dist.sf(self.t, a=k, scale=theta)
        expected = pdf / sf
        assert h0 == pytest.approx(expected, rel=1e-6)

    def test_gamma_cumulative_hazard(self):
        _, H_base = compute_hazard_terms("gamma", self.t, self.params)
        k, theta = self.params["shape"], self.params["scale"]
        expected = -gamma_dist.logsf(self.t, a=k, scale=theta)
        assert H_base == pytest.approx(expected, rel=1e-6)

    def test_gamma_survival_identity(self):
        _, H_base = compute_hazard_terms("gamma", self.t, self.params)
        S0 = np.exp(-H_base)
        k, theta = self.params["shape"], self.params["scale"]
        expected = gamma_dist.sf(self.t, a=k, scale=theta)
        assert pytest.approx(expected, rel=1e-6) == S0

    def test_gamma_density_identity(self):
        const, H_base = compute_hazard_terms("gamma", self.t, self.params)
        f0 = np.exp(const) * np.exp(-H_base)
        k, theta = self.params["shape"], self.params["scale"]
        expected = gamma_dist.pdf(self.t, a=k, scale=theta)
        assert f0 == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Cross-model and edge-case tests
# ---------------------------------------------------------------------------
class TestCrossModelAndEdgeCases:
    """Tests spanning multiple models or covering edge cases."""

    def test_exponential_rate_vs_scale(self):
        """Both parameterizations of exponential give identical results."""
        t = np.array([10.0, 100.0])
        const_r, H_r = compute_hazard_terms("exponential", t, {"rate": 0.01})
        const_s, H_s = compute_hazard_terms("exponential", t, {"scale": 100.0})
        assert const_r == pytest.approx(const_s, rel=1e-6)
        assert H_r == pytest.approx(H_s, rel=1e-6)

    def test_gompertz_near_zero_gamma(self):
        """Gompertz with gamma near 0 degrades to exponential-like H0 = b*t."""
        t = np.array([10.0, 50.0])
        b = 0.001
        params = {"rate": b, "gamma": 1e-15}
        _, H_base = compute_hazard_terms("gompertz", t, params)
        expected = b * t
        assert H_base == pytest.approx(expected, rel=1e-6)

    def test_unknown_model_raises(self):
        """Unknown model name raises ValueError."""
        t = np.array([10.0])
        with pytest.raises(ValueError, match="Unknown hazard model"):
            compute_hazard_terms("unknown_model", t, {})

    def test_weibull_rho_1_is_exponential(self):
        """Weibull with rho=1 is equivalent to exponential with same scale."""
        t = np.array([10.0, 50.0, 100.0])
        scale = 100.0
        const_w, H_w = compute_hazard_terms(
            "weibull", t, {"scale": scale, "rho": 1.0}
        )
        const_e, H_e = compute_hazard_terms(
            "exponential", t, {"scale": scale}
        )
        assert np.exp(const_w) == pytest.approx(np.exp(const_e), rel=1e-6)
        assert H_w == pytest.approx(H_e, rel=1e-6)
