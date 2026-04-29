"""Unit tests for simace.phenotyping.hazards."""

from __future__ import annotations

import numpy as np
import pytest

from simace.phenotyping.hazards import (
    BASELINE_HAZARDS,
    BASELINE_PARAMS,
    compute_event_times,
    standardize_beta,
)

ALL_DISTRIBUTIONS = sorted(BASELINE_HAZARDS)
DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "weibull": {"scale": 316.228, "rho": 2.0},
    "exponential": {"rate": 0.01},
    "gompertz": {"rate": 1e-4, "gamma": 0.05},
    "lognormal": {"mu": 4.0, "sigma": 0.5},
    "loglogistic": {"scale": 50.0, "shape": 2.0},
    "gamma": {"shape": 2.0, "scale": 25.0},
}


def _draws(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    liability = rng.standard_normal(n)
    neg_log_u = rng.exponential(size=n)
    return liability, neg_log_u


@pytest.mark.parametrize("distribution", ALL_DISTRIBUTIONS)
def test_compute_event_times_finite(distribution):
    liability, neg_log_u = _draws()
    t = compute_event_times(neg_log_u, liability, 0.0, 1.0, distribution, DEFAULT_PARAMS[distribution])
    assert t.shape == liability.shape
    assert np.all(np.isfinite(t))
    assert np.all(t > 0)


@pytest.mark.parametrize("distribution", ALL_DISTRIBUTIONS)
def test_compute_event_times_monotone_in_z(distribution):
    """Higher liability (positive scaled_beta) → earlier mean event time."""
    n = 5000
    rng = np.random.default_rng(0)
    high = np.full(n, 1.0)
    low = np.full(n, -1.0)
    neg_log_u = rng.exponential(size=n)
    t_high = compute_event_times(neg_log_u, high, 0.0, 0.5, distribution, DEFAULT_PARAMS[distribution])
    t_low = compute_event_times(neg_log_u, low, 0.0, 0.5, distribution, DEFAULT_PARAMS[distribution])
    assert t_high.mean() < t_low.mean(), (
        f"{distribution}: expected higher liability → earlier onset, got "
        f"mean(t_high)={t_high.mean():.3f} mean(t_low)={t_low.mean():.3f}"
    )


def test_compute_event_times_unknown_distribution():
    liability, neg_log_u = _draws(n=10)
    with pytest.raises(ValueError, match="Unknown baseline hazard"):
        compute_event_times(neg_log_u, liability, 0.0, 1.0, "not_a_distribution", {})


def test_compute_event_times_missing_param():
    liability, neg_log_u = _draws(n=10)
    with pytest.raises(KeyError):
        compute_event_times(neg_log_u, liability, 0.0, 1.0, "weibull", {"scale": 100.0})


def test_baseline_params_keys_match_registry():
    assert set(BASELINE_PARAMS) == set(BASELINE_HAZARDS)


def test_standardize_beta_zero_std():
    """Constant liability → scaled_beta is 0.0 (avoids divide-by-zero)."""
    liability = np.full(50, 3.0)
    mean, scaled_beta = standardize_beta(liability, beta=2.0, standardize=True)
    assert mean == pytest.approx(3.0)
    assert scaled_beta == 0.0


def test_standardize_beta_unit_std():
    rng = np.random.default_rng(7)
    liability = rng.standard_normal(10000)
    mean, scaled_beta = standardize_beta(liability, beta=1.5, standardize=True)
    assert scaled_beta == pytest.approx(1.5 / np.std(liability))
    assert mean == pytest.approx(liability.mean())


def test_standardize_beta_no_standardize():
    liability = np.array([0.0, 5.0, 10.0])
    mean, scaled_beta = standardize_beta(liability, beta=2.5, standardize=False)
    assert mean == 0.0
    assert scaled_beta == 2.5
