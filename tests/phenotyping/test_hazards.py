"""Unit tests for simace.phenotyping.hazards."""

import numpy as np
import pytest

from simace.phenotyping.hazards import (
    BASELINE_HAZARDS,
    BASELINE_PARAMS,
    coerce_standardize_mode,
    compute_event_times,
    iter_generation_groups,
    resolve_hazard_mode,
    standardize_beta,
    standardize_liability,
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


# ---------------------------------------------------------------------------
# coerce_standardize_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["none", "global", "per_generation"])
def test_coerce_standardize_mode_string_passthrough(value):
    assert coerce_standardize_mode(value) == value


def test_coerce_standardize_mode_legacy_bool():
    assert coerce_standardize_mode(True) == "global"
    assert coerce_standardize_mode(False) == "none"


@pytest.mark.parametrize("bad", ["per_gen", "True", "GLOBAL", "", None, 1, 0.5])
def test_coerce_standardize_mode_invalid_raises(bad):
    with pytest.raises(ValueError, match="standardize must be one of"):
        coerce_standardize_mode(bad)


# ---------------------------------------------------------------------------
# resolve_hazard_mode
# ---------------------------------------------------------------------------


def test_resolve_hazard_mode_inherits_when_none():
    assert resolve_hazard_mode("global", None) == "global"
    assert resolve_hazard_mode("per_generation", None) == "per_generation"
    assert resolve_hazard_mode(True, None) == "global"
    assert resolve_hazard_mode(False, None) == "none"


def test_resolve_hazard_mode_override_takes_precedence():
    assert resolve_hazard_mode("global", "per_generation") == "per_generation"
    assert resolve_hazard_mode("per_generation", "none") == "none"
    assert resolve_hazard_mode("none", True) == "global"
    assert resolve_hazard_mode("global", False) == "none"


# ---------------------------------------------------------------------------
# standardize_liability
# ---------------------------------------------------------------------------


def test_standardize_liability_none_returns_input():
    rng = np.random.default_rng(0)
    L = rng.standard_normal(100)
    out = standardize_liability(L, "none")
    np.testing.assert_array_equal(out, L)


def test_standardize_liability_global():
    rng = np.random.default_rng(1)
    L = rng.normal(2.0, 3.0, size=10_000)
    out = standardize_liability(L, "global")
    assert out.mean() == pytest.approx(0.0, abs=1e-10)
    assert out.std() == pytest.approx(1.0, abs=1e-10)


def test_standardize_liability_global_zero_std_returns_centered():
    L = np.full(50, 3.0)
    out = standardize_liability(L, "global")
    np.testing.assert_array_equal(out, np.zeros(50))


def test_standardize_liability_per_generation_each_gen_unit_variance():
    rng = np.random.default_rng(2)
    n_per = 5000
    gen0 = rng.normal(1.0, 1.0, n_per)
    gen1 = rng.normal(-2.0, 3.0, n_per)
    L = np.concatenate([gen0, gen1])
    g = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    out = standardize_liability(L, "per_generation", g)
    for gi in (0.0, 1.0):
        sub = out[g == gi]
        assert sub.mean() == pytest.approx(0.0, abs=1e-10)
        assert sub.std() == pytest.approx(1.0, abs=1e-10)


def test_standardize_liability_per_generation_requires_generation():
    L = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="generation is required"):
        standardize_liability(L, "per_generation")


def test_standardize_liability_per_gen_singleton_returns_centered():
    """A generation with one individual gets L - mean (== 0), not NaN."""
    L = np.array([5.0, 1.0, 2.0, 3.0])
    g = np.array([0, 1, 1, 1])
    out = standardize_liability(L, "per_generation", g)
    assert out[0] == pytest.approx(0.0)  # singleton gen 0
    assert np.all(np.isfinite(out))


def test_standardize_liability_legacy_bool_passthrough():
    rng = np.random.default_rng(3)
    L = rng.standard_normal(1000)
    np.testing.assert_array_equal(standardize_liability(L, False), L)
    out_true = standardize_liability(L, True)
    assert out_true.mean() == pytest.approx(0.0, abs=1e-10)
    assert out_true.std() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# standardize_beta
# ---------------------------------------------------------------------------


def test_standardize_beta_none_returns_zeros_and_beta():
    L = np.array([0.0, 5.0, 10.0])
    mean, sbeta = standardize_beta(L, beta=2.5, mode="none")
    assert mean.shape == (3,)
    assert sbeta.shape == (3,)
    np.testing.assert_array_equal(mean, np.zeros(3))
    np.testing.assert_array_equal(sbeta, np.full(3, 2.5))


def test_standardize_beta_global_arrays_constant():
    rng = np.random.default_rng(7)
    L = rng.standard_normal(10_000)
    mean, sbeta = standardize_beta(L, beta=1.5, mode="global")
    assert mean.shape == L.shape
    assert sbeta.shape == L.shape
    assert np.all(mean == mean[0])
    assert np.all(sbeta == sbeta[0])
    assert mean[0] == pytest.approx(L.mean())
    assert sbeta[0] == pytest.approx(1.5 / L.std())


def test_standardize_beta_global_zero_std_returns_zero_beta():
    L = np.full(50, 3.0)
    mean, sbeta = standardize_beta(L, beta=2.0, mode="global")
    np.testing.assert_array_equal(mean, np.full(50, 3.0))
    np.testing.assert_array_equal(sbeta, np.zeros(50))


def test_standardize_beta_per_generation():
    rng = np.random.default_rng(8)
    n_per = 5000
    L = np.concatenate([rng.normal(1.0, 1.0, n_per), rng.normal(-2.0, 3.0, n_per)])
    g = np.concatenate([np.zeros(n_per), np.ones(n_per)])
    mean, sbeta = standardize_beta(L, beta=1.0, mode="per_generation", generation=g)
    # Each individual carries their own gen's stats
    assert mean[g == 0][0] == pytest.approx(L[g == 0].mean())
    assert mean[g == 1][0] == pytest.approx(L[g == 1].mean())
    assert sbeta[g == 0][0] == pytest.approx(1.0 / L[g == 0].std())
    assert sbeta[g == 1][0] == pytest.approx(1.0 / L[g == 1].std())
    # Within each gen, mean and sbeta are constant
    assert np.all(mean[g == 0] == mean[g == 0][0])
    assert np.all(sbeta[g == 1] == sbeta[g == 1][0])


def test_standardize_beta_per_generation_requires_generation():
    L = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="generation is required"):
        standardize_beta(L, beta=1.0, mode="per_generation")


def test_standardize_beta_per_gen_singleton_zero_beta():
    L = np.array([5.0, 1.0, 2.0, 3.0])
    g = np.array([0, 1, 1, 1])
    mean, sbeta = standardize_beta(L, beta=1.0, mode="per_generation", generation=g)
    assert mean[0] == pytest.approx(5.0)
    assert sbeta[0] == 0.0  # singleton → degenerate std → no scaling
    assert sbeta[1] == pytest.approx(1.0 / L[1:].std())


def test_standardize_beta_legacy_bool_passthrough():
    rng = np.random.default_rng(11)
    L = rng.standard_normal(1000)
    mean_t, sbeta_t = standardize_beta(L, beta=2.0, mode=True)
    mean_g, sbeta_g = standardize_beta(L, beta=2.0, mode="global")
    np.testing.assert_array_equal(mean_t, mean_g)
    np.testing.assert_array_equal(sbeta_t, sbeta_g)
    mean_f, sbeta_f = standardize_beta(L, beta=2.0, mode=False)
    np.testing.assert_array_equal(mean_f, np.zeros(1000))
    np.testing.assert_array_equal(sbeta_f, np.full(1000, 2.0))


# ---------------------------------------------------------------------------
# iter_generation_groups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "global", False, True])
def test_iter_generation_groups_non_per_gen_yields_single_full_mask(mode):
    g = np.array([0, 0, 1, 1, 2])
    masks = list(iter_generation_groups(mode, g))
    assert len(masks) == 1
    assert masks[0].shape == (5,)
    assert masks[0].all()


def test_iter_generation_groups_per_gen_yields_one_mask_per_unique_gen():
    g = np.array([0, 0, 1, 1, 2])
    masks = list(iter_generation_groups("per_generation", g))
    assert len(masks) == 3
    np.testing.assert_array_equal(masks[0], np.array([True, True, False, False, False]))
    np.testing.assert_array_equal(masks[1], np.array([False, False, True, True, False]))
    np.testing.assert_array_equal(masks[2], np.array([False, False, False, False, True]))


def test_iter_generation_groups_per_gen_single_gen_yields_one_mask():
    g = np.zeros(10)
    masks = list(iter_generation_groups("per_generation", g))
    assert len(masks) == 1
    assert masks[0].all()


def test_iter_generation_groups_empty_per_gen_yields_nothing():
    g = np.array([], dtype=int)
    masks = list(iter_generation_groups("per_generation", g))
    assert masks == []
