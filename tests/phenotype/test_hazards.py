"""Unit tests for simace.phenotype.hazards: error paths, exact contracts, CLI helpers.

Finiteness, clamping, unit moments after standardization, generation-group
partitioning and prevalence monotonicity are property-tested in
``test_hazards_properties.py`` and are not repeated here.
"""

import argparse

import numpy as np
import pytest

from simace.phenotype.hazards import (
    BASELINE_HAZARDS,
    BASELINE_PARAMS,
    add_hazard_cli_args,
    coerce_standardize_mode,
    compute_event_times,
    iter_generation_groups,
    parse_hazard_cli,
    resolve_hazard_mode,
    standardize_beta,
    standardize_liability,
    true_lifetime_prevalence_weibull,
    validate_hazard_params,
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


def test_standardize_liability_global_zero_std_returns_centered():
    L = np.full(50, 3.0)
    out = standardize_liability(L, "global")
    np.testing.assert_array_equal(out, np.zeros(50))


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


def test_standardize_beta_global_zero_std_returns_zero_beta():
    L = np.full(50, 3.0)
    mean, sbeta = standardize_beta(L, beta=2.0, mode="global")
    np.testing.assert_array_equal(mean, np.full(50, 3.0))
    np.testing.assert_array_equal(sbeta, np.zeros(50))


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


def test_iter_generation_groups_per_gen_single_gen_yields_one_mask():
    g = np.zeros(10)
    masks = list(iter_generation_groups("per_generation", g))
    assert len(masks) == 1
    assert masks[0].all()


def test_iter_generation_groups_empty_per_gen_yields_nothing():
    g = np.array([], dtype=int)
    masks = list(iter_generation_groups("per_generation", g))
    assert masks == []


@pytest.mark.parametrize(
    ("scale", "rho", "beta", "max_age", "expected"),
    [
        (2160.0, 0.8, 1.0, 80.0, 0.1029),  # baseline trait1
        (333.0, 1.2, 1.5, 80.0, 0.2676),  # baseline trait2
    ],
)
def test_true_lifetime_prevalence_weibull_known_values(scale, rho, beta, max_age, expected):
    k = true_lifetime_prevalence_weibull(scale, rho, beta, max_age)
    assert 0.0 < k < 1.0
    assert k == pytest.approx(expected, abs=5e-3)


def test_true_lifetime_prevalence_weibull_matches_generative_model():
    # The quadrature must reproduce the _nb_weibull inversion
    # T = scale * (E / z) ** (1/rho), E ~ Exp(1), z = exp(beta * L), L ~ N(0,1).
    scale, rho, beta, max_age = 2160.0, 0.8, 1.0, 80.0
    rng = np.random.default_rng(0)
    n = 1_000_000
    liability = rng.standard_normal(n)
    neg_log_u = rng.exponential(1.0, n)
    z = np.exp(beta * liability)
    t = scale * (neg_log_u / z) ** (1.0 / rho)
    k_mc = float((t <= max_age).mean())
    k = true_lifetime_prevalence_weibull(scale, rho, beta, max_age)
    assert k == pytest.approx(k_mc, abs=2e-3)


# ---------------------------------------------------------------------------
# validate_hazard_params
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="unknown frailty distribution"):
        validate_hazard_params("not_a_real_dist", {}, "frailty")


@pytest.mark.parametrize(
    ("distribution", "missing"),
    [
        ("weibull", "rho"),
        ("gompertz", "gamma"),
        ("lognormal", "sigma"),
    ],
)
def test_validate_rejects_missing_required_keys(distribution, missing):
    bad_params = {k: v for k, v in DEFAULT_PARAMS[distribution].items() if k != missing}
    with pytest.raises(ValueError, match="missing required hazard params"):
        validate_hazard_params(distribution, bad_params, "frailty")


# ---------------------------------------------------------------------------
# add_hazard_cli_args + parse_hazard_cli error paths
# ---------------------------------------------------------------------------


def _parser_for(trait: int, name: str = "frailty") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_hazard_cli_args(parser, trait, name=name)
    return parser


def test_missing_distribution_flag_raises():
    args = _parser_for(trait=1).parse_args([])
    with pytest.raises(ValueError, match="--frailty-distribution1 is required"):
        parse_hazard_cli(args, trait=1, name="frailty")


def test_missing_required_param_flag_raises():
    args = _parser_for(trait=1).parse_args(["--frailty-distribution1", "weibull", "--frailty-scale1", "100.0"])
    with pytest.raises(ValueError, match="--frailty-rho1 is required"):
        parse_hazard_cli(args, trait=1, name="frailty")


def test_kebab_name_maps_to_snake_attr():
    """``name='cure-frailty'`` registers attrs like ``cure_frailty_distribution1``."""
    args = _parser_for(trait=1, name="cure-frailty").parse_args(
        ["--cure-frailty-distribution1", "weibull", "--cure-frailty-scale1", "50.0", "--cure-frailty-rho1", "1.5"]
    )
    dist, params = parse_hazard_cli(args, trait=1, name="cure-frailty")
    assert dist == "weibull"
    assert params == {"scale": 50.0, "rho": 1.5}
