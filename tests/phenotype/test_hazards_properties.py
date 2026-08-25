"""Property-based tests for :mod:`simace.phenotype.hazards`.

The centrepiece is a deterministic inverse-hazard oracle: every baseline
inverter claims to solve ``H0(t) * z = -log(U)`` for ``t``, so an independently
written ``H0`` recovers the input.  No parameter is recovered from a Monte
Carlo sample and no tolerance here absorbs sampling uncertainty — each one
covers floating-point error or the documented ``_ndtri_approx`` approximation,
and is recorded beside its assertion with the domain it was measured over.

Calibration (D4 gate), seed 20260825, 200_000 draws per distribution over
exactly the strategy domain in ``_HAZARD_PARAM_BOUNDS`` / ``_hazard_call``:

    distribution   worst rel err   worst rel err (subnormal S(t))
    weibull            3.39e-15        3.70e-15
    exponential        4.25e-16        3.74e-16
    gompertz           2.75e-15        4.63e-15
    lognormal          2.86e-08        3.10e-04
    loglogistic        2.72e-15        (never reached)
    gamma              1.13e-11        4.06e-16

Those figures come from uniform sampling, which reaches neither precision
boundary of the survival-probability representation; a ``thorough`` Hypothesis
run found the upper one within minutes. See ``_SURVIVAL_ULP_SAFETY`` for the
derived bound that covers both, and its own calibration.
"""

import argparse

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from simace.phenotype.hazards import (
    BASELINE_HAZARDS,
    STANDARDIZE_CHOICES,
    add_hazard_cli_args,
    compute_event_times,
    hazard_cli_flag_attrs,
    iter_generation_groups,
    parse_hazard_cli,
    standardize_beta,
    standardize_liability,
    true_lifetime_prevalence_weibull,
    validate_hazard_params,
)

CLAMP_LO, CLAMP_HI = 1e-10, 1e6

# Round-trip tolerances, one per distribution, from the calibration above.
# Every margin is at least 30x the worst observed error.
_ROUNDTRIP_RTOL = {
    "weibull": 1e-12,
    "exponential": 1e-12,
    "gompertz": 1e-12,
    "loglogistic": 1e-12,
    # scipy's iterative gamma inverse, worst 1.13e-11 near the clamp floor.
    "gamma": 1e-9,
    # _ndtri_approx (hazards.py:118) is Acklam's rational approximation, not an
    # exact inverse normal CDF: measured 1.13e-9 relative in the quantile
    # itself, worst 2.86e-08 through the round-trip.
    "lognormal": 1e-6,
}

# The lognormal and gamma inverters are the two that materialise the survival
# probability S(t) = exp(-target) as a double before inverting it; the other
# four work with ``target`` directly. Storing S(t) quantises it, and the kernel
# recovers ``target`` as -log(S), so one ULP of S maps to a relative error of
# about ``ulp(S) / (2 * S * target)`` in the recovered value. That single
# expression covers both ends: S near 1 (tiny target -- catastrophic
# cancellation) and S subnormal (target above ~708 -- fewer than 53 significant
# bits left). Measured against 235_000 draws deliberately weighted toward both
# boundaries, the observed error never exceeded 1.4x this bound, so the safety
# factor below leaves roughly 6x margin. Neither end is excluded from the
# strategy domain -- the bound simply widens where double precision genuinely
# cannot carry the answer.
_SURVIVAL_KERNELS = ("gamma", "lognormal")
_SURVIVAL_ULP_SAFETY = 8.0

# Strategy domain, fixed here so the calibration probe and the assertions below
# cover exactly the same inputs.
_HAZARD_PARAM_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "weibull": {"scale": (1.0, 1000.0), "rho": (0.2, 10.0)},
    "exponential": {"rate": (1e-4, 10.0)},
    "gompertz": {"rate": (1e-6, 1.0), "gamma": (1e-4, 1.0)},
    "lognormal": {"mu": (-2.0, 6.0), "sigma": (0.05, 3.0)},
    "loglogistic": {"scale": (1.0, 1000.0), "shape": (0.2, 10.0)},
    "gamma": {"shape": (0.1, 20.0), "scale": (0.1, 500.0)},
}


def _bounded_float(lo: float, hi: float):
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


def _cumulative_hazard(distribution: str, t: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Independently written ``H0(t)`` for each baseline distribution.

    Closed form where one exists; log-space survival via scipy for lognormal
    and gamma, so the oracle itself never underflows where the kernel does not.
    """
    if distribution == "weibull":
        return (t / params["scale"]) ** params["rho"]
    if distribution == "exponential":
        return params["rate"] * t
    if distribution == "gompertz":
        return (params["rate"] / params["gamma"]) * np.expm1(params["gamma"] * t)
    if distribution == "loglogistic":
        return np.log1p((t / params["scale"]) ** params["shape"])
    if distribution == "lognormal":
        return -norm.logsf((np.log(t) - params["mu"]) / params["sigma"])
    if distribution == "gamma":
        return -gamma_dist.logsf(t, params["shape"], scale=params["scale"])
    raise AssertionError(f"no oracle for {distribution}")


@st.composite
def _hazard_params(draw, distribution: str) -> dict[str, float]:
    """Draw a valid parameter dict for one baseline distribution."""
    return {key: draw(_bounded_float(*bounds)) for key, bounds in _HAZARD_PARAM_BOUNDS[distribution].items()}


@st.composite
def _hazard_call(draw, *, distribution=None):
    """Draw ``(distribution, params, neg_log_u, liability, mean, scaled_beta)``.

    ``neg_log_u`` deliberately reaches exactly ``0.0``: that boundary is where
    the Weibull and lognormal kernels used to leave the documented clamp, and
    excluding it would hide the very inputs the bounds contract is about.
    """
    if distribution is None:
        distribution = draw(st.sampled_from(sorted(BASELINE_HAZARDS)))
    params = draw(_hazard_params(distribution))
    n = draw(st.integers(min_value=1, max_value=12))
    neg_log_u = draw(
        hnp.arrays(
            np.float64,
            n,
            elements=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        )
    )
    liability = draw(
        hnp.arrays(
            np.float64,
            n,
            elements=st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
        )
    )
    mean = draw(_bounded_float(-2.0, 2.0))
    scaled_beta = draw(_bounded_float(-2.0, 2.0))
    return distribution, params, neg_log_u, liability, mean, scaled_beta


class TestEventTimes:
    """``compute_event_times`` return contract and inverse-hazard oracle."""

    @given(call=_hazard_call())
    def test_event_times_are_finite_and_clamped(self, call):
        """Every distribution returns finite times inside ``[1e-10, 1e6]``.

        The contract is stated by ``compute_event_times``' docstring and was
        enforced by only four of the six inversion paths: ``_nb_weibull``
        returned its raw inverse (0 at ``neg_log_u=0``, and above ``1e6`` under
        an extreme frailty), and ``_nb_lognormal`` returned NaN as
        ``neg_log_u -> 0`` because ``_ndtri_approx(1.0)`` diverges.
        """
        distribution, params, neg_log_u, liability, mean, scaled_beta = call
        t = compute_event_times(neg_log_u, liability, mean, scaled_beta, distribution, params)
        assert np.all(np.isfinite(t)), f"{distribution}: non-finite event times"
        assert np.all(t >= CLAMP_LO), f"{distribution}: below the documented floor"
        assert np.all(t <= CLAMP_HI), f"{distribution}: above the documented ceiling"

    @given(call=_hazard_call())
    def test_inverse_hazard_round_trip(self, call):
        """``H0(t) * z`` recovers ``neg_log_u`` wherever ``t`` is not clamped.

        Only values strictly inside ``[1e-10, 1e6]`` are checked: at either
        clamp the contract is saturation, not inversion.  For the two kernels
        that route through ``S(t) = exp(-target)`` the tolerance carries the
        derived ``_SURVIVAL_ULP_SAFETY`` term, which widens exactly where that
        double cannot represent the input and stays at the flat rate elsewhere.
        """
        distribution, params, neg_log_u, liability, mean, scaled_beta = call
        t = compute_event_times(neg_log_u, liability, mean, scaled_beta, distribution, params)
        inside = (t > CLAMP_LO) & (t < CLAMP_HI) & (neg_log_u > 0)
        if not inside.any():
            return
        z = np.exp(scaled_beta * (liability[inside] - mean))
        recovered = _cumulative_hazard(distribution, t[inside], params) * z
        want = neg_log_u[inside]
        finite = np.isfinite(recovered)
        if not finite.any():
            return

        want, recovered, z = want[finite], recovered[finite], z[finite]
        tolerance = np.full(want.shape, _ROUNDTRIP_RTOL[distribution])
        if distribution in _SURVIVAL_KERNELS:
            target = want / z
            survival = np.exp(-target)
            # S(t) underflows to zero only past target ~= 745, where both
            # kernels return the 1e6 ceiling -- already excluded above. Asserted
            # rather than guarded, so the reasoning is checked and not assumed.
            assert np.all(survival > 0.0)
            tolerance = tolerance + _SURVIVAL_ULP_SAFETY * np.spacing(survival) / (2.0 * survival * target)
        assert np.all(np.abs(recovered - want) <= tolerance * want)

    @given(call=_hazard_call(), liability=_bounded_float(-4.0, 4.0))
    def test_event_time_is_monotone_in_neg_log_u(self, call, liability):
        """Sorting ``neg_log_u`` sorts the event times, at fixed liability.

        Ties are permitted: both clamps saturate, and distinct inputs can map
        to the same boundary value.
        """
        distribution, params, neg_log_u, _, mean, scaled_beta = call
        liabilities = np.full(len(neg_log_u), liability)
        t = compute_event_times(np.sort(neg_log_u), liabilities, mean, scaled_beta, distribution, params)
        assert np.all(np.diff(t) >= 0.0), f"{distribution}: event time is not monotone in neg_log_u"


@st.composite
def _liability_and_generation(draw, *, spread=True, max_n=12):
    """Draw ``(liability, generation)`` with a controlled within-group spread.

    With ``spread=True`` the liabilities are distinct by a drawn step and then
    permuted, so every subset of two or more rows — the whole frame under
    ``global``, one generation under ``per_generation`` — has a spread of at
    least that step and the z-score oracle is well-conditioned.  With
    ``spread=False`` every value is identical: the degenerate branch, which
    must return finite values rather than NaN.
    """
    n = draw(st.integers(min_value=1, max_value=max_n))
    n_gen = draw(st.integers(min_value=1, max_value=3))
    generation = np.asarray([draw(st.integers(min_value=0, max_value=n_gen - 1)) for _ in range(n)], dtype=np.int64)

    base = draw(_bounded_float(-4.0, 4.0))
    if not spread:
        return np.full(n, base, dtype=np.float64), generation

    step = draw(_bounded_float(1e-2, 2.0))
    order = np.asarray(draw(st.permutations(range(n))), dtype=np.int64)
    return base + step * order.astype(np.float64), generation


class TestStandardization:
    """``standardize_beta`` / ``standardize_liability`` / ``iter_generation_groups``."""

    @given(
        pair=_liability_and_generation(),
        beta=_bounded_float(-3.0, 3.0),
        mode=st.sampled_from(STANDARDIZE_CHOICES),
    )
    def test_scaled_beta_matches_standardized_liability(self, pair, beta, mode):
        """``scaled_beta * (L - mean) == beta * standardize_liability(L, mode)``.

        The load-bearing coupling between the two helpers: the hazard kernels
        consume the left-hand form, every threshold model the right-hand one.
        Tolerance is pure floating point — both sides are the same product in a
        different association order, and the degenerate branch zeroes
        ``scaled_beta`` while ``standardize_liability`` returns a residual
        below its own ``1e-12`` threshold.  Measured worst 3.98e-15 absolute
        over this domain (seed 20260825, 200_000 draws).
        """
        liability, generation = pair
        mean_arr, beta_arr = standardize_beta(liability, beta, mode, generation)
        standardized = standardize_liability(liability, mode, generation)
        assert np.allclose(beta_arr * (liability - mean_arr), beta * standardized, rtol=0.0, atol=1e-9)

    @given(pair=_liability_and_generation(), mode=st.sampled_from(["global", "per_generation"]))
    def test_standardized_groups_have_unit_moments(self, pair, mode):
        """Each standardized group is centred with unit standard deviation.

        A singleton group has zero variance by construction and takes the
        degenerate branch, so it is checked for centring only.  Measured worst
        over this domain (seed 20260825, 200_000 draws): 5.11e-14 for the mean,
        4.44e-16 for ``|std - 1|``.
        """
        liability, generation = pair
        out = standardize_liability(liability, mode, generation)
        groups = (
            [np.ones(len(liability), dtype=bool)]
            if mode == "global"
            else [generation == g for g in np.unique(generation)]
        )
        for mask in groups:
            sub = out[mask]
            assert np.isfinite(sub).all()
            assert abs(float(sub.mean())) < 1e-8
            if mask.sum() > 1:
                assert abs(float(sub.std()) - 1.0) < 1e-8

    @given(
        pair=_liability_and_generation(spread=False),
        beta=_bounded_float(-3.0, 3.0),
        mode=st.sampled_from(STANDARDIZE_CHOICES),
    )
    def test_degenerate_groups_stay_finite(self, pair, mode, beta):
        """An all-equal group returns ``L - mean``, never NaN from a 0/0 divide."""
        liability, generation = pair
        assert np.isfinite(standardize_liability(liability, mode, generation)).all()
        mean_arr, beta_arr = standardize_beta(liability, beta, mode, generation)
        assert np.isfinite(mean_arr).all()
        assert np.isfinite(beta_arr).all()

    @given(pair=_liability_and_generation(), mode=st.sampled_from(STANDARDIZE_CHOICES))
    def test_generation_groups_partition_rows(self, pair, mode):
        """Every row is covered by exactly one mask, in all three modes."""
        _, generation = pair
        coverage = np.zeros(len(generation), dtype=np.int64)
        for mask in iter_generation_groups(mode, generation):
            assert mask.dtype == np.bool_
            assert len(mask) == len(generation)
            coverage += mask
        assert np.all(coverage == 1)


class TestTrueLifetimePrevalence:
    """``true_lifetime_prevalence_weibull`` against a closed form and its bounds."""

    @given(
        scale=_bounded_float(1.0, 1000.0),
        rho=_bounded_float(0.2, 10.0),
        max_age=_bounded_float(1.0, 200.0),
    )
    def test_zero_beta_matches_the_closed_form(self, scale, rho, max_age):
        """At ``beta=0`` the frailty is 1 and quadrature collapses to an exact form.

        ``K = 1 - exp(-(max_age / scale) ** rho)``; the Gauss-Hermite weights
        sum to one, so this is an exact oracle rather than a quadrature
        approximation.  Tolerance is floating point only.
        """
        got = true_lifetime_prevalence_weibull(scale, rho, 0.0, max_age)
        want = 1.0 - np.exp(-((max_age / scale) ** rho))
        assert abs(got - want) < 1e-12

    @given(
        scale=_bounded_float(1.0, 1000.0),
        rho=_bounded_float(0.2, 10.0),
        beta=_bounded_float(0.0, 3.0),
        age_low=_bounded_float(1.0, 200.0),
        age_step=_bounded_float(0.0, 200.0),
    )
    def test_prevalence_is_a_bounded_non_decreasing_function_of_age(self, scale, rho, beta, age_low, age_step):
        """``K`` stays in ``[0, 1]`` and never falls as ``max_age`` grows."""
        k_low = true_lifetime_prevalence_weibull(scale, rho, beta, age_low)
        k_high = true_lifetime_prevalence_weibull(scale, rho, beta, age_low + age_step)
        for value in (k_low, k_high):
            assert 0.0 <= value <= 1.0
        assert k_high >= k_low - 1e-12


_CLI_NAME = "frailty"


def _build_parser(trait: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    add_hazard_cli_args(parser, trait, name=_CLI_NAME)
    return parser


class TestHazardCli:
    """``add_hazard_cli_args`` → argv → ``parse_hazard_cli`` → ``validate_hazard_params``."""

    @given(
        trait=st.integers(min_value=1, max_value=2),
        distribution=st.sampled_from(sorted(BASELINE_HAZARDS)),
        data=st.data(),
    )
    def test_cli_round_trip(self, trait, distribution, data):
        """Every distribution's canonical flags parse back to valid params."""
        params = data.draw(_hazard_params(distribution))
        # ``--flag=value`` form throughout: several parameters are legitimately
        # negative (lognormal ``mu``), and argparse reads a bare ``-2.0`` as a flag.
        argv = [f"--{_CLI_NAME}-distribution{trait}={distribution}"]
        argv += [f"--{_CLI_NAME}-{key}{trait}={value!r}" for key, value in params.items()]

        args = _build_parser(trait).parse_args(argv)
        parsed_distribution, parsed_params = parse_hazard_cli(args, trait, name=_CLI_NAME)
        assert parsed_distribution == distribution
        assert parsed_params == params
        validate_hazard_params(parsed_distribution, parsed_params, "frailty")

    @given(trait=st.integers(min_value=1, max_value=2))
    def test_registered_attrs_match_the_declared_set(self, trait):
        """``hazard_cli_flag_attrs`` lists exactly what the parser registers."""
        args = _build_parser(trait).parse_args([f"--{_CLI_NAME}-distribution{trait}=weibull"])
        assert set(vars(args)) == hazard_cli_flag_attrs(trait, name=_CLI_NAME)

    @given(
        trait=st.integers(min_value=1, max_value=2),
        rate=_bounded_float(1e-4, 10.0),
        scale=_bounded_float(1e-4, 10.0),
        supply=st.sampled_from(["rate", "scale", "both"]),
    )
    def test_exponential_accepts_rate_or_scale(self, trait, rate, scale, supply):
        """Exponential parses either flag, and canonical ``rate`` wins when both are given.

        ``_invert_exponential`` and ``validate_hazard_params`` have always
        accepted either key; ``parse_hazard_cli`` iterated over the canonical
        ``BASELINE_PARAMS["exponential"] == ["rate"]`` and was the only layer
        that rejected ``scale``.
        """
        argv = [f"--{_CLI_NAME}-distribution{trait}=exponential"]
        if supply in ("rate", "both"):
            argv.append(f"--{_CLI_NAME}-rate{trait}={rate!r}")
        if supply in ("scale", "both"):
            argv.append(f"--{_CLI_NAME}-scale{trait}={scale!r}")

        args = _build_parser(trait).parse_args(argv)
        distribution, params = parse_hazard_cli(args, trait, name=_CLI_NAME)
        validate_hazard_params(distribution, params, "frailty")
        assert params == ({"scale": scale} if supply == "scale" else {"rate": rate})
