"""Property-based survival invariant for every baseline-hazard model.

A cumulative baseline hazard ``H0(t)`` is non-negative and non-decreasing in
``t`` for any valid parameters — equivalently ``S0 = exp(-H0)`` is a survival
function in (0, 1]. One property covers all seven model branches; the existing
tests pin exact values at one or two fixed ``t`` per model and never assert
monotonicity across a sweep, so a sign error in any single branch's log-space
algebra that still matched at the fixed point would slip through.
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from simace.core.compute_hazard_terms import compute_hazard_terms

_pos = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)


@st.composite
def _model_and_params(draw):
    model = draw(
        st.sampled_from(
            ["weibull", "exponential", "gompertz", "lognormal", "loglogistic", "gamma", "first_passage"]
        )
    )
    if model == "weibull":
        params = {"scale": draw(_pos), "rho": draw(_pos)}
    elif model == "exponential":
        params = {"rate": draw(_pos)}
    elif model == "gompertz":
        params = {"rate": draw(_pos), "gamma": draw(st.floats(-0.1, 0.1, allow_nan=False, allow_infinity=False))}
    elif model == "lognormal":
        params = {
            "mu": draw(st.floats(-2.0, 5.0, allow_nan=False, allow_infinity=False)),
            "sigma": draw(st.floats(0.1, 3.0, allow_nan=False, allow_infinity=False)),
        }
    elif model == "loglogistic":
        params = {"scale": draw(_pos), "shape": draw(_pos)}
    elif model == "gamma":
        params = {"shape": draw(st.floats(0.2, 20.0, allow_nan=False, allow_infinity=False)), "scale": draw(_pos)}
    else:  # first_passage
        drift = draw(st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False).filter(lambda x: abs(x) > 1e-3))
        params = {"drift": drift, "shape": draw(_pos)}
    return model, params


@st.composite
def _t_grid(draw):
    ts = draw(
        st.lists(
            st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=20,
            unique=True,
        )
    )
    return np.sort(np.asarray(ts, dtype=np.float64))


@settings(deadline=None, max_examples=150)
@given(mp=_model_and_params(), t=_t_grid())
def test_cumulative_hazard_nonneg_nondecreasing(mp, t):
    model, params = mp
    _const, h_base = compute_hazard_terms(model, t, params)

    # Restrict to the well-behaved domain: at extreme params/t the log-survival
    # can underflow to +/-inf. The shape invariant is asserted where defined.
    assume(np.all(np.isfinite(h_base)))

    assert np.all(h_base >= -1e-9)
    # non-decreasing, with a magnitude-scaled tolerance for float round-off
    tol = 1e-7 * np.maximum(1.0, np.abs(h_base[1:]))
    assert np.all(np.diff(h_base) >= -tol)
