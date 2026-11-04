"""Fixed-draw monotonicity of event times in liability (frailty, adult.ltm) and prevalence (adult.ltm).

With ``standardize="none"``, a fixed seed and ``beta > 0``, raising any
individual's liability never delays their event: frailty divides a fixed
``-log U`` by a larger ``exp(beta * L)``, and adult.ltm lowers the case
CIR, so a case stays a case and onsets no later. Raising adult.ltm's
prevalence K lowers the threshold and moves every case's onset earlier.
first_passage, adult.cox, cure_frailty and simple_ltm are excluded: their
draws or ranks shift with the inputs.

Every increase here is at least ``_MIN_STEP``. One-ulp steps are not
monotone: scipy's ``erfc``, the lognormal kernel's ``ndtri`` approximation and
scipy's gamma ``isf`` each move the wrong way by an ulp or so.
``tools/probe_event_time_monotonicity.py`` (seed 0; 1.8M liability pairs per
frailty distribution, 4.8M for adult.ltm, a third and a half of them one ulp
apart) measures a worst relative increase of 6.25e-13 (lognormal), 1.26e-15
(gamma) and 6.63e-14 (adult.ltm), and none for the closed-form kernels. A
``_MIN_STEP`` shift moves the inputs to every one of those functions by many
orders of magnitude more than their error, so the comparison stays exact.

The prevalence step has a second floor. ``logistic_cif_onset`` clips the CIR
to ``K - 1e-10``, so a case whose CIR lies within 1e-10 below K onsets later
(by 8.33e-10 at K = 0.3) when K rises by about 1e-10. A ``_MIN_STEP`` rise is
far outside that window.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from simace.phenotype.models.adult import AdultModel
from simace.phenotype.models.frailty import FrailtyModel
from tests.conftest import HAZARD_PARAM_BOUNDS, hazard_params

_MIN_STEP = 1e-6
_LIABILITY = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
_STEP = st.one_of(st.just(0.0), st.floats(min_value=_MIN_STEP, max_value=3.0))
_BETA = st.floats(min_value=0.1, max_value=3.0)
_BETA_SEX = st.one_of(st.just(0.0), st.floats(min_value=-1.0, max_value=1.0))
_SEEDS = st.integers(min_value=0, max_value=2**31 - 1)


@st.composite
def _population(draw) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Liability, a non-negative per-row step (zero or at least ``_MIN_STEP``), sex and generation."""
    n = draw(st.integers(min_value=1, max_value=40))
    liability = draw(hnp.arrays(np.float64, n, elements=_LIABILITY))
    step = draw(hnp.arrays(np.float64, n, elements=_STEP))
    sex = draw(hnp.arrays(np.int64, n, elements=st.integers(0, 1)))
    generation = draw(hnp.arrays(np.int64, n, elements=st.integers(0, 3)))
    return liability, step, sex, generation


def _simulate(model, liability, seed, sex, generation) -> np.ndarray:
    return model.simulate(liability, seed=seed, standardize="none", sex=sex, generation=generation)


@given(
    distribution=st.sampled_from(sorted(HAZARD_PARAM_BOUNDS)),
    data=st.data(),
    population=_population(),
    beta=_BETA,
    beta_sex=_BETA_SEX,
    seed=_SEEDS,
)
def test_frailty_time_is_non_increasing_in_liability(distribution, data, population, beta, beta_sex, seed) -> None:
    liability, step, sex, generation = population
    model = FrailtyModel(
        distribution=distribution, hazard_params=data.draw(hazard_params(distribution)), beta=beta, beta_sex=beta_sex
    )
    t = _simulate(model, liability, seed, sex, generation)
    t_raised = _simulate(model, liability + step, seed, sex, generation)
    assert np.all(t_raised <= t)
    np.testing.assert_array_equal(t_raised[step == 0], t[step == 0])


@given(
    population=_population(),
    K=st.floats(min_value=0.01, max_value=0.99),
    beta=_BETA,
    beta_sex=_BETA_SEX,
    seed=_SEEDS,
)
def test_adult_ltm_onset_is_non_increasing_in_liability(population, K, beta, beta_sex, seed) -> None:
    liability, step, sex, generation = population
    model = AdultModel(method="ltm", prevalence=K, beta=beta, beta_sex=beta_sex)
    t = _simulate(model, liability, seed, sex, generation)
    t_raised = _simulate(model, liability + step, seed, sex, generation)
    assert np.all(t_raised <= t)
    np.testing.assert_array_equal(t_raised[step == 0], t[step == 0])


@given(
    population=_population(),
    K=st.floats(min_value=0.01, max_value=0.98),
    data=st.data(),
    beta=_BETA,
    beta_sex=_BETA_SEX,
    seed=_SEEDS,
)
def test_adult_ltm_onset_is_non_increasing_in_prevalence(population, K, data, beta, beta_sex, seed) -> None:
    liability, _, sex, generation = population
    K_raised = data.draw(st.floats(min_value=K + _MIN_STEP, max_value=0.99))
    t = _simulate(
        AdultModel(method="ltm", prevalence=K, beta=beta, beta_sex=beta_sex), liability, seed, sex, generation
    )
    t_raised = _simulate(
        AdultModel(method="ltm", prevalence=K_raised, beta=beta, beta_sex=beta_sex), liability, seed, sex, generation
    )
    assert np.all(t_raised <= t)
