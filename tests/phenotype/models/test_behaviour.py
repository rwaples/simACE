"""Behavioural contracts shared by the phenotype models.

Each test is parametrized over the models it applies to and asserts a
direction or a rate: sex effects move onset the right way, higher liability
onsets earlier, threshold models realise their configured prevalence.
Construction, config and CLI surfaces live in the per-model files.
"""

import numpy as np
import pytest

from simace.phenotype.models import AdultModel, CureFrailtyModel, FirstPassageModel, FrailtyModel

WEIBULL = {"scale": 316.228, "rho": 2.0}
GOMPERTZ = {"rate": 0.0133, "gamma": 0.2019}
SENTINEL = 1e6


def frailty(**kw):
    return FrailtyModel(distribution="weibull", hazard_params=WEIBULL, **kw)


def first_passage(**kw):
    return FirstPassageModel(drift=-0.01, shape=100.0, **kw)


def adult_ltm(**kw):
    return AdultModel(method="ltm", prevalence=kw.pop("prevalence", 0.2), **kw)


def adult_cox(**kw):
    return AdultModel(method="cox", prevalence=kw.pop("prevalence", 0.2), **kw)


def cure_frailty(**kw):
    return CureFrailtyModel(distribution="gompertz", hazard_params=GOMPERTZ, prevalence=kw.pop("prevalence", 0.2), **kw)


ALL_MODELS = [frailty, first_passage, adult_ltm, adult_cox, cure_frailty]
PREVALENCE_MODELS = [adult_ltm, adult_cox, cure_frailty]


def _simulate(model, liability, *, sex=None, standardize=True, generation=None, seed=42):
    if generation is None:
        generation = np.zeros(len(liability), dtype=int)
    return model.simulate(liability=liability, seed=seed, standardize=standardize, sex=sex, generation=generation)


def _half_sexes(n):
    return np.array([0.0] * (n // 2) + [1.0] * (n // 2))


def _case_mean_by_sex(t, n):
    cases = t < SENTINEL
    female = t[: n // 2][cases[: n // 2]]
    male = t[n // 2 :][cases[n // 2 :]]
    return female.mean(), male.mean()


@pytest.mark.parametrize("make", ALL_MODELS)
def test_beta_sex_zero_matches_no_sex(make):
    rng = np.random.default_rng(0)
    liability = rng.standard_normal(500)
    sex = rng.integers(0, 2, size=500).astype(float)
    t_no_sex = _simulate(make(beta=1.0), liability)
    t_zero = _simulate(make(beta=1.0, beta_sex=0.0), liability, sex=sex)
    np.testing.assert_array_equal(t_no_sex, t_zero)


@pytest.mark.parametrize("make", ALL_MODELS)
@pytest.mark.parametrize("beta_sex", [0.5, -0.5])
def test_beta_sex_sign_orders_onset_by_sex(make, beta_sex):
    n = 20000
    liability = np.random.default_rng(0).standard_normal(n)
    sex = _half_sexes(n)
    t = _simulate(make(beta=1.0, beta_sex=beta_sex), liability, sex=sex)
    female_mean, male_mean = _case_mean_by_sex(t, n)
    if beta_sex > 0:
        assert male_mean < female_mean
    else:
        assert female_mean < male_mean


@pytest.mark.parametrize("make", ALL_MODELS)
def test_higher_liability_earlier_onset_among_cases(make):
    n = 50000
    liability = np.random.default_rng(0).standard_normal(n)
    t = _simulate(make(beta=2.0), liability)
    cases = t < SENTINEL
    case_L, case_t = liability[cases], t[cases]
    high = case_L > np.percentile(case_L, 75)
    low = case_L < np.percentile(case_L, 25)
    assert case_t[high].mean() < case_t[low].mean()


@pytest.mark.parametrize("make", PREVALENCE_MODELS)
def test_case_rate_matches_scalar_prevalence(make):
    n = 50000
    liability = np.random.default_rng(0).standard_normal(n)
    t = _simulate(make(prevalence=0.10), liability)
    assert abs(np.mean(t < SENTINEL) - 0.10) < 0.02
    assert np.all(t[t >= SENTINEL - 1] == SENTINEL)
    assert np.all(t[t < SENTINEL] > 0)


@pytest.mark.parametrize("make", PREVALENCE_MODELS)
def test_case_rate_matches_sex_specific_prevalence(make):
    n = 50000
    liability = np.random.default_rng(0).standard_normal(n)
    sex = _half_sexes(n)
    prev = {"female": 0.08, "male": 0.15}
    t = _simulate(make(prevalence=prev), liability, sex=sex)
    cases = t < SENTINEL
    assert abs(cases[: n // 2].mean() - prev["female"]) < 0.02
    assert abs(cases[n // 2 :].mean() - prev["male"]) < 0.02


@pytest.mark.parametrize("make", PREVALENCE_MODELS)
def test_case_rate_matches_per_generation_prevalence(make):
    gen_prev = {0: 0.05, 1: 0.10, 2: 0.20}
    generation = np.repeat([0, 1, 2], 20000)
    liability = np.random.default_rng(0).standard_normal(len(generation))
    t = _simulate(make(prevalence=gen_prev), liability, generation=generation)
    cases = t < SENTINEL
    for gen, expected in gen_prev.items():
        assert abs(cases[generation == gen].mean() - expected) < 0.02, f"gen {gen}"


@pytest.mark.parametrize("make", [adult_ltm, adult_cox])
def test_higher_liability_more_likely_a_case(make):
    n = 50000
    liability = np.random.default_rng(0).standard_normal(n)
    t = _simulate(make(prevalence=0.10), liability)
    cases = t < SENTINEL
    assert cases[liability > 1.0].mean() > cases[liability < -1.0].mean()


@pytest.mark.parametrize(("make", "center"), [(adult_ltm, np.mean), (adult_cox, np.median)])
def test_case_ages_centred_on_cip_x0(make, center):
    n = 50000
    liability = np.random.default_rng(0).standard_normal(n)
    t = _simulate(make(prevalence=0.20, cip_x0=60.0), liability)
    assert abs(center(t[t < SENTINEL]) - 60.0) < 3.0


@pytest.mark.parametrize(
    ("distribution", "params"),
    [
        ("weibull", WEIBULL),
        ("exponential", {"rate": 0.01}),
        ("gompertz", GOMPERTZ),
        ("lognormal", {"mu": 4.0, "sigma": 0.8}),
        ("loglogistic", {"scale": 60.0, "shape": 4.0}),
        ("gamma", {"shape": 2.0, "scale": 1000.0}),
    ],
)
def test_cure_frailty_accepts_every_baseline(distribution, params):
    liability = np.random.default_rng(0).standard_normal(5000)
    m = CureFrailtyModel(distribution=distribution, hazard_params=params, prevalence=0.10, beta=1.0)
    t = _simulate(m, liability)
    cases = t[t < SENTINEL]
    assert len(cases) > 0
    assert np.all(cases > 0)


def test_first_passage_positive_drift_higher_liability_hits_more_often():
    n = 20000
    m = FirstPassageModel(drift=0.05, shape=100.0, beta=1.0)
    censored_high = np.mean(_simulate(m, np.full(n, 2.0), standardize=False) >= SENTINEL)
    censored_low = np.mean(_simulate(m, np.full(n, -2.0), standardize=False) >= SENTINEL)
    assert censored_high < censored_low
