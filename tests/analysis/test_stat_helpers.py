"""The small derived statistics plotting reads from :mod:`simace.analysis.stats` (#90)."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from simace.analysis.stats.censoring import confusion_rates
from simace.analysis.stats.heritability import dempster_lerner_factor, ge_h2_metrics
from simace.analysis.stats.incidence import age_at_incidence_fraction, per_1000_person_years


def test_confusion_rates_and_an_empty_margin():
    rates = confusion_rates(tp=30, fn=10, fp=5, tn=55)
    assert (rates.sensitivity, rates.specificity, rates.ppv, rates.npv) == (0.75, 55 / 60, 30 / 35, 55 / 65)
    assert math.isnan(confusion_rates(tp=0, fn=0, fp=1, tn=1).sensitivity)


@pytest.mark.parametrize("K", [0.01, 0.1, 0.5])
def test_dempster_lerner_factor_is_k_one_minus_k_over_z_squared(K):
    z = norm.pdf(norm.ppf(1 - K))
    assert dempster_lerner_factor(K) == pytest.approx(K * (1 - K) / z**2, rel=1e-12)


def test_dempster_lerner_factor_clips_k_into_the_open_interval():
    assert dempster_lerner_factor(0.0) == dempster_lerner_factor(1e-3)


def test_zero_additive_variance_snp_like_is_nan():
    derived = ge_h2_metrics(var_a=0.0, var_liability=1.0, cov_a_non_genetic=0.0, n=10)
    assert np.isnan(derived.h2_snp_like)


def test_uncorrelated_components_give_the_baseline_identity():
    derived = ge_h2_metrics(var_a=0.5, var_liability=1.0, cov_a_non_genetic=0.0, n=1000)
    assert derived.ge_cov_fraction == 0.0
    assert derived.h2_realized_A == 0.5
    assert derived.h2_snp_like == 0.5
    assert derived.null_sd > 0.0


def test_age_at_incidence_fraction_takes_the_first_age_reaching_the_target():
    ages = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    curve = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    assert age_at_incidence_fraction(ages, curve, 0.5) == 20.0
    assert age_at_incidence_fraction(ages, curve, 0.6) == 30.0  # 0.24 first reached at 30
    assert age_at_incidence_fraction(ages, curve, 2.0) == 40.0  # never reached: the last age


def test_per_1000_person_years():
    assert per_1000_person_years(5, 2500) == 2.0
