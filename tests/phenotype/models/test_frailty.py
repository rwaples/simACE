"""Construction / validation tests for FrailtyModel.

Construction, config and CLI surfaces are tested here; shared behavioural
contracts (sex effects, liability ordering) live in ``test_behaviour.py``.
"""

import pytest

from simace.phenotype.models import FrailtyModel

WEIBULL = {"distribution": "weibull", "scale": 316.228, "rho": 2.0}


def test_construct_minimal():
    m = FrailtyModel(distribution="weibull", hazard_params={"scale": 100.0, "rho": 2.0}, beta=1.0)
    assert m.name == "frailty"
    assert m.beta == 1.0
    assert m.beta_sex == 0.0


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="unknown frailty distribution"):
        FrailtyModel(distribution="bogus", hazard_params={}, beta=1.0)


def test_missing_hazard_params_raises():
    with pytest.raises(ValueError, match="missing required hazard params"):
        FrailtyModel(distribution="weibull", hazard_params={"scale": 100.0}, beta=1.0)


def test_inf_beta_raises():
    with pytest.raises(ValueError, match="beta must be finite"):
        FrailtyModel(distribution="weibull", hazard_params={"scale": 100.0, "rho": 2.0}, beta=float("inf"))


def test_exponential_accepts_either_rate_or_scale():
    FrailtyModel(distribution="exponential", hazard_params={"rate": 0.01})
    FrailtyModel(distribution="exponential", hazard_params={"scale": 100.0})


def test_from_config_reads_phenotype_params():
    params = {
        "phenotype_params1": {"distribution": "weibull", "scale": 316.228, "rho": 2.0},
        "beta1": 1.5,
        "beta_sex1": 0.2,
    }
    m = FrailtyModel.from_config(params, trait_num=1)
    assert m.distribution == "weibull"
    assert m.hazard_params == {"scale": 316.228, "rho": 2.0}
    assert m.beta == 1.5
    assert m.beta_sex == 0.2


def test_from_config_missing_distribution_traitful_message():
    params = {"phenotype_params1": {"scale": 316.228, "rho": 2.0}, "beta1": 1.0}
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*'distribution'"):
        FrailtyModel.from_config(params, trait_num=1)


def test_from_config_rejects_prevalence():
    params = {"phenotype_params1": {"distribution": "weibull", "scale": 316.228, "rho": 2.0, "prevalence": 0.1}}
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*unknown key\(s\) \['prevalence'\]"):
        FrailtyModel.from_config({**params, "beta1": 1.0}, trait_num=1)
