"""Construction / validation tests for AdultModel."""

import pytest

from simace.phenotype.models import AdultModel


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown adult method"):
        AdultModel(method="oops", prevalence=0.1)


def test_inf_beta_raises():
    with pytest.raises(ValueError, match="beta must be finite"):
        AdultModel(method="ltm", prevalence=0.1, beta=float("inf"))


def test_from_config_reads_phenotype_params():
    params = {
        "phenotype_params1": {"method": "ltm", "cip_x0": 60.0, "cip_k": 0.3, "prevalence": 0.15},
        "beta1": 1.0,
    }
    m = AdultModel.from_config(params, trait_num=1)
    assert m.method == "ltm"
    assert m.cip_x0 == 60.0
    assert m.cip_k == 0.3
    assert m.prevalence == 0.15


def test_from_config_method_missing_traitful_message():
    params = {"phenotype_params1": {"cip_x0": 60.0, "prevalence": 0.1}, "beta1": 1.0}
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*'method'"):
        AdultModel.from_config(params, trait_num=1)


def test_from_config_prevalence_missing_traitful_message():
    params = {"phenotype_params1": {"method": "ltm"}, "beta1": 1.0}
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*'prevalence'"):
        AdultModel.from_config(params, trait_num=1)
