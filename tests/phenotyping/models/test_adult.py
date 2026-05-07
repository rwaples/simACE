"""Construction / validation tests for AdultModel."""

import argparse

import numpy as np
import pytest

from simace.phenotyping.models import AdultModel


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


def test_to_params_dict_round_trips_prevalence():
    m = AdultModel(method="cox", prevalence=0.2, cip_x0=55.0, cip_k=0.25)
    assert m.to_params_dict() == {
        "method": "cox",
        "cip_x0": 55.0,
        "cip_k": 0.25,
        "prevalence": 0.2,
    }


def _parser_with_all_models():
    from simace.phenotyping.models import MODELS

    parser = argparse.ArgumentParser()
    for trait in (1, 2):
        parser.add_argument(f"--phenotype-model{trait}", default="adult")
        parser.add_argument(f"--beta{trait}", type=float, default=1.0)
        parser.add_argument(f"--beta-sex{trait}", type=float, default=0.0)
        for cls in MODELS.values():
            cls.add_cli_args(parser, trait)
    return parser


def test_from_cli_happy_path():
    parser = _parser_with_all_models()
    args = parser.parse_args(
        [
            "--phenotype-model1",
            "adult",
            "--adult-method1",
            "ltm",
            "--adult-cip-x0-1",
            "60",
            "--adult-cip-k-1",
            "0.3",
            "--adult-prevalence1",
            "0.1",
            "--phenotype-model2",
            "adult",
            "--adult-method2",
            "cox",
            "--adult-prevalence2",
            "0.2",
        ]
    )
    m1 = AdultModel.from_cli(args, 1)
    assert m1.method == "ltm"
    assert m1.cip_x0 == 60.0
    assert m1.prevalence == 0.1
    # cip_x0 / cip_k default through the from_cli when not supplied for trait 2
    m2 = AdultModel.from_cli(args, 2)
    assert m2.cip_x0 == 50.0


def test_from_cli_rejects_foreign_frailty_flag():
    parser = _parser_with_all_models()
    args = parser.parse_args(
        [
            "--phenotype-model1",
            "adult",
            "--adult-method1",
            "ltm",
            "--adult-prevalence1",
            "0.1",
            "--frailty-rho1",
            "2.0",  # foreign
            "--phenotype-model2",
            "adult",
            "--adult-method2",
            "ltm",
            "--adult-prevalence2",
            "0.1",
        ]
    )
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*--frailty-rho1"):
        AdultModel.from_cli(args, 1)


def test_simulate_runs():
    m = AdultModel(method="ltm", prevalence=0.1, beta=1.0)
    liability = np.random.default_rng(0).standard_normal(200)
    t = m.simulate(
        liability=liability,
        seed=42,
        standardize=True,
        sex=np.zeros(200),
        generation=np.zeros(200, dtype=int),
    )
    assert t.shape == (200,)
    assert np.all(np.isfinite(t))


def test_cli_flag_attrs_set():
    attrs = AdultModel.cli_flag_attrs(2)
    assert attrs == {
        "adult_method2",
        "adult_cip_x0_2",
        "adult_cip_k_2",
        "adult_prevalence2",
        "adult_standardize_hazard2",
    }
