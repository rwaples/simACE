"""Construction / validation tests for CureFrailtyModel."""

import argparse

import numpy as np
import pytest

from simace.phenotyping.models import CureFrailtyModel

GOMPERTZ = {"rate": 0.0133, "gamma": 0.2019}


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="unknown cure_frailty distribution"):
        CureFrailtyModel(distribution="bogus", hazard_params={}, prevalence=0.1)


def test_missing_hazard_params_raises():
    with pytest.raises(ValueError, match="missing required hazard params"):
        CureFrailtyModel(distribution="gompertz", hazard_params={"rate": 0.01}, prevalence=0.1)


def test_from_config_reads_prevalence_and_distribution():
    params = {
        "phenotype_params1": {"distribution": "gompertz", **GOMPERTZ},
        "prevalence1": 0.10,
        "beta1": 1.0,
    }
    m = CureFrailtyModel.from_config(params, trait_num=1)
    assert m.distribution == "gompertz"
    assert m.hazard_params == GOMPERTZ
    assert m.prevalence == 0.10


def test_to_params_dict_excludes_prevalence():
    m = CureFrailtyModel(distribution="gompertz", hazard_params=GOMPERTZ, prevalence=0.1)
    assert m.to_params_dict() == {"distribution": "gompertz", **GOMPERTZ}


def _parser_with_all_models():
    from simace.phenotyping.models import MODELS

    parser = argparse.ArgumentParser()
    for trait in (1, 2):
        parser.add_argument(f"--phenotype-model{trait}", default="cure_frailty")
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
            "cure_frailty",
            "--cure-frailty-distribution1",
            "gompertz",
            "--cure-frailty-rate1",
            "0.0133",
            "--cure-frailty-gamma1",
            "0.2019",
            "--cure-frailty-prevalence1",
            "0.1",
            "--phenotype-model2",
            "cure_frailty",
            "--cure-frailty-distribution2",
            "gompertz",
            "--cure-frailty-rate2",
            "0.0133",
            "--cure-frailty-gamma2",
            "0.2019",
            "--cure-frailty-prevalence2",
            "0.2",
        ]
    )
    m = CureFrailtyModel.from_cli(args, 1)
    assert m.distribution == "gompertz"
    assert m.prevalence == 0.1


def test_from_cli_rejects_foreign_first_passage_flag():
    parser = _parser_with_all_models()
    args = parser.parse_args(
        [
            "--phenotype-model1",
            "cure_frailty",
            "--cure-frailty-distribution1",
            "gompertz",
            "--cure-frailty-rate1",
            "0.01",
            "--cure-frailty-gamma1",
            "0.2",
            "--cure-frailty-prevalence1",
            "0.1",
            "--first-passage-drift1",
            "-0.5",  # foreign
            "--phenotype-model2",
            "cure_frailty",
            "--cure-frailty-distribution2",
            "gompertz",
            "--cure-frailty-rate2",
            "0.01",
            "--cure-frailty-gamma2",
            "0.2",
            "--cure-frailty-prevalence2",
            "0.1",
        ]
    )
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*--first-passage-drift1"):
        CureFrailtyModel.from_cli(args, 1)


def test_simulate_runs():
    m = CureFrailtyModel(distribution="gompertz", hazard_params=GOMPERTZ, prevalence=0.1, beta=1.0)
    liability = np.random.default_rng(0).standard_normal(500)
    t = m.simulate(
        liability=liability,
        seed=42,
        standardize=True,
        sex=np.zeros(500),
        generation=np.zeros(500, dtype=int),
    )
    assert t.shape == (500,)
    cases = t < 1e6
    assert cases.sum() > 0
