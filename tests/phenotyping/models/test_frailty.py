"""Construction / validation tests for FrailtyModel.

These complement ``tests/phenotyping/test_phenotype.py``, which exercises
the simulate path through compatibility shims. Here we test the new
class-level surface directly: validation, ``from_config``, ``from_cli``,
and ``to_params_dict``.
"""

import argparse

import numpy as np
import pytest

from simace.phenotyping.hazards import HAZARD_FLAG_ROOTS
from simace.phenotyping.models import FrailtyModel

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


def test_to_params_dict_round_trips_through_from_config():
    src = {"phenotype_params2": dict(WEIBULL), "beta2": 1.0}
    m = FrailtyModel.from_config(src, trait_num=2)
    rt = {"phenotype_params2": m.to_params_dict(), "beta2": m.beta}
    m2 = FrailtyModel.from_config(rt, trait_num=2)
    assert m == m2


def _build_parser():
    parser = argparse.ArgumentParser()
    for trait in (1, 2):
        parser.add_argument(f"--phenotype-model{trait}", default="frailty")
        parser.add_argument(f"--beta{trait}", type=float, default=1.0)
        parser.add_argument(f"--beta-sex{trait}", type=float, default=0.0)
    # Register every model's flags so the foreign-flag check has something
    # to compare against.
    from simace.phenotyping.models import MODELS

    for trait in (1, 2):
        for cls in MODELS.values():
            cls.add_cli_args(parser, trait)
    return parser


def test_from_cli_happy_path():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--phenotype-model1",
            "frailty",
            "--frailty-distribution1",
            "weibull",
            "--frailty-scale1",
            "316.228",
            "--frailty-rho1",
            "2.0",
            "--beta1",
            "1.0",
            "--phenotype-model2",
            "frailty",
            "--frailty-distribution2",
            "weibull",
            "--frailty-scale2",
            "100.0",
            "--frailty-rho2",
            "1.5",
        ]
    )
    m1 = FrailtyModel.from_cli(args, 1)
    assert m1.distribution == "weibull"
    assert m1.hazard_params == {"scale": 316.228, "rho": 2.0}


def test_from_cli_missing_distribution_raises():
    parser = _build_parser()
    args = parser.parse_args(["--phenotype-model1", "frailty", "--phenotype-model2", "frailty"])
    with pytest.raises(ValueError, match=r"--frailty-distribution1 is required"):
        FrailtyModel.from_cli(args, 1)


def test_from_cli_rejects_foreign_adult_flag():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--phenotype-model1",
            "frailty",
            "--frailty-distribution1",
            "weibull",
            "--frailty-scale1",
            "316.228",
            "--frailty-rho1",
            "2.0",
            "--adult-method1",
            "ltm",  # foreign
            "--phenotype-model2",
            "frailty",
        ]
    )
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*--adult-method1"):
        FrailtyModel.from_cli(args, 1)


def test_simulate_finite_array():
    m = FrailtyModel(distribution="weibull", hazard_params={"scale": 100.0, "rho": 2.0})
    liability = np.random.default_rng(0).standard_normal(200)
    t = m.simulate(liability=liability, seed=42, standardize=True, sex=None, generation=np.zeros(200, dtype=int))
    assert t.shape == (200,)
    assert np.all(np.isfinite(t))
    assert np.all(t > 0)


def test_cli_flag_attrs_covers_all_hazard_roots():
    attrs = FrailtyModel.cli_flag_attrs(1)
    expected = {"frailty_distribution1"} | {f"frailty_{r}1" for r in HAZARD_FLAG_ROOTS}
    assert attrs == expected
