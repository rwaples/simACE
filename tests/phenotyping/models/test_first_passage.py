"""Construction / validation tests for FirstPassageModel."""

import argparse

import numpy as np
import pytest

from simace.phenotyping.models import FirstPassageModel


def test_zero_drift_raises():
    with pytest.raises(ValueError, match="drift must be non-zero"):
        FirstPassageModel(drift=0.0, shape=1.0)


def test_inf_beta_raises():
    with pytest.raises(ValueError, match="beta must be finite"):
        FirstPassageModel(drift=-0.5, shape=1.0, beta=float("inf"))


def test_from_config_reads_phenotype_params():
    params = {
        "phenotype_params1": {"drift": -0.5, "shape": 2.0},
        "beta1": 1.5,
    }
    m = FirstPassageModel.from_config(params, trait_num=1)
    assert m.drift == -0.5
    assert m.shape == 2.0
    assert m.beta == 1.5


def test_from_config_missing_key_traitful_message():
    params = {"phenotype_params1": {"drift": -0.5}, "beta1": 1.0}
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*'shape'"):
        FirstPassageModel.from_config(params, trait_num=1)


def test_to_params_dict():
    m = FirstPassageModel(drift=-0.3, shape=1.5)
    assert m.to_params_dict() == {"drift": -0.3, "shape": 1.5}


def _parser_with_all_models():
    from simace.phenotyping.models import MODELS

    parser = argparse.ArgumentParser()
    for trait in (1, 2):
        parser.add_argument(f"--phenotype-model{trait}", default="first_passage")
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
            "first_passage",
            "--first-passage-drift1",
            "-0.5",
            "--first-passage-shape1",
            "1.0",
            "--phenotype-model2",
            "first_passage",
            "--first-passage-drift2",
            "0.05",
            "--first-passage-shape2",
            "2.0",
        ]
    )
    m1 = FirstPassageModel.from_cli(args, 1)
    assert m1.drift == -0.5
    m2 = FirstPassageModel.from_cli(args, 2)
    assert m2.drift == 0.05


def test_from_cli_rejects_foreign_adult_prevalence():
    parser = _parser_with_all_models()
    args = parser.parse_args(
        [
            "--phenotype-model1",
            "first_passage",
            "--first-passage-drift1",
            "-0.5",
            "--first-passage-shape1",
            "1.0",
            "--adult-prevalence1",
            "0.1",  # foreign
            "--phenotype-model2",
            "first_passage",
            "--first-passage-drift2",
            "-0.5",
            "--first-passage-shape2",
            "1.0",
        ]
    )
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*--adult-prevalence1"):
        FirstPassageModel.from_cli(args, 1)


def test_simulate_runs():
    m = FirstPassageModel(drift=-0.5, shape=1.0, beta=1.0)
    liability = np.random.default_rng(0).standard_normal(200)
    t = m.simulate(
        liability=liability,
        seed=42,
        standardize=True,
        sex=None,
        generation=np.zeros(200, dtype=int),
    )
    assert t.shape == (200,)
    assert np.all(np.isfinite(t))


def test_cli_flag_attrs_set():
    assert FirstPassageModel.cli_flag_attrs(1) == {
        "first_passage_drift1",
        "first_passage_shape1",
    }
