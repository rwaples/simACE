"""Construction / validation / simulation tests for SimpleLtmModel."""

import argparse

import numpy as np
import pytest

from simace.phenotype.models import SimpleLtmModel

_FIXED = {"kind": "fixed", "age": 30.0}
_NORMAL = {"kind": "normal", "mean": 35.0, "sd": 8.0}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_onset_kind_raises():
    with pytest.raises(ValueError, match=r"onset\.kind must be one of"):
        SimpleLtmModel(prevalence=0.1, onset={"kind": "oops"})


def test_fixed_onset_requires_age():
    with pytest.raises(ValueError, match="requires 'age'"):
        SimpleLtmModel(prevalence=0.1, onset={"kind": "fixed"})


def test_normal_onset_requires_mean_sd():
    with pytest.raises(ValueError, match="requires"):
        SimpleLtmModel(prevalence=0.1, onset={"kind": "normal", "mean": 30.0})


def test_normal_onset_rejects_nonpositive_sd():
    with pytest.raises(ValueError, match="sd must be > 0"):
        SimpleLtmModel(prevalence=0.1, onset={"kind": "normal", "mean": 30.0, "sd": 0.0})


def test_inf_beta_raises():
    with pytest.raises(ValueError, match="beta must be finite"):
        SimpleLtmModel(prevalence=0.1, onset=_FIXED, beta=float("inf"))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_config_reads_phenotype_params():
    params = {
        "phenotype_params1": {"prevalence": 0.15, "onset": {"kind": "fixed", "age": 42.0}},
        "beta1": 1.0,
    }
    m = SimpleLtmModel.from_config(params, trait_num=1)
    assert m.prevalence == 0.15
    assert m.onset == {"kind": "fixed", "age": 42.0}


def test_from_config_prevalence_missing_traitful_message():
    params = {"phenotype_params1": {"onset": _FIXED}, "beta1": 1.0}
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*'prevalence'"):
        SimpleLtmModel.from_config(params, trait_num=1)


def test_from_config_onset_missing_traitful_message():
    params = {"phenotype_params1": {"prevalence": 0.1}, "beta1": 1.0}
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*'onset'"):
        SimpleLtmModel.from_config(params, trait_num=1)


def test_to_params_dict_round_trips():
    m = SimpleLtmModel(prevalence=0.2, onset=dict(_NORMAL))
    out = m.to_params_dict()
    assert out == {"prevalence": 0.2, "onset": {"kind": "normal", "mean": 35.0, "sd": 8.0}}
    # round-trip back through from_config
    rebuilt = SimpleLtmModel.from_config({"phenotype_params1": out, "beta1": 1.0}, trait_num=1)
    assert rebuilt.onset == m.onset
    assert rebuilt.prevalence == m.prevalence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser_with_all_models():
    from simace.phenotype.models import MODELS

    parser = argparse.ArgumentParser()
    for trait in (1, 2):
        parser.add_argument(f"--phenotype-model{trait}", default="simple_ltm")
        parser.add_argument(f"--beta{trait}", type=float, default=1.0)
        parser.add_argument(f"--beta-sex{trait}", type=float, default=0.0)
        for cls in MODELS.values():
            cls.add_cli_args(parser, trait)
    return parser


def test_from_cli_fixed_and_normal():
    parser = _parser_with_all_models()
    args = parser.parse_args(
        [
            "--phenotype-model1",
            "simple_ltm",
            "--simple-ltm-prevalence1",
            "0.1",
            "--simple-ltm-onset-kind-1",
            "fixed",
            "--simple-ltm-onset-age-1",
            "30",
            "--phenotype-model2",
            "simple_ltm",
            "--simple-ltm-prevalence2",
            "0.2",
            "--simple-ltm-onset-kind-2",
            "normal",
            "--simple-ltm-onset-mean-2",
            "35",
            "--simple-ltm-onset-sd-2",
            "8",
        ]
    )
    m1 = SimpleLtmModel.from_cli(args, 1)
    assert m1.prevalence == 0.1
    assert m1.onset == {"kind": "fixed", "age": 30.0}
    m2 = SimpleLtmModel.from_cli(args, 2)
    assert m2.onset == {"kind": "normal", "mean": 35.0, "sd": 8.0}


def test_from_cli_missing_prevalence_raises():
    parser = _parser_with_all_models()
    args = parser.parse_args(
        ["--phenotype-model1", "simple_ltm", "--simple-ltm-onset-kind-1", "fixed", "--simple-ltm-onset-age-1", "30"]
    )
    with pytest.raises(ValueError, match=r"--simple-ltm-prevalence1 is required"):
        SimpleLtmModel.from_cli(args, 1)


def test_from_cli_missing_onset_age_raises():
    parser = _parser_with_all_models()
    args = parser.parse_args(
        ["--phenotype-model1", "simple_ltm", "--simple-ltm-prevalence1", "0.1", "--simple-ltm-onset-kind-1", "fixed"]
    )
    with pytest.raises(ValueError, match=r"--simple-ltm-onset-age-1 is required"):
        SimpleLtmModel.from_cli(args, 1)


def test_from_cli_rejects_foreign_frailty_flag():
    parser = _parser_with_all_models()
    args = parser.parse_args(
        [
            "--phenotype-model1",
            "simple_ltm",
            "--simple-ltm-prevalence1",
            "0.1",
            "--simple-ltm-onset-kind-1",
            "fixed",
            "--simple-ltm-onset-age-1",
            "30",
            "--frailty-rho1",
            "2.0",  # foreign
        ]
    )
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*--frailty-rho1"):
        SimpleLtmModel.from_cli(args, 1)


def test_cli_flag_attrs_set():
    assert SimpleLtmModel.cli_flag_attrs(2) == {
        "simple_ltm_prevalence2",
        "simple_ltm_onset_kind_2",
        "simple_ltm_onset_age_2",
        "simple_ltm_onset_mean_2",
        "simple_ltm_onset_sd_2",
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def test_fixed_onset_cases_share_age_controls_sentinel():
    liability = np.random.default_rng(0).standard_normal(5000)
    generation = np.zeros(5000, dtype=int)
    m = SimpleLtmModel(prevalence=0.1, onset={"kind": "fixed", "age": 30.0})
    t = m.simulate(liability=liability, seed=42, standardize="global", sex=np.zeros(5000), generation=generation)
    cases = t < 1e6
    assert np.allclose(t[cases], 30.0)
    assert np.all(t[~cases] == 1e6)


def test_normal_onset_distribution_and_reproducibility():
    liability = np.random.default_rng(1).standard_normal(20000)
    generation = np.zeros(20000, dtype=int)
    m = SimpleLtmModel(prevalence=0.3, onset={"kind": "normal", "mean": 35.0, "sd": 8.0})
    kw = {"liability": liability, "standardize": "global", "sex": np.zeros(20000), "generation": generation}
    t1 = m.simulate(seed=7, **kw)
    t2 = m.simulate(seed=7, **kw)
    np.testing.assert_array_equal(t1, t2)  # reproducible under fixed seed
    cases = t1 < 1e6
    assert 30.0 < t1[cases].mean() < 40.0
    assert 6.0 < t1[cases].std() < 10.0


def test_realised_case_fraction_matches_prevalence():
    liability = np.random.default_rng(2).standard_normal(50000)
    generation = np.zeros(50000, dtype=int)
    m = SimpleLtmModel(prevalence=0.1, onset=_FIXED)
    t = m.simulate(liability=liability, seed=0, standardize="global", sex=np.zeros(50000), generation=generation)
    assert 0.09 < (t < 1e6).mean() < 0.11


def test_large_sd_draws_stay_clipped():
    liability = np.random.default_rng(3).standard_normal(5000)
    generation = np.zeros(5000, dtype=int)
    m = SimpleLtmModel(prevalence=0.5, onset={"kind": "normal", "mean": 0.0, "sd": 1e9})
    t = m.simulate(liability=liability, seed=5, standardize="global", sex=np.zeros(5000), generation=generation)
    assert np.all(t >= 0.01)
    assert np.all(t <= 1e6)
