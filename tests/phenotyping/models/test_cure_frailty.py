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
        "phenotype_params1": {"distribution": "gompertz", "prevalence": 0.10, **GOMPERTZ},
        "beta1": 1.0,
    }
    m = CureFrailtyModel.from_config(params, trait_num=1)
    assert m.distribution == "gompertz"
    assert m.hazard_params == GOMPERTZ
    assert m.prevalence == 0.10


def test_from_config_prevalence_missing_traitful_message():
    params = {"phenotype_params1": {"distribution": "gompertz", **GOMPERTZ}, "beta1": 1.0}
    with pytest.raises(ValueError, match=r"phenotype\.trait1.*'prevalence'"):
        CureFrailtyModel.from_config(params, trait_num=1)


def test_to_params_dict_includes_prevalence():
    m = CureFrailtyModel(distribution="gompertz", hazard_params=GOMPERTZ, prevalence=0.1)
    assert m.to_params_dict() == {"distribution": "gompertz", "prevalence": 0.1, **GOMPERTZ}


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


def test_hazard_invariant_to_liability_shift_under_global():
    """Under standardize='global', shifting liability by a constant ``c`` should
    not change case-onset times: z-scores absorb the shift, the threshold step
    sees the same cases, and the hazard step's ``z = exp(beta * z_score(L))``
    is shift-invariant.

    A pre-refactor bug in cure_frailty passed the *standardized* L to the
    hazard kernel (which expects raw L) — the kernel then computed
    ``exp(scaled_beta * (L_z - mean))`` instead of ``exp(scaled_beta * (L_raw - mean))``,
    introducing a constant ``-beta * mean / std`` offset that made onset times
    drift with mean shifts even under standardize='global'.
    """
    rng = np.random.default_rng(0)
    n = 5000
    base = rng.normal(0.0, 1.0, n)
    sex = np.zeros(n)
    generation = np.zeros(n, dtype=int)
    m = CureFrailtyModel(
        distribution="gompertz",
        hazard_params=GOMPERTZ,
        prevalence=0.3,
        beta=1.5,
    )
    t_centered = m.simulate(liability=base, seed=42, standardize="global", sex=sex, generation=generation)
    t_shifted = m.simulate(
        liability=base + 5.0,  # additive shift
        seed=42,
        standardize="global",
        sex=sex,
        generation=generation,
    )
    np.testing.assert_allclose(t_centered, t_shifted, rtol=1e-10, atol=1e-10)


def test_hazard_invariant_to_liability_scale_under_global():
    """Same invariance under multiplicative scale: shifting variance changes
    ``std`` but z-scores and ``beta / std`` compensate so that the realised
    hazard is unchanged. Pre-refactor cure_frailty had a 1/std² instead of
    1/std error that broke this invariance."""
    rng = np.random.default_rng(1)
    n = 5000
    base = rng.normal(0.0, 1.0, n)
    sex = np.zeros(n)
    generation = np.zeros(n, dtype=int)
    m = CureFrailtyModel(
        distribution="gompertz",
        hazard_params=GOMPERTZ,
        prevalence=0.3,
        beta=1.5,
    )
    t_unit = m.simulate(liability=base, seed=42, standardize="global", sex=sex, generation=generation)
    t_wide = m.simulate(
        liability=base * 3.0,  # std=3 instead of 1
        seed=42,
        standardize="global",
        sex=sex,
        generation=generation,
    )
    np.testing.assert_allclose(t_unit, t_wide, rtol=1e-10, atol=1e-10)
