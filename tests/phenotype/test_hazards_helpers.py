"""Unit tests for the hazard helper / CLI API.

The Numba kernels (``_nb_weibull``, ``_nb_exponential``, ...) are tested
indirectly through the phenotype models. This file targets the pure-Python
validation + argparse helpers.
"""

import argparse

import pytest

from simace.phenotype.hazards import (
    BASELINE_HAZARDS,
    BASELINE_PARAMS,
    add_hazard_cli_args,
    parse_hazard_cli,
    validate_hazard_params,
)

# ---------------------------------------------------------------------------
# validate_hazard_params
# ---------------------------------------------------------------------------


VALID_PARAMS = {
    "weibull": {"scale": 100.0, "rho": 2.0},
    "exponential": {"rate": 0.01},
    "gompertz": {"rate": 0.001, "gamma": 0.05},
    "lognormal": {"mu": 4.0, "sigma": 0.5},
    "loglogistic": {"scale": 60.0, "shape": 3.0},
    "gamma": {"shape": 2.0, "scale": 30.0},
}


@pytest.mark.parametrize("distribution", sorted(BASELINE_HAZARDS))
def test_validate_accepts_canonical_params(distribution):
    validate_hazard_params(distribution, VALID_PARAMS[distribution], "frailty")


def test_validate_accepts_exponential_scale_alternative():
    """Exponential accepts ``scale`` in place of ``rate``."""
    validate_hazard_params("exponential", {"scale": 100.0}, "frailty")


def test_validate_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="unknown frailty distribution"):
        validate_hazard_params("not_a_real_dist", {}, "frailty")


@pytest.mark.parametrize(
    ("distribution", "missing"),
    [
        ("weibull", "rho"),
        ("gompertz", "gamma"),
        ("lognormal", "sigma"),
    ],
)
def test_validate_rejects_missing_required_keys(distribution, missing):
    bad_params = {k: v for k, v in VALID_PARAMS[distribution].items() if k != missing}
    with pytest.raises(ValueError, match="missing required hazard params"):
        validate_hazard_params(distribution, bad_params, "frailty")


def test_baseline_params_dictionary_keys_match_baseline_hazards():
    """Sanity check on the registry — every distribution has a param spec."""
    assert set(BASELINE_PARAMS) == set(BASELINE_HAZARDS)


# ---------------------------------------------------------------------------
# add_hazard_cli_args + parse_hazard_cli round-trip
# ---------------------------------------------------------------------------


def _parser_for(trait: int, name: str = "frailty") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_hazard_cli_args(parser, trait, name=name)
    return parser


def test_round_trip_weibull():
    parser = _parser_for(trait=1)
    args = parser.parse_args(
        [
            "--frailty-distribution1",
            "weibull",
            "--frailty-scale1",
            "100.0",
            "--frailty-rho1",
            "2.0",
        ]
    )
    dist, params = parse_hazard_cli(args, trait=1, name="frailty")
    assert dist == "weibull"
    assert params == {"scale": 100.0, "rho": 2.0}


def test_round_trip_exponential_with_scale_alternative():
    """``parse_hazard_cli`` only returns the canonical keys for the distribution;
    the alternate ``scale`` form is honored by ``validate_hazard_params`` once
    the user constructs the dict directly, not by the CLI helper."""
    parser = _parser_for(trait=2)
    args = parser.parse_args(
        [
            "--frailty-distribution2",
            "exponential",
            "--frailty-rate2",
            "0.01",
        ]
    )
    dist, params = parse_hazard_cli(args, trait=2, name="frailty")
    assert dist == "exponential"
    assert params == {"rate": 0.01}


def test_missing_distribution_flag_raises():
    parser = _parser_for(trait=1)
    args = parser.parse_args([])
    with pytest.raises(ValueError, match="--frailty-distribution1 is required"):
        parse_hazard_cli(args, trait=1, name="frailty")


def test_missing_required_param_flag_raises():
    parser = _parser_for(trait=1)
    args = parser.parse_args(
        [
            "--frailty-distribution1",
            "weibull",
            "--frailty-scale1",
            "100.0",
            # --frailty-rho1 omitted
        ]
    )
    with pytest.raises(ValueError, match="--frailty-rho1 is required"):
        parse_hazard_cli(args, trait=1, name="frailty")


def test_kebab_name_maps_to_snake_attr():
    """``name='cure-frailty'`` registers attrs like ``cure_frailty_distribution1``."""
    parser = _parser_for(trait=1, name="cure-frailty")
    args = parser.parse_args(
        [
            "--cure-frailty-distribution1",
            "weibull",
            "--cure-frailty-scale1",
            "50.0",
            "--cure-frailty-rho1",
            "1.5",
        ]
    )
    dist, params = parse_hazard_cli(args, trait=1, name="cure-frailty")
    assert dist == "weibull"
    assert params == {"scale": 50.0, "rho": 1.5}
