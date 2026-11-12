"""Sanity checks on the phenotype model registry."""

import pytest

from simace.phenotype.models import (
    MODELS,
    AdultModel,
    CureFrailtyModel,
    FirstPassageModel,
    FrailtyModel,
    PhenotypeModel,
    SimpleLtmModel,
)

EXPECTED = {
    "frailty": FrailtyModel,
    "cure_frailty": CureFrailtyModel,
    "adult": AdultModel,
    "first_passage": FirstPassageModel,
    "simple_ltm": SimpleLtmModel,
}


def test_registry_keys_match_expected():
    assert set(MODELS) == set(EXPECTED)


# A minimal valid phenotype_params per model, with each from_config's other inputs.
VALID_PARAMS = {
    "frailty": {"distribution": "weibull", "scale": 316.228, "rho": 2.0},
    "cure_frailty": {"distribution": "exponential", "scale": 50.0, "prevalence": 0.1},
    "adult": {"method": "ltm", "prevalence": 0.1},
    "first_passage": {"drift": -0.5, "shape": 1.0},
    "simple_ltm": {"prevalence": 0.1, "onset": {"kind": "fixed", "age": 30}},
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_from_config_accepts_valid_params(name):
    EXPECTED[name].from_config({"phenotype_params1": VALID_PARAMS[name], "beta1": 1.0}, trait_num=1)


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("stray", ["standardise_hazard", "cip_k_typo"])
def test_from_config_rejects_unknown_keys(name, stray):
    """A misspelled or foreign key must fail, not leave its setting at the default (#37)."""
    params = {"phenotype_params1": {**VALID_PARAMS[name], stray: 1}, "beta1": 1.0}
    with pytest.raises(ValueError, match=rf"phenotype\.trait1.*unknown key\(s\) \['{stray}'\]"):
        EXPECTED[name].from_config(params, trait_num=1)


@pytest.mark.parametrize("name", ["frailty", "cure_frailty", "first_passage", "simple_ltm"])
def test_from_config_rejects_another_models_key(name):
    params = {"phenotype_params1": {**VALID_PARAMS[name], "cip_k": 0.2}, "beta1": 1.0}
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['cip_k'\]"):
        EXPECTED[name].from_config(params, trait_num=1)


def test_from_config_rejects_unknown_keys_before_subclass_hook():
    """A subclass with no key checks of its own still gets them from the inherited ``from_config``."""
    hook_calls = []

    class _Probe(PhenotypeModel):
        name = "simple_ltm"
        config_keys = SimpleLtmModel.config_keys

        @classmethod
        def _from_config(cls, phenotype_params, trait_num, *, beta, beta_sex):
            hook_calls.append(phenotype_params)
            return cls()

        simulate = None

    params = {"phenotype_params2": {**VALID_PARAMS["simple_ltm"], "onset_typo": 1}, "beta2": 1.0}
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['onset_typo'\]") as excinfo:
        _Probe.from_config(params, trait_num=2)
    assert str(excinfo.value).count("phenotype.trait") == 1
    assert str(excinfo.value).startswith("phenotype.trait2: ")
    assert hook_calls == []

    _Probe.from_config({"phenotype_params2": VALID_PARAMS["simple_ltm"], "beta2": 1.0}, trait_num=2)
    assert hook_calls == [VALID_PARAMS["simple_ltm"]]


@pytest.mark.parametrize(("name", "cls"), list(EXPECTED.items()))
def test_registry_class_subclasses_phenotype_model(name, cls):
    assert MODELS[name] is cls
    assert issubclass(cls, PhenotypeModel)
    assert cls.name == name
