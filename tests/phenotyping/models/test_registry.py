"""Sanity checks on the phenotype model registry."""

import pytest

from simace.phenotyping.models import (
    MODELS,
    AdultModel,
    CureFrailtyModel,
    FirstPassageModel,
    FrailtyModel,
    PhenotypeModel,
)

EXPECTED = {
    "frailty": FrailtyModel,
    "cure_frailty": CureFrailtyModel,
    "adult": AdultModel,
    "first_passage": FirstPassageModel,
}


def test_registry_keys_match_expected():
    assert set(MODELS) == set(EXPECTED)


@pytest.mark.parametrize(("name", "cls"), list(EXPECTED.items()))
def test_registry_class_subclasses_phenotype_model(name, cls):
    assert MODELS[name] is cls
    assert issubclass(cls, PhenotypeModel)
    assert cls.name == name


@pytest.mark.parametrize("cls", list(EXPECTED.values()))
def test_cli_flag_attrs_are_disjoint_per_trait(cls):
    """Each model's CLI flags must not collide with itself across traits 1 vs 2."""
    a1 = cls.cli_flag_attrs(1)
    a2 = cls.cli_flag_attrs(2)
    assert a1.isdisjoint(a2)


def test_cli_flag_attrs_are_disjoint_across_models():
    """Different models must not register colliding attribute names."""
    seen: dict[str, str] = {}
    for trait in (1, 2):
        for cls in EXPECTED.values():
            for attr in cls.cli_flag_attrs(trait):
                assert attr not in seen, f"flag attr {attr!r} registered by both {seen.get(attr)} and {cls.name}"
                seen[attr] = cls.name
