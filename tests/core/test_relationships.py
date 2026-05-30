"""Tests for simace.core.relationships canonical relationship semantics (ADR 0009)."""

import pytest
from pedigree_graph import PAIR_KINSHIP

from simace.core.relationships import (
    RELATIONSHIP_TYPES,
    expected_liability_corr,
    shared_environment_coefficient,
)


def test_maternal_vs_paternal_half_sib_share_c_differently():
    """The load-bearing distinction: households are mother-defined, so maternal
    half-sibs share C but paternal half-sibs do not."""
    assert shared_environment_coefficient("MHS") == 1.0
    assert shared_environment_coefficient("PHS") == 0.0


@pytest.mark.parametrize(
    ("rt", "expected"),
    [
        ("MZ", 1.0),
        ("FS", 1.0),
        ("MHS", 1.0),
        ("PHS", 0.0),
        ("MO", 0.0),
        ("FO", 0.0),
        ("1C", 0.0),
    ],
)
def test_shared_environment_coefficient(rt, expected):
    assert shared_environment_coefficient(rt) == expected


@pytest.mark.parametrize(
    ("rt", "want"),
    [
        ("MZ", 2.0 + 3.0),  # A + C
        ("FS", 0.5 * 2.0 + 3.0),  # 0.5A + C
        ("MHS", 0.25 * 2.0 + 3.0),  # 0.25A + C (maternal: shares C)
        ("PHS", 0.25 * 2.0),  # 0.25A     (paternal: no C)
        ("MO", 0.5 * 2.0),  # 0.5A
        ("FO", 0.5 * 2.0),  # 0.5A
        ("1C", 0.125 * 2.0),  # 0.125A
    ],
)
def test_expected_liability_corr(rt, want):
    assert expected_liability_corr(rt, A=2.0, C=3.0) == pytest.approx(want)


def test_kinship_sourced_from_registry():
    """expected_liability_corr(rt, A=1, C=0) must equal 2*PAIR_KINSHIP[rt] — i.e.
    the coefficient is derived from the registry, never a re-declared literal."""
    for rt in RELATIONSHIP_TYPES:
        assert expected_liability_corr(rt, A=1.0, C=0.0) == pytest.approx(2.0 * PAIR_KINSHIP[rt])


def test_every_relationship_type_is_covered():
    """No type in the canonical subset raises — both helpers cover all 7."""
    for rt in RELATIONSHIP_TYPES:
        assert shared_environment_coefficient(rt) in (0.0, 1.0)
        expected_liability_corr(rt, A=1.0, C=1.0)


@pytest.mark.parametrize("fn", [shared_environment_coefficient, lambda rt: expected_liability_corr(rt, 1.0, 1.0)])
def test_unknown_relationship_type_raises(fn):
    with pytest.raises(ValueError, match="unknown relationship type"):
        fn("GP")  # in PAIR_KINSHIP but outside the canonical 7-type subset
    with pytest.raises(ValueError, match="unknown relationship type"):
        fn("NONSENSE")
