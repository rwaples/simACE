"""Tests for simace.core.relationships canonical relationship semantics (ADR 0009)."""

import pytest
from pedigree_graph import RELATIONSHIPS

from simace.core.relationships import (
    RELATIONSHIP_TYPES,
    analytic_applies,
    expected_liability_corr,
    reference_corr,
    reference_midparent_slope,
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
    """expected_liability_corr(rt, A=1, C=0) must equal 2*RELATIONSHIPS[rt].nominal_kinship
    — i.e. the coefficient is derived from the registry, never a re-declared literal."""
    for rt in RELATIONSHIP_TYPES:
        assert expected_liability_corr(rt, A=1.0, C=0.0) == pytest.approx(2.0 * RELATIONSHIPS[rt].nominal_kinship)


def test_canonical_subset_is_a_subset_of_the_registry():
    """Every canonical type is a registry code, so the registry can be indexed by it."""
    assert set(RELATIONSHIP_TYPES) <= set(RELATIONSHIPS)


def test_every_relationship_type_is_covered():
    """No type in the canonical subset raises — both helpers cover all 7."""
    for rt in RELATIONSHIP_TYPES:
        assert shared_environment_coefficient(rt) in (0.0, 1.0)
        expected_liability_corr(rt, A=1.0, C=1.0)


@pytest.mark.parametrize("fn", [shared_environment_coefficient, lambda rt: expected_liability_corr(rt, 1.0, 1.0)])
def test_unknown_relationship_type_raises(fn):
    with pytest.raises(ValueError, match="unknown relationship type"):
        fn("GP")  # a registry code, but outside the canonical 7-type subset
    with pytest.raises(ValueError, match="unknown relationship type"):
        fn("NONSENSE")


@pytest.mark.parametrize(
    ("rt", "want"),
    [("MZ", 1.0), ("FS", 1.0), ("MO", 0.5), ("FO", 0.3), ("MHS", 1.0 - 0.09), ("PHS", 0.09), ("1C", None)],
)
def test_shared_environment_coefficient_under_transmission(rt, want):
    got = shared_environment_coefficient(rt, b_m=0.5, b_f=0.3)
    assert got == (None if want is None else pytest.approx(want))


def test_expected_liability_corr_divides_by_liability_variance():
    # MO: (0.5 * A + b_m * C) / V_P
    assert expected_liability_corr("MO", A=0.4, C=0.2, b_m=0.5, V_P=2.0) == pytest.approx((0.2 + 0.1) / 2.0)
    assert expected_liability_corr("1C", A=0.4, C=0.2, b_m=0.5) is None


_RM = {"A1": 0.4, "C1": 0.3, "E1": 0.3, "assort1": 0.0, "assort2": 0.0, "assort_matrix": None}


@pytest.mark.parametrize(
    ("overrides", "rt"),
    [
        ({"assort1": 0.3}, "MO"),
        ({"assort2": {0: 0.0, 2: 0.2}}, "MO"),
        ({"assort_matrix": [[0.0, 0.1], [0.1, 0.0]]}, "MO"),
        ({"C1": {0: 0.3, 2: 0.2}}, "MO"),
        ({"E1": {0: 0.3, 2: 0.5}}, "MO"),
        ({}, "1C"),
    ],
    ids=["am", "per_gen_am", "cross_am", "dict_c", "dict_e", "lineage_mixed"],
)
def test_analytic_gate_rejects_what_the_formula_cannot_describe(overrides, rt):
    params = {**_RM, **overrides, "Ctm_maternal1": 0.5}
    assert not analytic_applies(rt, params, 1)
    assert reference_corr(rt, params, 1) is None
    assert reference_corr(rt, params, 1, truth={rt: {"r": 0.42}}) == 0.42


def test_analytic_gate_passes_without_transmission_whatever_the_mating():
    params = {**_RM, "assort1": 0.3, "E1": {0: 0.3, 2: 0.5}}
    assert analytic_applies("1C", params, 1)
    assert reference_corr("FS", params, 1, truth={"FS": {"r": 0.9}}) == pytest.approx(0.5 * 0.4 + 0.3)
    assert reference_midparent_slope(params, 1) == 0.4


def test_reference_lines_under_the_gate():
    params = {**_RM, "Ctm_maternal1": 0.5, "Ctm_paternal1": 0.3}
    assert reference_corr("PHS", params, 1) == pytest.approx(0.25 * 0.4 + 0.09 * 0.3)
    assert reference_midparent_slope(params, 1) == pytest.approx(0.4 + 0.8 * 0.3)
    assert reference_midparent_slope({**params, "assort1": 0.2}, 1, truth=0.61) == 0.61


def test_reference_lines_divide_by_liability_variance_without_transmission_too():
    params = {"A1": 0.8, "C1": 0.4, "E1": 0.8, "assort1": 0.0, "assort2": 0.0}  # V_P = 2
    assert reference_corr("FS", params, 1) == pytest.approx((0.4 + 0.4) / 2.0)
    assert reference_midparent_slope(params, 1) == pytest.approx(0.4)
    nearby = reference_corr("FS", {**params, "Ctm_maternal1": 1e-9}, 1)
    assert nearby == pytest.approx(reference_corr("FS", params, 1))  # continuous in b


def test_wright_fisher_is_random_mating_whatever_it_inherits():
    params = {**_RM, "assort1": 0.3, "mating_model": "wright_fisher", "Ctm_maternal1": 0.5}
    assert analytic_applies("MO", params, 1)


def test_transmission_is_per_trait():
    params = {**_RM, "A2": 0.5, "C2": 0.1, "E2": 0.4, "Ctm_maternal2": 0.5, "assort1": 0.3}
    assert analytic_applies("MO", params, 1)
    assert not analytic_applies("MO", params, 2)
