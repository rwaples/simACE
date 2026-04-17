"""Tests for simace.analysis.validate — pedigree validation functions."""

from __future__ import annotations

import pytest

from simace.analysis.validate import (
    compute_family_size_distribution,
    compute_per_generation_stats,
    validate_assortative_mating,
    validate_consanguineous_matings,
    validate_half_sibs,
    validate_heritability,
    validate_population,
    validate_statistical,
    validate_structural,
    validate_twins,
)
from simace.core.pedigree_graph import extract_sibling_pairs
from simace.simulation.simulate import run_simulation

# ---------------------------------------------------------------------------
# Module-scoped fixtures (simulation runs once per file)
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = dict(
    seed=42,
    N=1000,
    G_ped=3,
    G_sim=3,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.2,
    A2=0.5,
    C2=0.2,
    rA=0.3,
    rC=0.5,
    assort1=0.0,
    assort2=0.0,
)


@pytest.fixture(scope="module")
def val_pedigree():
    return run_simulation(**_DEFAULT_PARAMS)


@pytest.fixture(scope="module")
def val_params():
    return {**_DEFAULT_PARAMS, "E1": 0.3, "E2": 0.3, "rE": 0.0}


@pytest.fixture(scope="module")
def val_indexed(val_pedigree):
    return val_pedigree.set_index("id")


@pytest.fixture(scope="module")
def val_sibling_pairs(val_pedigree):
    return extract_sibling_pairs(val_pedigree)


@pytest.fixture(scope="module")
def heritability_result(val_pedigree, val_params, val_indexed, val_sibling_pairs):
    return validate_heritability(val_pedigree, val_params, val_indexed, val_sibling_pairs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_passed(result: dict) -> None:
    """Assert every check in a validation result dict has passed=True."""
    for key, value in result.items():
        if isinstance(value, dict) and "passed" in value:
            assert value["passed"], f"Check '{key}' failed: {value.get('details', '')}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateStructural:
    def test_all_checks_pass(self, val_pedigree, val_params):
        result = validate_structural(val_pedigree, val_params)
        _all_passed(result)

    def test_expected_keys(self, val_pedigree, val_params):
        result = validate_structural(val_pedigree, val_params)
        assert "id_integrity" in result
        assert "parent_references" in result
        assert "sex_parent_consistency" in result
        assert "sex_distribution" in result


class TestValidateTwins:
    def test_all_checks_pass(self, val_pedigree, val_params, val_indexed):
        result = validate_twins(val_pedigree, val_params, val_indexed)
        _all_passed(result)

    def test_twin_rate_present(self, val_pedigree, val_params, val_indexed):
        result = validate_twins(val_pedigree, val_params, val_indexed)
        assert "twin_rate" in result
        assert "observed_rate" in result["twin_rate"]


class TestValidateHalfSibs:
    def test_passes(self, val_pedigree, val_params, val_sibling_pairs):
        result = validate_half_sibs(val_pedigree, val_params, val_sibling_pairs)
        _all_passed(result)

    def test_numeric_fields(self, val_pedigree, val_params, val_sibling_pairs):
        result = validate_half_sibs(val_pedigree, val_params, val_sibling_pairs)
        for value in result.values():
            if isinstance(value, dict) and "observed" in value:
                assert isinstance(value["observed"], (int, float))


class TestValidateConsanguineous:
    def test_passes(self, val_pedigree, val_params):
        result = validate_consanguineous_matings(val_pedigree, val_params)
        _all_passed(result)

    def test_non_negative_counts(self, val_pedigree, val_params):
        result = validate_consanguineous_matings(val_pedigree, val_params)
        for key, value in result.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if k.startswith("n_"):
                        assert v >= 0, f"{key}.{k} = {v}"


class TestValidateStatistical:
    def test_all_checks_pass(self, val_pedigree, val_params, val_indexed):
        result = validate_statistical(val_pedigree, val_params, val_indexed)
        _all_passed(result)

    def test_variance_keys(self, val_pedigree, val_params, val_indexed):
        result = validate_statistical(val_pedigree, val_params, val_indexed)
        for comp in ["A1", "C1", "E1", "A2", "C2", "E2"]:
            assert f"variance_{comp}" in result

    def test_total_variance_keys(self, val_pedigree, val_params, val_indexed):
        result = validate_statistical(val_pedigree, val_params, val_indexed)
        assert "total_variance_trait1" in result
        assert "total_variance_trait2" in result


class TestValidateHeritability:
    def test_result_present(self, heritability_result):
        assert isinstance(heritability_result, dict)
        assert len(heritability_result) > 0

    def test_mz_correlations_present(self, heritability_result):
        mz_keys = [k for k in heritability_result if "mz" in k.lower()]
        assert len(mz_keys) > 0

    def test_falconer_present(self, heritability_result):
        falc_keys = [k for k in heritability_result if "falconer" in k.lower()]
        assert len(falc_keys) > 0


class TestComputePerGenerationStats:
    def test_three_generations(self, val_pedigree, val_params):
        result = compute_per_generation_stats(val_pedigree, val_params)
        assert "generation_1" in result
        assert "generation_2" in result
        assert "generation_3" in result

    def test_gen_size(self, val_pedigree, val_params):
        result = compute_per_generation_stats(val_pedigree, val_params)
        for g in range(1, 4):
            assert result[f"generation_{g}"]["n"] == 1000

    def test_liability_stats_present(self, val_pedigree, val_params):
        result = compute_per_generation_stats(val_pedigree, val_params)
        gen1 = result["generation_1"]
        assert "liability1_mean" in gen1
        assert "liability1_variance" in gen1
        assert "A1_var" in gen1


class TestValidatePopulation:
    def test_all_checks_pass(self, val_pedigree, val_params):
        result = validate_population(val_pedigree, val_params)
        _all_passed(result)

    def test_expected_keys(self, val_pedigree, val_params):
        result = validate_population(val_pedigree, val_params)
        assert "generation_sizes" in result
        assert "generation_count" in result


class TestComputeFamilySizeDistribution:
    def test_structure(self, val_pedigree, val_params):
        result = compute_family_size_distribution(val_pedigree, val_params)
        assert "mother" in result
        assert "father" in result
        for parent_type in ["mother", "father"]:
            entry = result[parent_type]
            assert "mean" in entry
            assert "median" in entry
            assert "n_parents" in entry

    def test_mean_around_two(self, val_pedigree, val_params):
        result = compute_family_size_distribution(val_pedigree, val_params)
        assert result["mother"]["mean"] == pytest.approx(2.0, abs=0.5)


class TestValidateAssortativeMating:
    def test_zero_assort_near_zero_corr(self, val_pedigree, val_params, val_indexed):
        result = validate_assortative_mating(val_pedigree, val_params, val_indexed)
        _all_passed(result)

    def test_result_has_mate_correlation(self, val_pedigree, val_params, val_indexed):
        result = validate_assortative_mating(val_pedigree, val_params, val_indexed)
        corr_keys = [k for k in result if "mate" in k.lower() or "corr" in k.lower()]
        assert len(corr_keys) > 0
