"""Tests for simace.analysis.validate — pedigree validation functions."""

import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from pedigree_graph import PedigreeGraph

from simace.analysis.validate import (
    cli as validate_cli,
)
from simace.analysis.validate import (
    compute_family_size_distribution,
    compute_per_generation_stats,
    run_validation,
    validate_assortative_mating,
    validate_consanguineous_matings,
    validate_half_sibs,
    validate_heritability,
    validate_population,
    validate_statistical,
    validate_structural,
    validate_twins,
)
from simace.core.pedigree_arrays import PedigreeArrays
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
    E1=0.3,
    A2=0.5,
    C2=0.2,
    E2=0.3,
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
    return {**_DEFAULT_PARAMS, "rE": 0.0}


@pytest.fixture(scope="module")
def val_ped(val_pedigree):
    return PedigreeArrays.from_frame(val_pedigree)


@pytest.fixture(scope="module")
def val_sibling_pairs(val_pedigree):
    all_pairs = PedigreeGraph(val_pedigree).extract_pairs(max_degree=2)
    return {k: all_pairs[k] for k in ("FS", "MHS", "PHS")}


@pytest.fixture(scope="module")
def heritability_result(val_pedigree, val_params, val_ped, val_sibling_pairs):
    return validate_heritability(val_pedigree, val_params, val_ped, val_sibling_pairs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_passed(result: dict) -> None:
    """Assert every check in a validation result dict has passed=True."""
    for key, value in result.items():
        if isinstance(value, dict) and "passed" in value:
            assert value["passed"], f"Check '{key}' failed: {value.get('details', '')}"


def _component_df(a: np.ndarray, c: np.ndarray, e: np.ndarray) -> pd.DataFrame:
    """Build a one-generation two-trait DataFrame with matching components."""
    return pd.DataFrame(
        {
            "id": np.arange(len(a)),
            "A1": a,
            "C1": c,
            "E1": e,
            "A2": a,
            "C2": c,
            "E2": e,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateStructural:
    def test_all_checks_pass(self, val_pedigree, val_params, val_ped):
        result = validate_structural(val_pedigree, val_params, val_ped)
        _all_passed(result)

    def test_expected_keys(self, val_pedigree, val_params, val_ped):
        result = validate_structural(val_pedigree, val_params, val_ped)
        assert "id_integrity" in result
        assert "parent_references" in result
        assert "sex_parent_consistency" in result
        assert "sex_distribution" in result


class TestValidateTwins:
    def test_all_checks_pass(self, val_pedigree, val_params, val_ped):
        result = validate_twins(val_pedigree, val_params, val_ped)
        _all_passed(result)

    def test_twin_rate_present(self, val_pedigree, val_params, val_ped):
        result = validate_twins(val_pedigree, val_params, val_ped)
        assert "twin_rate" in result
        assert "observed_rate" in result["twin_rate"]

    def test_wf_with_inherited_default_p_mztwin_passes_vacuously(self, val_pedigree, val_ped):
        # Under WF, inherited p_mztwin=0.02 must NOT trigger a failed twin_rate
        # check (the standard branch fails because 0.02 > 0.01).  Branch on
        # mating_model and report observed_rate=0.0, expected_rate=0.0.
        wf_params = {"mating_model": "wright_fisher", "p_mztwin": 0.02}
        # Use a freshly-simulated WF pedigree so n_twins is actually 0.
        wf_ped = run_simulation(
            seed=11,
            N=500,
            G_ped=2,
            G_sim=2,
            mating_lambda=0.5,
            p_mztwin=0.02,
            A1=0.5,
            C1=0.0,
            E1=0.5,
            A2=0.4,
            C2=0.0,
            E2=0.6,
            rA=0.0,
            rC=0.0,
            rE=0.0,
            mating_model="wright_fisher",
        )
        result = validate_twins(wf_ped, wf_params, PedigreeArrays.from_frame(wf_ped))
        _all_passed(result)
        assert result["twin_rate"]["expected_rate"] == 0.0
        assert result["twin_rate"]["observed_rate"] == 0.0

    def test_wf_fails_when_twins_present(self, val_pedigree, val_ped):
        """If a pedigree labelled mating_model=wright_fisher contains any twins
        (regression / corruption), validate_twins must fail the rate check
        rather than silently pass.  The val_pedigree fixture is a standard-model
        pedigree with twins; relabelling its params as WF must surface the
        violation.
        """
        wf_params = {"mating_model": "wright_fisher", "p_mztwin": 0.02}
        result = validate_twins(val_pedigree, wf_params, val_ped)
        assert (val_pedigree["twin"] != -1).any(), "fixture sanity: pedigree must contain twins"
        rate = result["twin_rate"]
        assert rate["passed"] is False
        assert rate["expected_rate"] == 0.0
        assert rate["observed_rate"] > 0.0
        assert rate["twin_pairs"] > 0


class TestValidateHalfSibs:
    def test_passes(self, val_pedigree, val_params, val_ped, val_sibling_pairs):
        result = validate_half_sibs(val_pedigree, val_params, val_ped, val_sibling_pairs)
        _all_passed(result)

    def test_numeric_fields(self, val_pedigree, val_params, val_ped, val_sibling_pairs):
        result = validate_half_sibs(val_pedigree, val_params, val_ped, val_sibling_pairs)
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
    def test_all_checks_pass(self, val_pedigree, val_params):
        result = validate_statistical(val_pedigree, val_params)
        _all_passed(result)

    def test_variance_keys(self, val_pedigree, val_params):
        result = validate_statistical(val_pedigree, val_params)
        for comp in ["A1", "C1", "E1", "A2", "C2", "E2"]:
            assert f"variance_{comp}" in result

    def test_total_variance_keys(self, val_pedigree, val_params):
        result = validate_statistical(val_pedigree, val_params)
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

    def test_covariance_primitives_present(self, val_pedigree, val_params):
        result = compute_per_generation_stats(val_pedigree, val_params)
        gen1 = result["generation_1"]
        assert "A1_cov_non_genetic" in gen1
        assert "A1_cov_C" in gen1
        assert "A1_cov_E" in gen1

    def test_covariance_uses_population_denominator(self):
        a = np.array([-1.0, 0.0, 1.0, 2.0])
        c = a.copy()
        e = np.zeros_like(a)
        df = _component_df(a, c, e)
        result = compute_per_generation_stats(df, {"N": len(df), "G_ped": 1})["generation_1"]
        expected = np.mean((a - a.mean()) * ((c + e) - (c + e).mean()))
        assert result["A1_cov_non_genetic"] == expected
        assert result["A1_cov_non_genetic"] == result["A1_var"]

    def test_independent_constructed_non_genetic_covariance_is_zero(self):
        a = np.array([-1.0, -1.0, 1.0, 1.0])
        c = np.array([-1.0, 1.0, -1.0, 1.0])
        e = np.zeros_like(a)
        df = _component_df(a, c, e)
        result = compute_per_generation_stats(df, {"N": len(df), "G_ped": 1})["generation_1"]
        assert result["A1_cov_non_genetic"] == 0.0

    def test_constructed_non_genetic_equals_a_gives_snp_like_identity(self):
        from simace.plotting.plot_heritability import _derive_ge_h2_metrics

        a = np.array([-1.0, 0.0, 1.0, 2.0])
        c = a.copy()
        e = np.zeros_like(a)
        df = _component_df(a, c, e)
        result = compute_per_generation_stats(df, {"N": len(df), "G_ped": 1})["generation_1"]
        derived = _derive_ge_h2_metrics(result["A1_var"], result["liability1_variance"], result["A1_cov_non_genetic"])
        assert result["A1_cov_non_genetic"] == result["A1_var"]
        assert derived["ge_cov_fraction"] == 2 * result["A1_var"] / result["liability1_variance"]
        assert derived["h2_snp_like"] == 1.0

    def test_zero_additive_variance_snp_like_is_nan(self):
        from simace.plotting.plot_heritability import _derive_ge_h2_metrics

        derived = _derive_ge_h2_metrics(var_a=0.0, var_liability=1.0, cov_a_non_genetic=0.0, n=10)
        assert np.isnan(derived["h2_snp_like"])


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
    def test_zero_assort_near_zero_corr(self, val_pedigree, val_params, val_ped):
        result = validate_assortative_mating(val_pedigree, val_params, val_ped)
        _all_passed(result)

    def test_result_has_mate_correlation(self, val_pedigree, val_params, val_ped):
        result = validate_assortative_mating(val_pedigree, val_params, val_ped)
        corr_keys = [k for k in result if "mate" in k.lower() or "corr" in k.lower()]
        assert len(corr_keys) > 0

    def test_cross_trait_branch_when_both_assort(self, val_pedigree, val_ped, val_params):
        # Force the cross-trait branch by claiming both traits assort.
        params = {**val_params, "assort1": 0.2, "assort2": 0.3}
        result = validate_assortative_mating(val_pedigree, params, val_ped)
        assert "mate_corr_cross_12" in result
        assert "mate_corr_cross_21" in result
        for key in ("mate_corr_cross_12", "mate_corr_cross_21"):
            assert "expected" in result[key]
            assert "observed" in result[key]

    def test_cross_trait_uses_assort_matrix_when_provided(self, val_pedigree, val_ped, val_params):
        params = {
            **val_params,
            "assort1": 0.2,
            "assort2": 0.3,
            "assort_matrix": [[0.2, 0.05], [0.05, 0.3]],
        }
        result = validate_assortative_mating(val_pedigree, params, val_ped)
        assert result["mate_corr_cross_12"]["expected"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# run_validation orchestrator + CLI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def written_scenario(tmp_path_factory, val_pedigree, val_params):
    tmp = tmp_path_factory.mktemp("validate_scenario")
    ped_path = tmp / "pedigree.parquet"
    params_path = tmp / "params.yaml"
    val_pedigree.to_parquet(ped_path)
    with open(params_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(val_params, fh)
    return ped_path, params_path, tmp


class TestRunValidation:
    def test_returns_summary(self, written_scenario):
        ped_path, params_path, _ = written_scenario
        result = run_validation(str(ped_path), str(params_path))
        assert "summary" in result
        s = result["summary"]
        assert s["checks_total"] == s["checks_passed"] + s["checks_failed"]
        assert s["passed"] is (s["checks_failed"] == 0)

    def test_top_level_categories_present(self, written_scenario):
        ped_path, params_path, _ = written_scenario
        result = run_validation(str(ped_path), str(params_path))
        for cat in (
            "structural",
            "twins",
            "half_sibs",
            "statistical",
            "heritability",
            "population",
            "per_generation",
            "assortative_mating",
            "consanguineous_matings",
            "summary",
            "family_size_distribution",
            "parameters",
        ):
            assert cat in result, f"missing category {cat!r}"

    def test_parameters_round_trip(self, written_scenario, val_params):
        ped_path, params_path, _ = written_scenario
        result = run_validation(str(ped_path), str(params_path))
        # Loaded params should match what we wrote
        assert result["parameters"]["A1"] == val_params["A1"]
        assert result["parameters"]["seed"] == val_params["seed"]


class TestValidateNegativePaths:
    """Corrupted pedigrees are caught by the appropriate structural / twin check."""

    @staticmethod
    def _tiny_params():
        return {**_DEFAULT_PARAMS, "rE": 0.0, "N": 100, "G_ped": 2, "G_sim": 2}

    @staticmethod
    def _tiny_pedigree():
        return run_simulation(**TestValidateNegativePaths._tiny_params())

    def test_non_contiguous_ids_fails_id_integrity(self):
        ped = self._tiny_pedigree()
        # Skip an integer in the id column — sort(ids) != arange(N*G_ped).
        ped.loc[ped.index[5], "id"] = ped["id"].max() + 99
        result = validate_structural(ped, self._tiny_params(), PedigreeArrays.from_frame(ped))
        assert result["id_integrity"]["passed"] is False

    def test_dangling_parent_id_fails_parent_references(self):
        ped = self._tiny_pedigree()
        params = self._tiny_params()
        # Force a mother index outside [0, N*G_ped) and not -1.
        non_founder_idx = ped.index[ped["mother"] != -1][0]
        ped.loc[non_founder_idx, "mother"] = params["N"] * params["G_ped"] + 50
        result = validate_structural(ped, params, PedigreeArrays.from_frame(ped))
        assert result["parent_references"]["passed"] is False

    def test_wrong_parent_sex_fails_sex_consistency(self):
        ped = self._tiny_pedigree()
        non_founder = ped[ped["mother"] != -1].iloc[0]
        ped.loc[ped["id"] == non_founder["mother"], "sex"] = 1  # 1 = male
        result = validate_structural(ped, self._tiny_params(), PedigreeArrays.from_frame(ped))
        assert result["sex_parent_consistency"]["passed"] is False

    def test_non_bidirectional_twin_fails_bidirectional_check(self):
        """A twin pointer that doesn't bounce back must fail twin_bidirectional."""
        ped = self._tiny_pedigree()
        # Pick three contiguous IDs to wire as a broken twin chain.
        a, b, c = 0, 1, 2
        ped.loc[ped["id"] == a, "twin"] = b
        ped.loc[ped["id"] == b, "twin"] = c  # broken: should be `a` to be bidirectional
        ped.loc[ped["id"] == c, "twin"] = b
        params = self._tiny_params()
        params["p_mztwin"] = 0.02
        result = validate_twins(ped, params, PedigreeArrays.from_frame(ped))
        assert result["twin_bidirectional"]["passed"] is False


class TestValidateCli:
    def test_writes_output_yaml(self, written_scenario, monkeypatch):
        ped_path, params_path, tmp = written_scenario
        out_path = tmp / "validation.yaml"
        argv = [
            "validate",
            "--pedigree",
            str(ped_path),
            "--params",
            str(params_path),
            "--output",
            str(out_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        validate_cli()
        assert out_path.exists()
        with open(out_path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        assert "summary" in loaded
        assert "structural" in loaded
