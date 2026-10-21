"""Tests for simace.analysis.quality — Analyze's quality-check families."""

import numpy as np
import polars as pl
import pytest

from simace.analysis.mate_correlation import mate_pedigree
from simace.analysis.quality import (
    build_quality_report,
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
from simace.analysis.quality._common import mate_correlations, sibling_moments
from simace.analysis.quality.am_equilibrium import validate_am_equilibrium
from simace.analysis.quality.assortative_mating import recorded_matings
from simace.ascertainment import run_ascertainment
from simace.core.pedigree_arrays import PedigreeArrays
from simace.simulation.assortment import AssortmentPlan
from simace.simulation.simulate import resolve_per_gen_param, run_simulation

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
def val_sibling_moments(val_pedigree, val_ped):
    return sibling_moments(val_pedigree, val_ped)


@pytest.fixture(scope="module")
def val_mate_pedigree(val_pedigree):
    return mate_pedigree(val_pedigree)


@pytest.fixture(scope="module")
def val_mates(val_pedigree, val_mate_pedigree):
    return mate_correlations(val_pedigree, val_mate_pedigree)


@pytest.fixture(scope="module")
def heritability_result(val_pedigree, val_params, val_ped, val_mate_pedigree, val_sibling_moments, val_mates):
    return validate_heritability(val_pedigree, val_params, val_ped, val_mate_pedigree, val_sibling_moments, val_mates)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_passed(result: dict) -> None:
    """Assert every check in a validation result dict has passed=True."""
    for key, value in result.items():
        if isinstance(value, dict) and "passed" in value:
            assert value["passed"], f"Check '{key}' failed: {value.get('details', '')}"


def _component_df(a: np.ndarray, c: np.ndarray, e: np.ndarray) -> pl.DataFrame:
    """Build a one-generation two-trait DataFrame with matching components."""
    return pl.DataFrame(
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


def _with_column_values(df: pl.DataFrame, col: str, values: np.ndarray) -> pl.DataFrame:
    """Return ``df`` with column ``col`` replaced, preserving its dtype."""
    return df.with_columns(pl.Series(col, values).cast(df.schema[col]))


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

    def test_wf_with_inherited_default_p_mztwin_scores_only_the_rate(self, val_pedigree, val_ped):
        # Under WF, inherited p_mztwin=0.02 must NOT trigger a failed twin_rate
        # check (the standard branch fails because 0.02 > 0.01).  Branch on
        # mating_model and report observed_rate=0.0, expected_rate=0.0. With
        # no twin records the structural twin checks are informational.
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
        assert result["twin_rate"]["passed"] is True
        assert result["twin_rate"]["expected_rate"] == 0.0
        assert result["twin_rate"]["observed_rate"] == 0.0
        for name in ("twin_bidirectional", "twin_same_parents", "twin_same_A1", "twin_same_A2", "twin_same_sex"):
            assert result[name].get("informational") is True
            assert "passed" not in result[name]

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
    def test_passes(self, val_pedigree, val_params, val_sibling_moments, val_mates):
        result = validate_half_sibs(val_pedigree, val_params, val_sibling_moments, val_mates)
        _all_passed(result)

    def test_numeric_fields(self, val_pedigree, val_params, val_sibling_moments, val_mates):
        result = validate_half_sibs(val_pedigree, val_params, val_sibling_moments, val_mates)
        for value in result.values():
            if isinstance(value, dict) and "observed" in value:
                assert isinstance(value["observed"], (int, float))


class TestValidateConsanguineous:
    def test_passes(self, val_pedigree, val_params, val_ped):
        result = validate_consanguineous_matings(val_pedigree, val_params, val_ped)
        _all_passed(result)

    def test_non_negative_counts(self, val_pedigree, val_params, val_ped):
        result = validate_consanguineous_matings(val_pedigree, val_params, val_ped)
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

    def test_total_variance_and_rc_are_reported_only(self, val_pedigree, val_params):
        """`simace validate` tests them (ADR 0030); Analyze keeps their values for `truth`."""
        result = validate_statistical(val_pedigree, val_params)
        for key in ("total_variance_trait1", "total_variance_trait2", "cross_trait_rC"):
            assert result[key]["informational"] is True
            assert "passed" not in result[key]
            assert np.isfinite(result[key]["observed"])

    def test_cross_trait_rA_is_scored_under_random_mating(self, val_pedigree, val_params):
        assert validate_statistical(val_pedigree, val_params)["cross_trait_rA"]["passed"] is True

    @pytest.mark.parametrize(
        "assort",
        [
            pytest.param({"assort1": 0.3}, id="trait1"),
            pytest.param({"assort2": 0.3}, id="trait2"),
            pytest.param({"assort1": 0.3, "assort2": 0.3}, id="both-traits"),
            pytest.param({"assort1": {0: 0.0, 2: 0.3}}, id="per-generation"),
        ],
    )
    def test_cross_trait_rA_is_informational_under_assortative_mating(self, val_pedigree, val_params, assort):
        # AM shifts the founders' rA away from the configured value, as it
        # inflates Var(A), so it is reported with its observed value only.
        result = validate_statistical(val_pedigree, {**val_params, **assort})
        random_mating = validate_statistical(val_pedigree, val_params)
        cross = result["cross_trait_rA"]
        assert cross.get("informational") is True
        assert "passed" not in cross
        assert cross["observed"] == random_mating["cross_trait_rA"]["observed"]
        # rE does not depend on assortment and stays scored.
        assert "passed" in result["cross_trait_rE"]

    def test_cross_trait_rA_stays_scored_under_wright_fisher(self, val_pedigree, val_params):
        # Wright-Fisher pairs at random whatever assortment it inherits.
        params = {**val_params, "assort1": 0.3, "assort2": 0.3, "mating_model": "wright_fisher"}
        assert "passed" in validate_statistical(val_pedigree, params)["cross_trait_rA"]


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

    def test_generation_count_reports_the_number_of_labels(self, val_pedigree, val_params):
        count = validate_population(val_pedigree, val_params)["generation_count"]
        assert count["passed"] is True
        assert count["observed"] == count["expected"] == val_params["G_ped"]

    @pytest.mark.parametrize(
        "relabel",
        [
            pytest.param(lambda g: np.where(g == 2, 1, g), id="missing-last"),
            pytest.param(lambda g: np.where(g == 1, 2, g), id="gap"),
            pytest.param(lambda g: g + 1, id="shifted"),
            pytest.param(lambda g: np.where(g == 2, 3, g), id="beyond-G_ped"),
            pytest.param(lambda g: np.where(np.arange(len(g)) == 0, 7, g), id="one-stray-row"),
        ],
    )
    def test_generation_count_fails_on_wrong_generation_labels(self, val_pedigree, val_params, relabel):
        generation = relabel(val_pedigree["generation"].to_numpy())
        ped = _with_column_values(val_pedigree, "generation", generation)
        result = validate_population(ped, val_params)
        assert result["generation_count"]["passed"] is False
        # The id ranges are untouched, so only the label check sees the damage.
        assert result["generation_sizes"]["passed"] is True


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


class TestRecordedMatings:
    """The validator's targets follow the simulator's mating schedule (#64)."""

    def test_one_entry_per_recorded_parents_generation_after_burn_in(self, val_params):
        params = {**val_params, "G_sim": 6, "G_ped": 4, "assort1": {0: 0.1, 4: 0.4}, "assort2": 0.3}
        matings = recorded_matings(params)
        assert [m.parents_generation for m in matings] == [0, 1, 2]
        assert [m.expected[0, 0] for m in matings] == pytest.approx([0.1, 0.4, 0.4])

    def test_cross_target_matches_the_assortment_plan(self, val_params):
        params = {
            **val_params,
            "G_sim": 6,
            "G_ped": 4,
            "assort1": {0: 0.1, 4: 0.4},
            "assort2": 0.3,
            "rE": 0.5,
            "E1": {0: 0.3, 4: 0.2},
        }
        e1 = resolve_per_gen_param(params["E1"], 6)
        plan = AssortmentPlan.build(
            assort1=params["assort1"],
            assort2=params["assort2"],
            R_mf_user=None,
            rA=params["rA"],
            rC=params["rC"],
            rE=params["rE"],
            A1=params["A1"],
            A2=params["A2"],
            C1_per_gen=[params["C1"]] * 6,
            C2_per_gen=[params["C2"]] * 6,
            E1_per_gen=e1,
            E2_per_gen=[params["E2"]] * 6,
            G_sim=6,
        )
        for m in recorded_matings(params):
            R_mf = plan.for_generation(m.parents_generation + 3)[3]
            np.testing.assert_allclose(m.expected, R_mf)
            assert m.cross_scored

    def test_a_single_recorded_generation_has_no_matings(self, val_params):
        assert recorded_matings({**val_params, "G_sim": 1, "G_ped": 1, "assort1": 0.3}) == []

    def test_assort_matrix_diagonal_overrides_the_scalar_assortment(self, val_params):
        """The simulator mates on the matrix diagonal; params.yaml echoes the raw scalars."""
        matrix = [[0.3, 0.05], [0.05, 0.4]]
        params = {**val_params, "assort1": 0.0, "assort2": 0.0, "assort_matrix": matrix}
        for m in recorded_matings(params):
            np.testing.assert_allclose(m.expected, matrix)
            assert m.cross_scored

    def test_single_trait_assortment_induces_the_other_cells(self, val_params):
        params = {**val_params, "rA": 1.0, "rC": 0.0, "A1": 0.5, "C1": 0.0, "E1": 0.5}
        params |= {"A2": 0.5, "C2": 0.0, "E2": 0.5, "assort1": 0.0, "assort2": 0.8}
        for m in recorded_matings(params):
            np.testing.assert_allclose(m.expected, [[0.2, 0.4], [0.4, 0.8]])
            assert not m.cross_scored

    def test_wright_fisher_targets_are_zero_whatever_assortment_it_inherits(self, val_params):
        params = {**val_params, "mating_model": "wright_fisher", "assort1": 0.3, "assort2": 0.3}
        for m in recorded_matings(params):
            np.testing.assert_array_equal(m.expected, np.zeros((2, 2)))
            assert not m.cross_scored

    def test_cross_is_scored_only_where_both_traits_assort(self, val_params):
        params = {**val_params, "G_sim": 4, "G_ped": 4, "assort1": {0: 0.0, 2: 0.3}, "assort2": 0.3}
        assert [m.cross_scored for m in recorded_matings(params)] == [False, True, True]


def _assert_information(result: dict) -> None:
    """Every mate correlation is information: no pass/fail and no per-generation entry (ADR 0031)."""
    for key, value in result.items():
        assert value["informational"], key
        assert "passed" not in value, key
        assert "_gen" not in key


class TestValidateAssortativeMating:
    def test_zero_assort_reports_information_against_zero(self, val_params, val_mates):
        result = validate_assortative_mating(val_params, val_mates)
        _assert_information(result)
        assert set(result) == {"mate_corr_liability1", "mate_corr_liability2", "mate_corr_A1", "mate_corr_A2"}
        for t in (1, 2):
            entry = result[f"mate_corr_liability{t}"]
            assert entry["expected"] == 0
            assert entry["observed"] == pytest.approx(val_mates.liability[t - 1, t - 1])
            assert entry["n_pairs"] == val_mates.n_pairs

    def test_cross_trait_when_both_assort(self, val_mates, val_params):
        params = {**val_params, "assort1": 0.2, "assort2": 0.3}
        result = validate_assortative_mating(params, val_mates)
        _assert_information(result)
        for key, cell in (("mate_corr_cross_12", (0, 1)), ("mate_corr_cross_21", (1, 0))):
            assert "expected" in result[key]
            assert result[key]["observed"] == pytest.approx(val_mates.liability[cell])

    def test_cross_trait_uses_assort_matrix_when_provided(self, val_mates, val_params):
        params = {
            **val_params,
            "assort1": 0.2,
            "assort2": 0.3,
            "assort_matrix": [[0.2, 0.05], [0.05, 0.3]],
        }
        result = validate_assortative_mating(params, val_mates)
        assert result["mate_corr_cross_12"]["expected"] == pytest.approx(0.05)

    def test_liability_target_reads_the_assort_matrix_diagonal(self, val_mates, val_params):
        params = {**val_params, "assort_matrix": [[0.2, 0.05], [0.05, 0.3]]}
        result = validate_assortative_mating(params, val_mates)
        assert result["mate_corr_liability1"]["expected"] == pytest.approx(0.2)
        assert result["mate_corr_liability2"]["expected"] == pytest.approx(0.3)

    def test_cross_trait_expected_includes_unique_environment(self, val_mates, val_params):
        """rho_w carries rE * sqrt(E1 * E2), as the simulator's does (#61)."""
        params = {**val_params, "assort1": 0.2, "assort2": 0.3, "rA": 0.0, "rC": 0.0, "rE": 0.8}
        result = validate_assortative_mating(params, val_mates)
        rho_w = 0.8 * np.sqrt(params["E1"] * params["E2"])
        for key in ("mate_corr_cross_12", "mate_corr_cross_21"):
            assert result[key]["expected"] == pytest.approx(rho_w * np.sqrt(0.2 * 0.3))

    @pytest.mark.parametrize(
        "over",
        [
            pytest.param({"rE": 0.5, "E1": {0: 0.3, 1: 0.2}}, id="target-varies"),
            pytest.param({"rE": 0.0, "E1": {0: 0.3, 1: 0.2}}, id="scales-change"),
            pytest.param({"assort1": {0: 0.0, 2: 0.3}}, id="one-generation-random"),
        ],
    )
    def test_cross_trait_has_no_pooled_target_across_generations(self, val_mates, val_params, over):
        """A pooled cross-trait Pearson mixes targets or trait scales (#61, #64)."""
        params = {**val_params, "assort1": 0.2, "assort2": 0.3, **over}
        result = validate_assortative_mating(params, val_mates)
        _assert_information(result)
        for key in ("mate_corr_cross_12", "mate_corr_cross_21"):
            assert "expected" not in result[key]
            assert result[key]["observed"] is not None

    @pytest.mark.parametrize(
        "over",
        [
            # G_sim=3 records iterations 1 and 2, which mate parental C/E generations 0 and 1.
            pytest.param({"rE": 0.5, "E1": {0: 0.3, 2: 0.2}}, id="unmated-generation"),
            # G_sim=6, G_ped=3 records iterations 4 and 5 (generations 3, 4); the change at 2 is burn-in.
            pytest.param({"rE": 0.5, "G_sim": 6, "E1": {0: 0.2, 2: 0.3}}, id="burn-in-only"),
        ],
    )
    def test_cross_trait_pools_per_generation_params_that_leave_recorded_matings_constant(
        self, val_mates, val_params, over
    ):
        params = {**val_params, "assort1": 0.2, "assort2": 0.3, **over}
        result = validate_assortative_mating(params, val_mates)
        rho_w = (
            params["rA"] * np.sqrt(params["A1"] * params["A2"])
            + params["rC"] * np.sqrt(params["C1"] * params["C2"])
            + params["rE"] * np.sqrt(0.3 * params["E2"])
        )
        for key in ("mate_corr_cross_12", "mate_corr_cross_21"):
            assert result[key]["expected"] == pytest.approx(rho_w * np.sqrt(0.2 * 0.3))

    def test_per_generation_assort1_drops_only_its_own_target(self, val_mates, val_params):
        params = {**val_params, "rA": 0.0, "rC": 0.0, "assort1": {0: 0.1, 2: 0.3}, "assort2": 0.2}
        result = validate_assortative_mating(params, val_mates)
        _assert_information(result)
        assert "expected" not in result["mate_corr_liability1"]
        assert result["mate_corr_liability1"]["observed"] is not None
        assert result["mate_corr_liability2"]["expected"] == pytest.approx(0.2)
        assert result["mate_corr_cross_12"]["expected"] == pytest.approx(0.0)
        assert "mate_corr_A1" in result

    def test_induced_cells_are_reported_without_a_target(self, val_mates, val_params):
        """Assortment on trait 2 alone induces the other cells; their nominal target is biased (ADR 0031)."""
        params = {**val_params, "rA": 1.0, "rC": 0.0, "A1": 0.5, "C1": 0.0, "E1": 0.5}
        params |= {"A2": 0.5, "C2": 0.0, "E2": 0.5, "assort1": 0.0, "assort2": 0.8}
        result = validate_assortative_mating(params, val_mates)
        _assert_information(result)
        assert result["mate_corr_liability2"]["expected"] == pytest.approx(0.8)
        for key in ("mate_corr_liability1", "mate_corr_cross_12", "mate_corr_cross_21"):
            assert "expected" not in result[key]
            assert result[key]["observed"] is not None

    def test_a_trait_random_in_one_generation_has_no_target(self, val_mates, val_params):
        """Its random generation is induced by the other trait's assortment there."""
        params = {**val_params, "rA": 0.0, "rC": 0.0, "assort1": {0: 0.0, 2: 0.3}, "assort2": 0.2}
        result = validate_assortative_mating(params, val_mates)
        assert "expected" not in result["mate_corr_liability1"]
        assert result["mate_corr_liability2"]["expected"] == pytest.approx(0.2)

    def test_cross_trait_assort_matrix_pools_under_per_generation_unique_environment(self, val_mates, val_params):
        """The simulator mates every generation on assort_matrix, whatever E does after the recorded matings."""
        params = {
            **val_params,
            "assort1": 0.2,
            "assort2": 0.3,
            "assort_matrix": [[0.2, 0.05], [0.05, 0.3]],
            "E1": {0: 0.3, 2: 0.2},
        }
        result = validate_assortative_mating(params, val_mates)
        for key in ("mate_corr_cross_12", "mate_corr_cross_21"):
            assert result[key]["expected"] == pytest.approx(0.05)

    def test_a_simulation_with_correlated_unique_environment_lands_near_its_target(self):
        """A correct simulation with rE != 0 and both traits assorting reports the rho_w target (#61)."""
        params = {
            **_DEFAULT_PARAMS,
            "N": 5000,
            "p_mztwin": 0.0,
            "A1": 0.2,
            "C1": 0.2,
            "E1": 0.6,
            "A2": 0.2,
            "C2": 0.2,
            "E2": 0.6,
            "rA": 0.0,
            "rC": 0.0,
            "rE": 0.8,
            "assort1": 0.8,
            "assort2": 0.8,
        }
        result = build_quality_report(run_simulation(**params), params)["assortative_mating"]
        _assert_information(result)
        for key in ("mate_corr_cross_12", "mate_corr_cross_21"):
            assert result[key]["expected"] == pytest.approx(0.8 * 0.6 * 0.8)
            assert result[key]["observed"] == pytest.approx(result[key]["expected"], abs=0.1)


# ---------------------------------------------------------------------------
# build_quality_report orchestrator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quality_report(val_pedigree, val_params):
    return build_quality_report(val_pedigree, val_params)


class TestBuildQualityReport:
    def test_returns_summary(self, quality_report):
        s = quality_report["summary"]
        assert s["checks_total"] == s["checks_passed"] + s["checks_failed"]
        assert s["passed"] is (s["checks_failed"] == 0)

    def test_top_level_categories_present(self, quality_report):
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
            assert cat in quality_report, f"missing category {cat!r}"

    def test_parameters_round_trip(self, quality_report, val_params):
        assert quality_report["parameters"]["A1"] == val_params["A1"]
        assert quality_report["parameters"]["seed"] == val_params["seed"]


# ---------------------------------------------------------------------------
# Checks with nothing to test are informational, not passes (#66)
# ---------------------------------------------------------------------------

# One recorded generation: no non-founders, siblings, twins or mating pairs.
_FOUNDERS_ONLY_PARAMS = {**_DEFAULT_PARAMS, "rE": 0.0, "N": 200, "G_ped": 1, "G_sim": 2, "p_mztwin": 0.0}

_NOTHING_TO_TEST = [
    ("structural", "sex_parent_consistency"),
    ("twins", "twin_bidirectional"),
    ("twins", "twin_same_parents"),
    ("twins", "twin_same_A1"),
    ("twins", "twin_same_A2"),
    ("twins", "twin_same_sex"),
    ("half_sibs", "half_sib_A1_correlation"),
    ("half_sibs", "half_sib_A2_correlation"),
    ("half_sibs", "half_sib_shared_C1"),
    ("half_sibs", "half_sib_shared_C2"),
    ("statistical", "e1_independence"),
    ("heritability", "mz_twin_A1_correlation"),
    ("heritability", "mz_twin_A2_correlation"),
    ("heritability", "dz_sibling_A1_correlation"),
    ("heritability", "dz_sibling_A2_correlation"),
    ("heritability", "falconer_estimate_trait1"),
    ("heritability", "falconer_estimate_trait2"),
    ("population", "family_size"),
    ("assortative_mating", "mate_corr_liability1"),
    ("assortative_mating", "mate_corr_liability2"),
]


def _assert_informational(result: dict) -> None:
    assert result.get("informational") is True, result
    assert "passed" not in result, result


@pytest.fixture(scope="module")
def founders_only_pedigree():
    return run_simulation(**_FOUNDERS_ONLY_PARAMS)


@pytest.fixture(scope="module")
def founders_only_report(founders_only_pedigree):
    return build_quality_report(founders_only_pedigree, _FOUNDERS_ONLY_PARAMS)


class TestNothingToTestIsInformational:
    @pytest.mark.parametrize(("category", "name"), _NOTHING_TO_TEST, ids=[f"{c}.{n}" for c, n in _NOTHING_TO_TEST])
    def test_check_without_data_is_informational(self, founders_only_report, category, name):
        _assert_informational(founders_only_report[category][name])

    def test_twin_rate_without_twins_is_still_scored(self, founders_only_pedigree):
        # Zero twins is evidence about the rate, so this check keeps its verdict.
        ped = PedigreeArrays.from_frame(founders_only_pedigree)
        assert validate_twins(founders_only_pedigree, _FOUNDERS_ONLY_PARAMS, ped)["twin_rate"]["passed"] is True
        high = {**_FOUNDERS_ONLY_PARAMS, "p_mztwin": 0.05}
        assert validate_twins(founders_only_pedigree, high, ped)["twin_rate"]["passed"] is False

    def test_twin_rate_without_non_founders_is_informational(self):
        params = {**_FOUNDERS_ONLY_PARAMS, "p_mztwin": 0.05}
        ped = run_simulation(**params)
        assert (ped["twin"] != -1).any(), "fixture sanity: founders must contain twins"
        _assert_informational(validate_twins(ped, params, PedigreeArrays.from_frame(ped))["twin_rate"])

    def test_e1_independence_without_sibling_groups_is_informational(self, val_pedigree, val_params):
        # Keep one child per mother, so no mother has two children to compare.
        founders = val_pedigree.filter(pl.col("mother") == -1)
        only_children = val_pedigree.filter(pl.col("mother") != -1).unique(
            subset="mother", keep="first", maintain_order=True
        )
        ped = pl.concat([founders, only_children])
        _assert_informational(validate_statistical(ped, val_params)["e1_independence"])

    @pytest.mark.parametrize(
        ("assort", "traits"),
        [
            pytest.param({"assort1": 0.3, "assort2": 0.3}, (1, 2), id="both-traits"),
            pytest.param({"assort1": {0: 0.3}, "assort2": 0.0}, (1,), id="per-generation"),
        ],
    )
    def test_am_equilibrium_outside_the_recursion_is_informational(self, val_pedigree, val_params, assort, traits):
        result = validate_am_equilibrium(val_pedigree, {**val_params, **assort})
        for t in traits:
            _assert_informational(result[f"am_equilibrium_A{t}"])

    def test_summary_counts_only_checks_that_ran(self, founders_only_report):
        scored = [
            name
            for category, checks in founders_only_report.items()
            if category not in ("per_generation", "summary", "family_size_distribution", "parameters")
            for name, result in checks.items()
            if "passed" in result
        ]
        assert founders_only_report["summary"]["checks_total"] == len(scored)
        assert not {name for _, name in _NOTHING_TO_TEST} & set(scored)


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
        ids = ped["id"].to_numpy().copy()
        ids[5] = ids.max() + 99
        ped = _with_column_values(ped, "id", ids)
        result = validate_structural(ped, self._tiny_params(), PedigreeArrays.from_frame(ped))
        assert result["id_integrity"]["passed"] is False

    def test_dangling_parent_id_fails_parent_references(self):
        ped = self._tiny_pedigree()
        params = self._tiny_params()
        # Force a mother index outside [0, N*G_ped) and not -1.
        mother = ped["mother"].to_numpy().copy()
        non_founder_idx = np.flatnonzero(mother != -1)[0]
        mother[non_founder_idx] = params["N"] * params["G_ped"] + 50
        ped = _with_column_values(ped, "mother", mother)
        result = validate_structural(ped, params, PedigreeArrays.from_frame(ped))
        assert result["parent_references"]["passed"] is False

    def test_wrong_parent_sex_fails_sex_consistency(self):
        ped = self._tiny_pedigree()
        mother_id = ped.filter(pl.col("mother") != -1)["mother"][0]
        sex = ped["sex"].to_numpy().copy()
        sex[ped["id"].to_numpy() == mother_id] = 1  # 1 = male
        ped = _with_column_values(ped, "sex", sex)
        result = validate_structural(ped, self._tiny_params(), PedigreeArrays.from_frame(ped))
        assert result["sex_parent_consistency"]["passed"] is False

    def test_non_bidirectional_twin_fails_bidirectional_check(self):
        """A twin pointer that doesn't bounce back must fail twin_bidirectional."""
        ped = self._tiny_pedigree()
        # Pick three contiguous IDs to wire as a broken twin chain.
        a, b, c = 0, 1, 2
        ids = ped["id"].to_numpy()
        twin = ped["twin"].to_numpy().copy()
        twin[ids == a] = b
        twin[ids == b] = c  # broken: should be `a` to be bidirectional
        twin[ids == c] = b
        ped = _with_column_values(ped, "twin", twin)
        params = self._tiny_params()
        params["p_mztwin"] = 0.02
        result = validate_twins(ped, params, PedigreeArrays.from_frame(ped))
        assert result["twin_bidirectional"]["passed"] is False


def _trait_for(pedigree, seed=0):
    """Minimal censored-trait frame for the trailing two generations."""
    max_gen = int(pedigree["generation"].max())
    phenotyped = pedigree.filter(pl.col("generation") >= max_gen - 1)
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "id": phenotyped["id"].to_numpy(),
            "generation": phenotyped["generation"].to_numpy(),
            "sex": phenotyped["sex"].to_numpy(),
            "affected1": rng.random(len(phenotyped)) < 0.2,
            "affected2": rng.random(len(phenotyped)) < 0.2,
        }
    )


@pytest.fixture(scope="module")
def gapped_pedigree(val_pedigree):
    """An ascertained pedigree: ids sorted but no longer contiguous.

    Built through run_ascertainment so parent references stay consistent --
    an ancestor closure with dangling links severed, which is what the
    pipeline actually writes to pedigree.parquet. A pedigree carrying
    *unsevered* references to absent ids is malformed, and raises here as it
    did before this work.
    """
    ascertained, _ = run_ascertainment(val_pedigree, _trait_for(val_pedigree), N_sample=250, seed=3)
    return ascertained


@pytest.fixture(scope="module")
def severed_pedigree(val_pedigree):
    """An ascertained pedigree carrying severed (-1) parent links."""
    ascertained, _ = run_ascertainment(val_pedigree, _trait_for(val_pedigree), dropout_rate=0.2, seed=7)
    return ascertained


class TestValidateOnAscertainedPedigrees:
    """Regression tests: validation used to crash on anything but a full pedigree.

    `build_quality_report` accepts any pedigree frame, so both shapes are
    supported inputs. Gapped ids raised IndexError out of consanguinity's
    id-indexed lookup (sized by row count, not max id); a severed parent raised
    KeyError out of assortative_mating's unguarded .loc.
    """

    def test_gapped_ids_do_not_raise(self, gapped_pedigree, val_params):
        ids = gapped_pedigree["id"].to_numpy()
        assert ids.max() >= len(ids), "fixture must have non-contiguous ids"
        report = build_quality_report(gapped_pedigree, val_params)
        assert report["summary"]["checks_total"] > 0

    def test_severed_parents_do_not_raise(self, severed_pedigree, val_params):
        mother = severed_pedigree["mother"].to_numpy()
        father = severed_pedigree["father"].to_numpy()
        assert ((mother == -1) ^ (father == -1)).any(), "fixture must have one-parent rows"
        report = build_quality_report(severed_pedigree, val_params)
        assert report["summary"]["checks_total"] > 0

    def test_consanguinity_skips_rows_with_a_severed_parent(self, severed_pedigree, val_params):
        """A row with one parent has no complete mating, so it is not eligible."""
        result = validate_consanguineous_matings(
            severed_pedigree, val_params, PedigreeArrays.from_frame(severed_pedigree)
        )
        assert "consanguineous_count" in result

    def test_mate_correlation_drops_pairs_with_an_absent_parent(self, severed_pedigree, val_params):
        """A row with a severed parent makes no mating pair."""
        mother = severed_pedigree["mother"].to_numpy()
        father = severed_pedigree["father"].to_numpy()
        both = (mother != -1) & (father != -1)
        n_pairs = len(set(zip(mother[both].tolist(), father[both].tolist(), strict=True)))
        params = {**val_params, "assort1": 0.3, "assort2": 0.0}
        result = validate_assortative_mating(
            params, mate_correlations(severed_pedigree, mate_pedigree(severed_pedigree))
        )
        assert result["mate_corr_liability1"]["n_pairs"] == n_pairs
