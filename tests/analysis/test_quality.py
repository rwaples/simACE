"""Tests for simace.analysis.quality — Analyze's quality-check families."""

import numpy as np
import polars as pl
import pytest

from simace.analysis.mate_correlation import mate_pedigree
from simace.analysis.quality import (
    FamilyResult,
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
from simace.analysis.quality.runner import FAMILIES
from simace.analysis.stats.heritability import ge_h2_metrics
from simace.ascertainment import run_ascertainment
from simace.core.pedigree_arrays import PedigreeArrays
from simace.simulation.assortment import AssortmentPlan
from simace.simulation.params import Assortment, Span
from simace.simulation.simulate import resolve_per_gen_param, run_simulation
from tests.conftest import mating_records, sim_params

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
    return run_simulation(**_DEFAULT_PARAMS).pedigree


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
    return validate_heritability(
        val_pedigree, sim_params(val_params), val_ped, val_mate_pedigree, val_sibling_moments, val_mates
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_passed(result: FamilyResult | dict) -> None:
    """Assert every check in a family's result passed."""
    if isinstance(result, FamilyResult):
        for check in result.checks:
            assert check.passed, f"Check '{check.name}' failed: {check.message}"
        return
    for key, value in result.items():
        if isinstance(value, dict) and "passed" in value:
            assert value["passed"], f"Check '{key}' failed: {value.get('details', '')}"


def _names(result: FamilyResult) -> list[str]:
    return [check.name for check in result.checks]


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
        result = validate_structural(val_pedigree, sim_params(val_params), val_ped)
        _all_passed(result)

    def test_expected_keys(self, val_pedigree, val_params, val_ped):
        result = validate_structural(val_pedigree, sim_params(val_params), val_ped)
        assert _names(result) == ["id_integrity", "parent_references", "sex_parent_consistency", "sex_distribution"]
        assert result.measurements.sex_ratio_male == result.check("sex_distribution").observed


class TestValidateTwins:
    def test_all_checks_pass(self, val_pedigree, val_params, val_ped):
        result = validate_twins(val_pedigree, sim_params(val_params), val_ped)
        _all_passed(result)

    def test_twin_rate_present(self, val_pedigree, val_params, val_ped):
        result = validate_twins(val_pedigree, sim_params(val_params), val_ped)
        assert result.check("twin_rate").observed == result.measurements.twin_rate
        assert result.measurements.twin_rate is not None

    def test_wf_with_inherited_default_p_mztwin_scores_only_the_rate(self, val_pedigree, val_ped):
        # Under WF, inherited p_mztwin=0.02 must NOT trigger a failed twin_rate
        # check (the standard branch fails because 0.02 > 0.01).  Branch on
        # mating_model and report an observed and expected rate of 0.0. With
        # no twin records the structural twin checks do not run.
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
        ).pedigree
        result = validate_twins(wf_ped, sim_params(wf_params), PedigreeArrays.from_frame(wf_ped))
        rate = result.check("twin_rate")
        assert rate.passed is True
        assert rate.expected == 0.0
        assert rate.observed == 0.0
        assert _names(result) == ["twin_rate"]

    def test_wf_fails_when_twins_present(self, val_pedigree, val_ped):
        """If a pedigree labelled mating_model=wright_fisher contains any twins
        (regression / corruption), validate_twins must fail the rate check
        rather than silently pass.  The val_pedigree fixture is a standard-model
        pedigree with twins; relabelling its params as WF must surface the
        violation.
        """
        wf_params = {"mating_model": "wright_fisher", "p_mztwin": 0.02}
        result = validate_twins(val_pedigree, sim_params(wf_params), val_ped)
        assert (val_pedigree["twin"] != -1).any(), "fixture sanity: pedigree must contain twins"
        rate = result.check("twin_rate")
        assert rate.passed is False
        assert rate.expected == 0.0
        assert rate.observed > 0.0
        assert result.measurements.twin_rate == rate.observed


class TestValidateHalfSibs:
    def test_passes(self, val_pedigree, val_params, val_sibling_moments, val_mates):
        result = validate_half_sibs(val_pedigree, sim_params(val_params), val_sibling_moments, val_mates)
        _all_passed(result)

    def test_numeric_fields(self, val_pedigree, val_params, val_sibling_moments, val_mates):
        result = validate_half_sibs(val_pedigree, sim_params(val_params), val_sibling_moments, val_mates)
        assert isinstance(result.measurements.pair_proportion, float)
        for trait in result.measurements.traits:
            for corr in (trait.A_corr, trait.liability_corr, trait.shared_C):
                assert isinstance(corr.observed, float)
                assert corr.n_pairs >= 10


class TestValidateConsanguineous:
    def test_passes(self, val_pedigree, val_params, val_ped):
        result = validate_consanguineous_matings(val_pedigree, val_ped)
        _all_passed(result)

    def test_non_negative_counts(self, val_pedigree, val_params, val_ped):
        counts = validate_consanguineous_matings(val_pedigree, val_ped).measurements
        assert counts.n_half_sib_matings >= 0
        assert counts.n_full_sib_matings >= 0
        assert counts.total_missing_gp_links >= 0


class TestValidateStatistical:
    def test_all_checks_pass(self, val_pedigree, val_params):
        result = validate_statistical(val_pedigree, sim_params(val_params))
        _all_passed(result)

    def test_founder_e_without_g_sim_is_read_at_the_schedule_span_burn_in(self, val_pedigree, val_params):
        """With ``G_sim`` absent, G_sim falls back to G_ped and burn-in is 0, as in ``generation0`` (#79)."""
        params = {k: v for k, v in val_params.items() if k != "G_sim"} | {"E1": {0: 0.3, 1: 0.6}}
        assert validate_statistical(val_pedigree, sim_params(params)).check("variance_E1").expected == 0.3

    def test_variance_keys(self, val_pedigree, val_params):
        result = validate_statistical(val_pedigree, sim_params(val_params))
        for comp in ["A1", "C1", "E1", "A2", "C2", "E2"]:
            assert result.check(f"variance_{comp}").observed is not None

    def test_total_variance_and_rc_are_reported_only(self, val_pedigree, val_params):
        """`simace validate` tests them (ADR 0030); Analyze keeps their values for `truth`."""
        result = validate_statistical(val_pedigree, sim_params(val_params))
        for key in ("total_variance_trait1", "total_variance_trait2", "cross_trait_rC"):
            assert result.find(key) is None
        assert all(np.isfinite(trait.var_liability) for trait in result.measurements.traits)
        assert np.isfinite(result.measurements.rC)

    def test_cross_trait_rA_is_scored_under_random_mating(self, val_pedigree, val_params):
        assert validate_statistical(val_pedigree, sim_params(val_params)).check("cross_trait_rA").passed is True

    @pytest.mark.parametrize(
        "assort",
        [
            pytest.param({"assort1": 0.3}, id="trait1"),
            pytest.param({"assort2": 0.3}, id="trait2"),
            pytest.param({"assort1": {0: 0.0, 2: 0.3}}, id="per-generation"),
        ],
    )
    def test_cross_trait_rA_is_measured_only_under_assortative_mating(self, val_pedigree, val_params, assort):
        # AM shifts the founders' rA away from the configured value, as it
        # inflates Var(A), so it is measured and not checked.
        result = validate_statistical(val_pedigree, sim_params({**val_params, **assort}))
        random_mating = validate_statistical(val_pedigree, sim_params(val_params))
        assert result.find("cross_trait_rA") is None
        assert result.measurements.rA == random_mating.check("cross_trait_rA").observed
        # rE does not depend on assortment and stays checked.
        assert result.find("cross_trait_rE") is not None

    def test_cross_trait_rA_stays_scored_under_wright_fisher(self, val_pedigree, val_params):
        # Wright-Fisher pairs at random whatever assortment it inherits.
        params = {**val_params, "assort1": 0.3, "assort2": 0.3, "mating_model": "wright_fisher"}
        assert validate_statistical(val_pedigree, sim_params(params)).find("cross_trait_rA") is not None


class TestValidateHeritability:
    def test_mz_correlations_present(self, heritability_result):
        for t, trait in enumerate(heritability_result.measurements.traits, start=1):
            assert trait.mz_A_corr is not None
            assert heritability_result.check(f"mz_twin_A{t}_correlation").observed == trait.mz_A_corr.observed


class TestComputePerGenerationStats:
    def test_gen_size(self, val_pedigree, val_params):
        result = compute_per_generation_stats(val_pedigree, sim_params(val_params))
        for g in range(1, 4):
            assert result[f"generation_{g}"]["n"] == 1000

    def test_liability_stats_present(self, val_pedigree, val_params):
        result = compute_per_generation_stats(val_pedigree, sim_params(val_params))
        gen1 = result["generation_1"]
        assert "liability1_mean" in gen1
        assert "liability1_variance" in gen1
        assert "A1_var" in gen1

    def test_covariance_primitives_present(self, val_pedigree, val_params):
        result = compute_per_generation_stats(val_pedigree, sim_params(val_params))
        gen1 = result["generation_1"]
        assert "A1_cov_non_genetic" in gen1
        assert "A1_cov_C" in gen1
        assert "A1_cov_E" in gen1

    def test_covariance_uses_population_denominator(self):
        a = np.array([-1.0, 0.0, 1.0, 2.0])
        c = a.copy()
        e = np.zeros_like(a)
        df = _component_df(a, c, e)
        result = compute_per_generation_stats(df, sim_params({"N": len(df), "G_ped": 1}))["generation_1"]
        expected = np.mean((a - a.mean()) * ((c + e) - (c + e).mean()))
        assert result["A1_cov_non_genetic"] == expected
        assert result["A1_cov_non_genetic"] == result["A1_var"]

    def test_independent_constructed_non_genetic_covariance_is_zero(self):
        a = np.array([-1.0, -1.0, 1.0, 1.0])
        c = np.array([-1.0, 1.0, -1.0, 1.0])
        e = np.zeros_like(a)
        df = _component_df(a, c, e)
        result = compute_per_generation_stats(df, sim_params({"N": len(df), "G_ped": 1}))["generation_1"]
        assert result["A1_cov_non_genetic"] == 0.0

    def test_constructed_non_genetic_equals_a_gives_snp_like_identity(self):
        a = np.array([-1.0, 0.0, 1.0, 2.0])
        c = a.copy()
        e = np.zeros_like(a)
        df = _component_df(a, c, e)
        result = compute_per_generation_stats(df, sim_params({"N": len(df), "G_ped": 1}))["generation_1"]
        derived = ge_h2_metrics(result["A1_var"], result["liability1_variance"], result["A1_cov_non_genetic"])
        assert result["A1_cov_non_genetic"] == result["A1_var"]
        assert derived.ge_cov_fraction == 2 * result["A1_var"] / result["liability1_variance"]
        assert derived.h2_snp_like == 1.0


class TestValidatePopulation:
    def test_all_checks_pass(self, val_pedigree, val_params):
        result = validate_population(val_pedigree, sim_params(val_params))
        _all_passed(result)

    def test_generation_count_reports_the_number_of_labels(self, val_pedigree, val_params):
        count = validate_population(val_pedigree, sim_params(val_params)).check("generation_count")
        assert count.passed is True
        assert count.observed == count.expected == val_params["G_ped"]

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
        result = validate_population(ped, sim_params(val_params))
        assert result.check("generation_count").passed is False
        # The id ranges are untouched, so only the label check sees the damage.
        assert result.check("generation_sizes").passed is True


class TestComputeFamilySizeDistribution:
    def test_structure(self, val_pedigree, val_params):
        result = compute_family_size_distribution(val_pedigree)
        assert "mother" in result
        assert "father" in result
        for parent_type in ["mother", "father"]:
            entry = result[parent_type]
            assert "mean" in entry
            assert "median" in entry
            assert "n_parents" in entry

    def test_mean_around_two(self, val_pedigree, val_params):
        result = compute_family_size_distribution(val_pedigree)
        assert result["mother"]["mean"] == pytest.approx(2.0, abs=0.5)


class TestRecordedMatings:
    """The validator's targets follow the simulator's mating schedule (#64)."""

    def test_one_entry_per_recorded_parents_generation_after_burn_in(self, val_params):
        params = {**val_params, "G_sim": 6, "G_ped": 4, "assort1": {0: 0.1, 4: 0.4}, "assort2": 0.3}
        matings = recorded_matings(*mating_records(params))
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
        assort1, assort2 = Assortment.create(Span(4, 6), params["assort1"], params["assort2"]).schedule
        plan = AssortmentPlan.build(
            assort1_per_gen=list(assort1),
            assort2_per_gen=list(assort2),
            R_mf_user=None,
            rA=params["rA"],
            rC=params["rC"],
            rE=params["rE"],
            A1=params["A1"],
            A2=params["A2"],
            C1=params["C1"],
            C2=params["C2"],
            E1_per_gen=e1,
            E2_per_gen=[params["E2"]] * 6,
            G_sim=6,
        )
        for m in recorded_matings(*mating_records(params)):
            R_mf = plan.for_generation(m.parents_generation + 3)[3]
            np.testing.assert_allclose(m.expected, R_mf)
            assert m.cross_scored

    def test_a_single_recorded_generation_has_no_matings(self, val_params):
        assert recorded_matings(*mating_records({**val_params, "G_sim": 1, "G_ped": 1, "assort1": 0.3})) == []

    def test_assort_matrix_diagonal_overrides_the_scalar_assortment(self, val_params):
        """The simulator mates on the matrix diagonal; params.yaml echoes the raw scalars."""
        matrix = [[0.3, 0.05], [0.05, 0.4]]
        params = {**val_params, "assort1": 0.0, "assort2": 0.0, "assort_matrix": matrix}
        for m in recorded_matings(*mating_records(params)):
            np.testing.assert_allclose(m.expected, matrix)
            assert m.cross_scored

    def test_single_trait_assortment_induces_the_other_cells(self, val_params):
        params = {**val_params, "rA": 1.0, "rC": 0.0, "A1": 0.5, "C1": 0.0, "E1": 0.5}
        params |= {"A2": 0.5, "C2": 0.0, "E2": 0.5, "assort1": 0.0, "assort2": 0.8}
        for m in recorded_matings(*mating_records(params)):
            np.testing.assert_allclose(m.expected, [[0.2, 0.4], [0.4, 0.8]])
            assert not m.cross_scored

    def test_wright_fisher_targets_are_zero_whatever_assortment_it_inherits(self, val_params):
        params = {**val_params, "mating_model": "wright_fisher", "assort1": 0.3, "assort2": 0.3}
        for m in recorded_matings(*mating_records(params)):
            np.testing.assert_array_equal(m.expected, np.zeros((2, 2)))
            assert not m.cross_scored

    def test_cross_is_scored_only_where_both_traits_assort(self, val_params):
        params = {**val_params, "G_sim": 4, "G_ped": 4, "assort1": {0: 0.0, 2: 0.3}, "assort2": 0.3}
        assert [m.cross_scored for m in recorded_matings(*mating_records(params))] == [False, True, True]


class TestValidateAssortativeMating:
    def test_measures_the_liability_and_genetic_mate_correlations(self, val_params, val_mates):
        result = validate_assortative_mating(sim_params(val_params), val_mates)
        assert result.checks == ()
        for t in (1, 2):
            assert result.measurements.liability[t - 1] == pytest.approx(val_mates.liability[t - 1, t - 1])
            assert result.measurements.genetic[t - 1] == pytest.approx(val_mates.genetic[t - 1])

    def test_no_recorded_mating_measures_nothing(self, val_params, val_mates):
        result = validate_assortative_mating(sim_params({**val_params, "G_sim": 1, "G_ped": 1}), val_mates)
        assert result.measurements.liability == (None, None)
        assert result.measurements.genetic == (None, None)


# ---------------------------------------------------------------------------
# build_quality_report orchestrator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quality_report(val_pedigree, val_params):
    return build_quality_report(val_pedigree, sim_params(val_params))


class TestBuildQualityReport:
    def test_rows_follow_family_order(self, quality_report):
        families = [row["id"].split(".")[0] for row in quality_report.rows()]
        assert families == sorted(families, key=FAMILIES.index)
        assert set(families) <= set(FAMILIES)

    def test_every_check_passes_on_a_correct_pedigree(self, quality_report):
        failed = [row["id"] for row in quality_report.rows() if row["status"] != "pass"]
        assert not failed

    def test_summaries_present(self, quality_report):
        assert set(quality_report.per_generation) == {"generation_1", "generation_2", "generation_3"}
        assert set(quality_report.family_size_distribution) == {"mother", "father"}


# ---------------------------------------------------------------------------
# Checks with nothing to test are not checked, not passes (#66)
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


def _assert_not_checked(report, category: str, name: str) -> None:
    """``category.name`` has no verdict on this rep: the family asserts no such check."""
    assert getattr(report, category).find(name) is None


@pytest.fixture(scope="module")
def founders_only_pedigree():
    return run_simulation(**_FOUNDERS_ONLY_PARAMS).pedigree


@pytest.fixture(scope="module")
def founders_only_report(founders_only_pedigree):
    return build_quality_report(founders_only_pedigree, sim_params(_FOUNDERS_ONLY_PARAMS))


class TestNothingToTestIsNotChecked:
    @pytest.mark.parametrize(("category", "name"), _NOTHING_TO_TEST, ids=[f"{c}.{n}" for c, n in _NOTHING_TO_TEST])
    def test_check_without_data_is_not_checked(self, founders_only_report, category, name):
        _assert_not_checked(founders_only_report, category, name)

    def test_twin_rate_without_twins_is_still_scored(self, founders_only_pedigree):
        # Zero twins is evidence about the rate, so this check keeps its verdict.
        ped = PedigreeArrays.from_frame(founders_only_pedigree)
        assert validate_twins(founders_only_pedigree, sim_params(_FOUNDERS_ONLY_PARAMS), ped).check("twin_rate").passed
        high = {**_FOUNDERS_ONLY_PARAMS, "p_mztwin": 0.05}
        assert validate_twins(founders_only_pedigree, sim_params(high), ped).check("twin_rate").passed is False

    def test_twin_rate_without_non_founders_is_not_checked(self):
        params = {**_FOUNDERS_ONLY_PARAMS, "p_mztwin": 0.05}
        ped = run_simulation(**params).pedigree
        assert (ped["twin"] != -1).any(), "fixture sanity: founders must contain twins"
        result = validate_twins(ped, sim_params(params), PedigreeArrays.from_frame(ped))
        assert result.find("twin_rate") is None
        assert result.measurements.twin_rate is None

    def test_e1_independence_without_sibling_groups_is_not_checked(self, val_pedigree, val_params):
        # Keep one child per mother, so no mother has two children to compare.
        founders = val_pedigree.filter(pl.col("mother") == -1)
        only_children = val_pedigree.filter(pl.col("mother") != -1).unique(
            subset="mother", keep="first", maintain_order=True
        )
        ped = pl.concat([founders, only_children])
        assert validate_statistical(ped, sim_params(val_params)).find("e1_independence") is None

    @pytest.mark.parametrize(
        ("assort", "traits"),
        [
            pytest.param({"assort1": 0.3, "assort2": 0.3}, (1, 2), id="both-traits"),
            pytest.param({"assort1": {0: 0.3, 1: 0.1}, "assort2": 0.0}, (1,), id="per-generation"),
        ],
    )
    def test_am_equilibrium_outside_the_recursion_is_not_checked(self, val_pedigree, val_params, assort, traits):
        result = validate_am_equilibrium(val_pedigree, sim_params({**val_params, **assort}))
        for t in traits:
            assert result.find(f"am_equilibrium_A{t}") is None

    def test_rows_hold_only_checks_that_ran(self, founders_only_report):
        scored = {row["id"] for row in founders_only_report.rows()}
        assert not {f"{c}.{n}" for c, n in _NOTHING_TO_TEST} & scored


class TestValidateNegativePaths:
    """Corrupted pedigrees are caught by the appropriate structural / twin check."""

    @staticmethod
    def _tiny_params():
        return {**_DEFAULT_PARAMS, "rE": 0.0, "N": 100, "G_ped": 2, "G_sim": 2}

    @staticmethod
    def _tiny_pedigree():
        return run_simulation(**TestValidateNegativePaths._tiny_params()).pedigree

    def test_non_contiguous_ids_fails_id_integrity(self):
        ped = self._tiny_pedigree()
        # Skip an integer in the id column — sort(ids) != arange(N*G_ped).
        ids = ped["id"].to_numpy().copy()
        ids[5] = ids.max() + 99
        ped = _with_column_values(ped, "id", ids)
        result = validate_structural(ped, sim_params(self._tiny_params()), PedigreeArrays.from_frame(ped))
        assert result.check("id_integrity").passed is False

    def test_dangling_parent_id_fails_parent_references(self):
        ped = self._tiny_pedigree()
        params = self._tiny_params()
        # Force a mother index outside [0, N*G_ped) and not -1.
        mother = ped["mother"].to_numpy().copy()
        non_founder_idx = np.flatnonzero(mother != -1)[0]
        mother[non_founder_idx] = params["N"] * params["G_ped"] + 50
        ped = _with_column_values(ped, "mother", mother)
        result = validate_structural(ped, sim_params(params), PedigreeArrays.from_frame(ped))
        assert result.check("parent_references").passed is False

    def test_wrong_parent_sex_fails_sex_consistency(self):
        ped = self._tiny_pedigree()
        mother_id = ped.filter(pl.col("mother") != -1)["mother"][0]
        sex = ped["sex"].to_numpy().copy()
        sex[ped["id"].to_numpy() == mother_id] = 1  # 1 = male
        ped = _with_column_values(ped, "sex", sex)
        result = validate_structural(ped, sim_params(self._tiny_params()), PedigreeArrays.from_frame(ped))
        assert result.check("sex_parent_consistency").passed is False

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
        result = validate_twins(ped, sim_params(params), PedigreeArrays.from_frame(ped))
        assert result.check("twin_bidirectional").passed is False


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
        report = build_quality_report(gapped_pedigree, sim_params(val_params))
        assert report.rows()

    def test_severed_parents_do_not_raise(self, severed_pedigree, val_params):
        mother = severed_pedigree["mother"].to_numpy()
        father = severed_pedigree["father"].to_numpy()
        assert ((mother == -1) ^ (father == -1)).any(), "fixture must have one-parent rows"
        report = build_quality_report(severed_pedigree, sim_params(val_params))
        assert report.rows()

    def test_consanguinity_skips_rows_with_a_severed_parent(self, severed_pedigree, val_params):
        """A row with one parent has no complete mating, so it is not eligible."""
        result = validate_consanguineous_matings(severed_pedigree, PedigreeArrays.from_frame(severed_pedigree))
        assert result.measurements.n_half_sib_matings is not None

    def test_mate_correlation_drops_pairs_with_an_absent_parent(self, severed_pedigree, val_params):
        """A row with a severed parent makes no mating pair."""
        mother = severed_pedigree["mother"].to_numpy()
        father = severed_pedigree["father"].to_numpy()
        both = (mother != -1) & (father != -1)
        n_pairs = len(set(zip(mother[both].tolist(), father[both].tolist(), strict=True)))
        params = {**val_params, "assort1": 0.3, "assort2": 0.0}
        mates = mate_correlations(severed_pedigree, mate_pedigree(severed_pedigree))
        assert mates.n_pairs == n_pairs
        assert validate_assortative_mating(sim_params(params), mates).measurements.liability[0] is not None
