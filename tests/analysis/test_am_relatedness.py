"""Tests for AM-aware additive-genetic relative correlations."""

import math

import pytest

from simace.analysis.mate_correlation import mate_pedigree
from simace.analysis.quality import build_quality_report
from simace.analysis.quality._common import mate_correlations
from simace.analysis.quality.am_relatedness import (
    am_expected_a_correlation,
    am_relatedness_mode,
    observed_mate_correlations,
)
from simace.simulation.simulate import run_simulation
from tests.conftest import sim_params


class TestExpectedFormulas:
    def test_reduces_to_relatedness_at_no_am(self):
        assert am_expected_a_correlation("FS", 0.0, 0.0) == pytest.approx(0.5)
        assert am_expected_a_correlation("PO", 0.0, 0.0) == pytest.approx(0.5)
        assert am_expected_a_correlation("HS", 0.0, 0.0) == pytest.approx(0.25)
        assert am_expected_a_correlation("MZ", 0.0, 0.0) == pytest.approx(1.0)

    def test_am_inflates(self):
        mu, r = 0.4, 0.6
        assert am_expected_a_correlation("FS", mu, r) == pytest.approx((1 + mu) / 2)
        assert am_expected_a_correlation("PO", mu, r) == pytest.approx((1 + mu) / 2)
        assert am_expected_a_correlation("HS", mu, r) == pytest.approx((1 + 2 * mu + mu * r) / 4)
        assert am_expected_a_correlation("MZ", mu, r) == 1.0
        # AM raises FS and HS above their random-mating relatedness
        assert am_expected_a_correlation("FS", mu, r) > 0.5
        assert am_expected_a_correlation("HS", mu, r) > 0.25

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown relationship"):
            am_expected_a_correlation("COUSIN", 0.3, 0.5)


class TestMode:
    def test_modes(self):
        assert am_relatedness_mode(sim_params({"assort1": 0.0, "assort2": 0.0}).assortment, 1) == "none"
        assert am_relatedness_mode(sim_params({"assort1": 0.5, "assort2": 0.0}).assortment, 1) == "single"
        assert am_relatedness_mode(sim_params({"assort1": 0.5, "assort2": 0.0}).assortment, 2) == "none"
        assert am_relatedness_mode(sim_params({"assort1": 0.5, "assort2": 0.5}).assortment, 1) == "bivariate"

    def test_wright_fisher_is_none(self):
        assert (
            am_relatedness_mode(sim_params({"assort1": 0.5, "mating_model": "wright_fisher"}).assortment, 1) == "none"
        )

    def test_matrix_diagonal_replaces_scalars(self):
        """The simulator mates on the matrix diagonal, whatever assort1/assort2 say (#93)."""
        matrix_only = {"assort1": 0.0, "assort2": 0.0, "assort_matrix": [[0.4, 0.0], [0.0, 0.0]]}
        assert am_relatedness_mode(sim_params(matrix_only).assortment, 1) == "single"
        assert am_relatedness_mode(sim_params(matrix_only).assortment, 2) == "none"
        both = {"assort1": 0.0, "assort2": 0.0, "assort_matrix": [[0.4, 0.1], [0.1, 0.3]]}
        assert am_relatedness_mode(sim_params(both).assortment, 2) == "bivariate"


def _params(**over):
    base = dict(
        seed=4,
        N=3000,
        G_ped=8,
        G_sim=8,
        mating_lambda=1.0,
        p_mztwin=0.0,
        A1=0.6,
        C1=0.0,
        E1=0.4,
        A2=0.6,
        C2=0.0,
        E2=0.4,
        rA=0.0,
        rC=0.0,
        rE=0.0,
        assort1=0.5,
        assort2=0.0,
        mating_model="standard",
    )
    base.update(over)
    return base


class TestIntegration:
    def test_single_trait_am_fs_hs_are_reported_with_inflated_expectations(self):
        """Measured, not checked: `simace validate`'s mendelian_residual tests the inheritance (ADR 0029)."""
        p = _params()
        df = run_simulation(**p).pedigree
        rep = build_quality_report(df, sim_params(p))
        fs = rep.heritability.measurements.traits[0].dz_A_corr
        hs = rep.half_sibs.measurements.traits[0].A_corr
        assert rep.half_sibs.checks == ()
        assert fs.observed == pytest.approx(fs.expected, abs=0.05)
        assert hs.observed == pytest.approx(hs.expected, abs=0.05)
        # AM-aware expected values exceed the random-mating relatedness
        assert fs.expected > 0.5
        assert hs.expected > 0.25
        # data-anchored mate correlations are recorded
        assert fs.mate_terms.mu_A > 0.0
        assert hs.mate_terms is not None

    def test_no_am_keeps_random_mating_expectation(self):
        p = _params(assort1=0.0)
        df = run_simulation(**p).pedigree
        rep = build_quality_report(df, sim_params(p))
        assert rep.heritability.measurements.traits[0].dz_A_corr.expected == pytest.approx(0.5)
        assert rep.half_sibs.measurements.traits[0].A_corr.expected == pytest.approx(0.25)

    def test_bivariate_am_reports_but_does_not_assert(self):
        """Under both-trait AM the single-trait formula does not apply, so FS/HS
        carry no expectation; their observed values are still measured."""
        p = _params(assort2=0.5)
        df = run_simulation(**p).pedigree
        rep = build_quality_report(df, sim_params(p))
        fs = rep.heritability.measurements.traits[0].dz_A_corr
        hs = rep.half_sibs.measurements.traits[0].A_corr
        assert fs.expected is None
        assert fs.mate_terms is None
        assert math.isfinite(fs.observed)
        assert hs.expected is None

    def test_matrix_only_assortment_is_checked_as_am(self):
        """A matrix with the scalars left at 0 mates exactly like matching scalars, so the report must agree (#93)."""
        matrix = [[0.5, 0.0], [0.0, 0.0]]
        p_matrix = _params(assort1=0.0, assort_matrix=matrix)
        p_scalar = _params(assort_matrix=matrix)
        df = run_simulation(**p_matrix).pedigree
        assert df.equals(run_simulation(**p_scalar).pedigree)
        reports = [build_quality_report(df, sim_params(p)) for p in (p_matrix, p_scalar)]
        assert repr(reports[0]) == repr(reports[1])  # repr: NaN observations compare equal

    def test_variance_a_measured_not_checked_under_am(self):
        """variance_A is measured (not checked) under AM; C/E stay checked."""
        p = _params()
        df = run_simulation(**p).pedigree
        statistical = build_quality_report(df, sim_params(p)).statistical
        assert statistical.find("variance_A1") is None
        assert statistical.measurements.traits[0].var_A > 0
        # the non-assorting trait and the C/E components remain checked
        assert statistical.find("variance_C1") is not None
        assert statistical.find("variance_A2") is not None

    def test_falconer_asserts_am_biased_value(self):
        """Under single-trait AM Falconer is scored against the AM-biased value
        Var(A)(1-mu_A)/V_P, which is below the configured A."""
        p = _params(p_mztwin=0.05, G_ped=12, G_sim=12, assort1=0.6)
        df = run_simulation(**p).pedigree
        rep = build_quality_report(df, sim_params(p))
        fal = rep.heritability.check("falconer_estimate_trait1")
        assert fal.passed, fal.message
        assert "AM-biased" in fal.message
        assert fal.expected < p["A1"]  # downward AM bias

    def test_genetic_mate_correlation_reported(self):
        """validate_assortative_mating emits the genetic mate correlation mu_A."""
        p = _params(assort1=0.6)
        am = build_quality_report(run_simulation(**p).pedigree, sim_params(p)).assortative_mating
        assert am.measurements.genetic[0] > 0.1  # AM induces genetic mate corr

        p0 = _params(assort1=0.0)
        am0 = build_quality_report(run_simulation(**p0).pedigree, sim_params(p0)).assortative_mating
        assert abs(am0.measurements.genetic[0]) < 0.1  # no AM -> ~0

    def test_observed_mate_correlations_positive_under_am(self):
        p = _params(assort1=0.6)
        df = run_simulation(**p).pedigree
        mu_a, r_ho, n = observed_mate_correlations(mate_correlations(df, mate_pedigree(df)), 1)
        assert n > 100
        assert 0.2 < r_ho < 0.8  # configured 0.6, sampled
        assert mu_a > 0.1  # genetic mate correlation induced by AM
