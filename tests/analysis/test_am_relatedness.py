"""Tests for AM-aware additive-genetic relative correlations."""

import pytest

from simace.analysis.validate import build_validation_report
from simace.analysis.validate.am_relatedness import (
    am_expected_a_correlation,
    am_relatedness_mode,
    observed_mate_correlations,
)
from simace.core.pedigree_arrays import PedigreeArrays
from simace.simulation.simulate import run_simulation


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
        assert am_relatedness_mode({"assort1": 0.0, "assort2": 0.0}, 1) == "none"
        assert am_relatedness_mode({"assort1": 0.5, "assort2": 0.0}, 1) == "single"
        assert am_relatedness_mode({"assort1": 0.5, "assort2": 0.0}, 2) == "none"
        assert am_relatedness_mode({"assort1": 0.5, "assort2": 0.5}, 1) == "bivariate"

    def test_wright_fisher_is_none(self):
        assert am_relatedness_mode({"assort1": 0.5, "mating_model": "wright_fisher"}, 1) == "none"


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
    def test_single_trait_am_fs_hs_pass_and_inflated(self):
        p = _params()
        df = run_simulation(**p)
        rep = build_validation_report(df, p)
        fs = rep["heritability"]["dz_sibling_A1_correlation"]
        hs = rep["half_sibs"]["half_sib_A1_correlation"]
        assert fs["passed"], fs["details"]
        assert hs["passed"], hs["details"]
        # AM-aware expected values exceed the random-mating relatedness
        assert fs["expected"] > 0.5
        assert hs["expected"] > 0.25
        # data-anchored mate correlations are recorded
        assert fs["mu_A"] > 0.0
        assert "r_ho" in hs

    def test_no_am_keeps_random_mating_expectation(self):
        p = _params(assort1=0.0)
        df = run_simulation(**p)
        rep = build_validation_report(df, p)
        assert rep["heritability"]["dz_sibling_A1_correlation"]["expected"] == pytest.approx(0.5)
        assert rep["half_sibs"]["half_sib_A1_correlation"]["expected"] == pytest.approx(0.25)

    def test_bivariate_am_reports_but_does_not_assert(self):
        """Under both-trait AM the single-trait formula does not apply, so FS/HS
        are reported informationally (no pass/fail) rather than scored."""
        p = _params(assort2=0.5)
        df = run_simulation(**p)
        rep = build_validation_report(df, p)
        fs = rep["heritability"]["dz_sibling_A1_correlation"]
        hs = rep["half_sibs"]["half_sib_A1_correlation"]
        # informational: no "passed" key, observed value still reported
        assert "passed" not in fs
        assert fs.get("informational") is True
        assert "not asserted" in fs["details"]
        assert "observed" in fs
        assert "passed" not in hs
        assert hs.get("informational") is True
        assert "not asserted" in hs["details"]

    def test_variance_a_reported_informational_under_am(self):
        """variance_A is reported (not asserted) under AM; C/E stay scored."""
        p = _params()
        df = run_simulation(**p)
        rep = build_validation_report(df, p)
        v_a = rep["statistical"]["variance_A1"]
        assert "passed" not in v_a
        assert v_a.get("informational") is True
        assert "observed" in v_a
        # the non-assorting trait and the C/E components remain scored
        assert "passed" in rep["statistical"]["variance_C1"]
        assert "passed" in rep["statistical"]["variance_A2"]

    def test_falconer_asserts_am_biased_value(self):
        """Under single-trait AM Falconer is scored against the AM-biased value
        Var(A)(1-mu_A)/V_P, which is below the configured A."""
        p = _params(p_mztwin=0.05, G_ped=12, G_sim=12, assort1=0.6)
        df = run_simulation(**p)
        rep = build_validation_report(df, p)
        fal = rep["heritability"]["falconer_estimate_trait1"]
        assert fal["passed"], fal["details"]
        assert "AM-biased" in fal["details"]
        assert fal["expected"] < p["A1"]  # downward AM bias

    def test_genetic_mate_correlation_reported(self):
        """validate_assortative_mating emits the genetic mate correlation mu_A."""
        p = _params(assort1=0.6)
        am = build_validation_report(run_simulation(**p), p)["assortative_mating"]
        assert "mate_corr_A1" in am
        assert am["mate_corr_A1"]["observed"] > 0.1  # AM induces genetic mate corr

        p0 = _params(assort1=0.0)
        am0 = build_validation_report(run_simulation(**p0), p0)["assortative_mating"]
        assert abs(am0["mate_corr_A1"]["observed"]) < 0.1  # no AM -> ~0

    def test_observed_mate_correlations_positive_under_am(self):
        p = _params(assort1=0.6)
        df = run_simulation(**p)
        mu_a, r_ho, n = observed_mate_correlations(df, PedigreeArrays.from_frame(df), 1)
        assert n > 100
        assert 0.2 < r_ho < 0.8  # configured 0.6, sampled
        assert mu_a > 0.1  # genetic mate correlation induced by AM
