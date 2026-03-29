"""Unit tests for PA-FGRS implementation."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, truncnorm

from fit_ace.pafgrs.pafgrs import (
    _compute_depth,
    _nb_trunc_norm,
    _nb_trunc_norm_above,
    _nb_trunc_norm_below,
    _nb_trunc_norm_mixture,
    _trunc_norm_above_py,
    _trunc_norm_below_py,
    _trunc_norm_mixture_py,
    _trunc_norm_py,
    build_kinship_from_pairs,
    build_sparse_kinship,
    compute_empirical_cip,
    compute_thresholds_and_w,
    compute_thresholds_and_w_by_sex,
    compute_true_cip_weibull,
    pa_fgrs,
    pa_fgrs_adt,
)

# Alias old names for existing tests → Python fallback
_trunc_norm_below = _trunc_norm_below_py
_trunc_norm_above = _trunc_norm_above_py
_trunc_norm = _trunc_norm_py
_trunc_norm_mixture = _trunc_norm_mixture_py


# ---------------------------------------------------------------------------
# Truncated normal helpers
# ---------------------------------------------------------------------------


class TestTruncNormBelow:
    def test_agrees_with_scipy(self):
        """Compare against scipy.stats.truncnorm for X ~ N(0,1) | X < 1."""
        mu, var, trunc = 0.0, 1.0, 1.0
        m, v = _trunc_norm_below(mu, var, trunc)
        dist = truncnorm(-np.inf, (trunc - mu) / np.sqrt(var), loc=mu, scale=np.sqrt(var))
        assert m == pytest.approx(dist.mean(), abs=1e-10)
        assert v == pytest.approx(dist.var(), abs=1e-10)

    def test_far_left_truncation(self):
        """When truncation is far below mean, moments collapse toward the bound."""
        m, v = _trunc_norm_below(0.0, 1.0, -5.0)
        assert m < -4.9
        assert v < 0.1


class TestTruncNormAbove:
    def test_agrees_with_scipy(self):
        """Compare against scipy for X ~ N(0,1) | X > -0.5."""
        mu, var, trunc = 0.0, 1.0, -0.5
        m, v = _trunc_norm_above(mu, var, trunc)
        dist = truncnorm((trunc - mu) / np.sqrt(var), np.inf, loc=mu, scale=np.sqrt(var))
        assert m == pytest.approx(dist.mean(), abs=1e-10)
        assert v == pytest.approx(dist.var(), abs=1e-10)


class TestTruncNorm:
    def test_below_only(self):
        """Infinite lower → delegates to trunc_norm_below."""
        m1, v1 = _trunc_norm(0.0, 1.0, -np.inf, 1.5)
        m2, v2 = _trunc_norm_below(0.0, 1.0, 1.5)
        assert m1 == pytest.approx(m2)
        assert v1 == pytest.approx(v2)

    def test_above_only(self):
        """Infinite upper → delegates to trunc_norm_above."""
        m1, v1 = _trunc_norm(0.0, 1.0, 0.5, np.inf)
        m2, v2 = _trunc_norm_above(0.0, 1.0, 0.5)
        assert m1 == pytest.approx(m2)
        assert v1 == pytest.approx(v2)

    def test_point_mass(self):
        """Equal bounds → point mass at that value."""
        m, v = _trunc_norm(0.0, 1.0, 2.0, 2.0)
        assert m == pytest.approx(2.0)
        assert v == pytest.approx(0.0)

    def test_doubly_truncated(self):
        """Doubly truncated mean should be between bounds."""
        m, v = _trunc_norm(0.0, 1.0, -1.0, 1.0)
        assert -1.0 < m < 1.0
        assert 0.0 < v < 1.0


class TestTruncNormMixture:
    def test_no_mixture_when_kp_zero(self):
        """kp=0 → no mixture, reduces to trunc_norm."""
        m1, v1 = _trunc_norm_mixture(0.0, 1.0, -np.inf, 1.0, 0.0)
        m2, v2 = _trunc_norm(0.0, 1.0, -np.inf, 1.0)
        assert m1 == pytest.approx(m2)
        assert v1 == pytest.approx(v2)

    def test_case_no_mixture(self):
        """Upper=inf (case) → no mixture, trunc above."""
        m1, v1 = _trunc_norm_mixture(0.0, 1.0, 1.0, np.inf, 0.1)
        m2, v2 = _trunc_norm_above(0.0, 1.0, 1.0)
        assert m1 == pytest.approx(m2)
        assert v1 == pytest.approx(v2)

    def test_mixture_shifts_mean_upward(self):
        """With positive kp, mixture mean should exceed the below-only mean."""
        thr = norm.ppf(0.9)  # threshold for 10% prevalence
        kp = 0.5 * 0.1  # w=0.5, prevalence=0.1
        m_mix, _ = _trunc_norm_mixture(0.0, 1.0, -np.inf, thr, kp)
        m_below, _ = _trunc_norm_below(0.0, 1.0, thr)
        assert m_mix > m_below


# ---------------------------------------------------------------------------
# Kinship matrix
# ---------------------------------------------------------------------------


def _make_simple_pedigree():
    """3-generation pedigree: 2 founders → 2 children → 1 grandchild.

    Gen 0: ids 0 (mom) and 1 (dad) — founders
    Gen 1: ids 2 and 3 — full siblings (children of 0 and 1)
    Gen 2: id 4 — child of 2 (mother) and unrelated founder 5
    Gen 2: id 5 — founder (father of 4)
    """
    ids = np.array([0, 1, 2, 3, 4, 5])
    mothers = np.array([-1, -1, 0, 0, 2, -1])
    fathers = np.array([-1, -1, 1, 1, 5, -1])
    twins = np.array([-1, -1, -1, -1, -1, -1])
    return ids, mothers, fathers, twins


class TestBuildSparseKinship:
    def test_self_kinship(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        K = build_sparse_kinship(ids, mothers, fathers, twins)
        # Non-inbred individuals: self-kinship = 0.5
        for i in range(len(ids)):
            assert K[i, i] == pytest.approx(0.5), f"Self-kinship of {i}"

    def test_parent_offspring(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        K = build_sparse_kinship(ids, mothers, fathers, twins)
        # Parent-offspring kinship = 0.25
        assert K[0, 2] == pytest.approx(0.25)  # mom → child
        assert K[1, 2] == pytest.approx(0.25)  # dad → child
        assert K[0, 3] == pytest.approx(0.25)  # mom → child
        assert K[2, 0] == pytest.approx(0.25)  # symmetric

    def test_full_siblings(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        K = build_sparse_kinship(ids, mothers, fathers, twins)
        # Full siblings: kinship = 0.25
        assert K[2, 3] == pytest.approx(0.25)

    def test_grandparent(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        K = build_sparse_kinship(ids, mothers, fathers, twins)
        # Grandparent-grandchild: kinship = 0.125
        assert K[0, 4] == pytest.approx(0.125)  # grandmother → grandchild
        assert K[1, 4] == pytest.approx(0.125)  # grandfather → grandchild

    def test_unrelated(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        K = build_sparse_kinship(ids, mothers, fathers, twins)
        # Founders unrelated to each other (except couples)
        assert K[0, 1] == pytest.approx(0.0)  # mom and dad unrelated
        assert K[0, 5] == pytest.approx(0.0)  # founder 0 and founder 5

    def test_symmetry(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        K = build_sparse_kinship(ids, mothers, fathers, twins)
        dense = K.toarray()
        np.testing.assert_array_almost_equal(dense, dense.T)

    def test_mz_twins(self):
        """MZ twins should have kinship = 0.5 (same as self-kinship)."""
        ids = np.array([0, 1, 2, 3])
        mothers = np.array([-1, -1, 0, 0])
        fathers = np.array([-1, -1, 1, 1])
        twins = np.array([-1, -1, 3, 2])  # 2 and 3 are MZ twins
        K = build_sparse_kinship(ids, mothers, fathers, twins)
        assert K[2, 3] == pytest.approx(0.5)


class TestComputeDepth:
    def test_simple(self):
        m = np.array([-1, -1, 0, 0, 2])
        f = np.array([-1, -1, 1, 1, 3])
        depth = _compute_depth(m, f, 5)
        np.testing.assert_array_equal(depth, [0, 0, 1, 1, 2])


# ---------------------------------------------------------------------------
# PA-FGRS algorithm
# ---------------------------------------------------------------------------


class TestPaFgrs:
    def test_no_relatives_returns_prior(self):
        """No relatives → posterior = prior (0 mean, h2 variance)."""
        covmat = np.array([[0.5]])  # h2 = 0.5, no relatives
        est, var = pa_fgrs(
            rel_status=np.array([], dtype=bool),
            rel_thr=np.array([]),
            rel_w=np.array([]),
            covmat=covmat,
        )
        assert est == pytest.approx(0.0)
        assert var == pytest.approx(0.5)

    def test_affected_relative_positive_score(self):
        """One affected first-degree relative → positive estimated liability."""
        h2 = 0.5
        kinship = 0.25  # parent-offspring
        covmat = np.array(
            [
                [h2, 2 * kinship * h2],
                [2 * kinship * h2, 1.0],
            ]
        )
        thr = norm.ppf(0.9)  # 10% prevalence
        est, var = pa_fgrs(
            rel_status=np.array([True]),
            rel_thr=np.array([thr]),
            rel_w=np.array([1.0]),
            covmat=covmat,
        )
        assert est > 0, "Affected relative should shift estimate upward"
        assert var < h2, "Posterior variance should shrink"

    def test_unaffected_relative_negative_score(self):
        """One unaffected first-degree relative (fully observed) → negative shift."""
        h2 = 0.5
        kinship = 0.25
        covmat = np.array(
            [
                [h2, 2 * kinship * h2],
                [2 * kinship * h2, 1.0],
            ]
        )
        thr = norm.ppf(0.9)
        est, _var = pa_fgrs(
            rel_status=np.array([False]),
            rel_thr=np.array([thr]),
            rel_w=np.array([1.0]),
            covmat=covmat,
        )
        assert est < 0, "Unaffected relative should shift estimate downward"

    def test_more_relatives_more_precision(self):
        """Two affected relatives should give more precise estimate than one."""
        h2 = 0.5
        thr = norm.ppf(0.9)

        # One parent
        covmat1 = np.array(
            [
                [h2, 0.25],
                [0.25, 1.0],
            ]
        )
        _, var1 = pa_fgrs(
            np.array([True]),
            np.array([thr]),
            np.array([1.0]),
            covmat1,
        )

        # Two parents
        covmat2 = np.array(
            [
                [h2, 0.25, 0.25],
                [0.25, 1.0, 0.0],
                [0.25, 0.0, 1.0],
            ]
        )
        _, var2 = pa_fgrs(
            np.array([True, True]),
            np.array([thr, thr]),
            np.array([1.0, 1.0]),
            covmat2,
        )
        assert var2 < var1, "Two relatives should give lower variance"

    def test_adt_no_mixture(self):
        """PA-FGRS[adt] should give a result (no crash) for simple case."""
        h2 = 0.5
        covmat = np.array([[h2, 0.25], [0.25, 1.0]])
        thr = norm.ppf(0.9)
        est, var = pa_fgrs_adt(
            np.array([True]),
            np.array([thr]),
            covmat,
        )
        assert est > 0
        assert 0 < var < h2

    def test_continuous_reduces_to_regression(self):
        """When t1==t2 (continuous trait), PA reduces to linear regression.

        For a single relative with value x, covmat = [[h2, cov], [cov, 1]]:
        E[A|X=x] = cov/1 * x = cov * x
        Var[A|X=x] = h2 - cov^2
        """
        h2 = 0.5
        cov_val = 0.25
        covmat = np.array([[h2, cov_val], [cov_val, 1.0]])
        # Continuous: pass value as both t1 and t2 (point mass)
        from fit_ace.pafgrs.pafgrs import _pa_fgrs_core_py as _pa_fgrs_core

        x = 1.5
        est, var = _pa_fgrs_core(
            np.array([x]),
            np.array([x]),
            np.array([1.0]),
            covmat,
        )
        assert est == pytest.approx(cov_val * x, abs=1e-10)
        assert var == pytest.approx(h2 - cov_val**2, abs=1e-10)


# ---------------------------------------------------------------------------
# CIP computation
# ---------------------------------------------------------------------------


class TestEmpiricalCip:
    def test_basic(self):
        """CIP should be non-decreasing and bounded by [0, prevalence]."""
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame(
            {
                "affected1": rng.random(n) < 0.1,
                "t_observed1": rng.uniform(0, 80, n),
            }
        )
        _ages, cip, prev = compute_empirical_cip(df, trait_num=1)
        assert np.all(np.diff(cip) >= -1e-10), "CIP should be non-decreasing"
        assert 0.05 < prev < 0.15
        assert cip[-1] == pytest.approx(prev, abs=0.02)


class TestTrueCipWeibull:
    def test_monotone_increasing(self):
        _ages, cip = compute_true_cip_weibull(scale=2160, rho=0.8, beta=1.0)
        assert cip[0] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(cip) >= -1e-10)

    def test_higher_beta_higher_cip(self):
        """Stronger liability effect → higher CIP at same age."""
        _, cip_low = compute_true_cip_weibull(scale=2160, rho=0.8, beta=0.5)
        _, cip_high = compute_true_cip_weibull(scale=2160, rho=0.8, beta=1.5)
        assert cip_high[-1] > cip_low[-1]


# ---------------------------------------------------------------------------
# Thresholds and w
# ---------------------------------------------------------------------------


class TestThresholdsAndW:
    def test_cases_w_is_one(self):
        affected = np.array([True, False, True])
        t_obs = np.array([30.0, 60.0, 50.0])
        cip_ages = np.array([0.0, 40.0, 80.0])
        cip_values = np.array([0.0, 0.05, 0.10])
        _thr, w = compute_thresholds_and_w(affected, t_obs, cip_ages, cip_values, 0.10)
        assert w[0] == 1.0
        assert w[2] == 1.0
        assert 0.0 < w[1] < 1.0

    def test_threshold_matches_prevalence(self):
        K = 0.10
        thr, _ = compute_thresholds_and_w(
            np.array([False]),
            np.array([80.0]),
            np.array([0.0, 80.0]),
            np.array([0.0, 0.10]),
            K,
        )
        expected = norm.ppf(1 - K)
        assert thr[0] == pytest.approx(expected)

    def test_age_dependent_thresholds(self):
        """Age-dependent: older controls get lower thresholds (more informative)."""
        K = 0.10
        cip_ages = np.array([0.0, 40.0, 80.0])
        cip_values = np.array([0.0, 0.05, 0.10])
        thr, w = compute_thresholds_and_w(
            np.array([False, False]),
            np.array([20.0, 70.0]),
            cip_ages,
            cip_values,
            K,
            age_dependent=True,
        )
        # Older control has higher CIP → lower threshold
        assert thr[1] < thr[0], "Older control should have lower threshold"
        # w should still be valid
        assert 0.0 <= w[0] <= 1.0
        assert 0.0 <= w[1] <= 1.0
        assert w[1] > w[0], "Older control has higher w"

    def test_age_dependent_at_max_age_matches_lifetime(self):
        """At max age, age-dependent threshold ≈ lifetime threshold."""
        K = 0.10
        cip_ages = np.array([0.0, 80.0])
        cip_values = np.array([0.0, K])
        thr_age, _ = compute_thresholds_and_w(
            np.array([False]),
            np.array([80.0]),
            cip_ages,
            cip_values,
            K,
            age_dependent=True,
        )
        thr_lifetime, _ = compute_thresholds_and_w(
            np.array([False]),
            np.array([80.0]),
            cip_ages,
            cip_values,
            K,
            age_dependent=False,
        )
        assert thr_age[0] == pytest.approx(thr_lifetime[0], abs=1e-6)

    def test_age_dependent_cases_get_onset_threshold(self):
        """Cases should get threshold at their onset age, not lifetime."""
        K = 0.10
        cip_ages = np.array([0.0, 40.0, 80.0])
        cip_values = np.array([0.0, 0.05, 0.10])
        thr, _ = compute_thresholds_and_w(
            np.array([True]),
            np.array([40.0]),
            cip_ages,
            cip_values,
            K,
            age_dependent=True,
        )
        expected = norm.ppf(1.0 - 0.05)  # CIP at age 40
        assert thr[0] == pytest.approx(expected, abs=1e-6)

    def test_sex_stratified_thresholds(self):
        """Sex-specific CIP tables produce different thresholds by sex."""
        affected = np.array([False, False, False, False])
        t_obs = np.array([50.0, 50.0, 50.0, 50.0])
        sex = np.array([0, 1, 0, 1])  # 0=female, 1=male

        # Females: higher prevalence → lower threshold
        cip_ages_f = np.array([0.0, 80.0])
        cip_vals_f = np.array([0.0, 0.20])
        # Males: lower prevalence → higher threshold
        cip_ages_m = np.array([0.0, 80.0])
        cip_vals_m = np.array([0.0, 0.05])

        thr, _w = compute_thresholds_and_w_by_sex(
            affected, t_obs, sex,
            cip_ages_f, cip_vals_f, 0.20,
            cip_ages_m, cip_vals_m, 0.05,
        )
        # Female threshold < male threshold (higher prevalence → lower threshold)
        assert thr[0] < thr[1]
        assert thr[0] == pytest.approx(thr[2])  # same sex → same threshold
        assert thr[1] == pytest.approx(thr[3])


# ---------------------------------------------------------------------------
# Numba vs Python agreement tests
# ---------------------------------------------------------------------------


class TestNumbaAgreement:
    """Verify numba-compiled functions match Python fallbacks."""

    @pytest.mark.parametrize(
        ("mu", "var", "trunc"),
        [
            (0.0, 1.0, 1.0),
            (0.0, 1.0, -1.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 1.5),
            (-1.0, 2.0, 0.5),
            (0.0, 1.0, -5.0),
        ],
    )
    def test_trunc_norm_below(self, mu, var, trunc):
        py_m, py_v = _trunc_norm_below_py(mu, var, trunc)
        nb_m, nb_v = _nb_trunc_norm_below(mu, var, trunc)
        assert nb_m == pytest.approx(py_m, abs=1e-12)
        assert nb_v == pytest.approx(py_v, abs=1e-12)

    @pytest.mark.parametrize(
        ("mu", "var", "trunc"),
        [
            (0.0, 1.0, -0.5),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 1.5),
            (-1.0, 2.0, 0.5),
            (0.0, 1.0, 5.0),
        ],
    )
    def test_trunc_norm_above(self, mu, var, trunc):
        py_m, py_v = _trunc_norm_above_py(mu, var, trunc)
        nb_m, nb_v = _nb_trunc_norm_above(mu, var, trunc)
        assert nb_m == pytest.approx(py_m, abs=1e-12)
        assert nb_v == pytest.approx(py_v, abs=1e-12)

    @pytest.mark.parametrize(
        ("mu", "var", "lower", "upper"),
        [
            (0.0, 1.0, -np.inf, 1.0),
            (0.0, 1.0, 0.5, np.inf),
            (0.0, 1.0, -1.0, 1.0),
            (0.0, 1.0, 2.0, 2.0),
        ],
    )
    def test_trunc_norm(self, mu, var, lower, upper):
        py_m, py_v = _trunc_norm_py(mu, var, lower, upper)
        nb_m, nb_v = _nb_trunc_norm(mu, var, lower, upper)
        assert nb_m == pytest.approx(py_m, abs=1e-12)
        assert nb_v == pytest.approx(py_v, abs=1e-12)

    @pytest.mark.parametrize("kp", [0.0, 0.05, 0.1])
    def test_trunc_norm_mixture(self, kp):
        thr = norm.ppf(0.9)
        py_m, py_v = _trunc_norm_mixture_py(0.0, 1.0, -np.inf, thr, kp)
        nb_m, nb_v = _nb_trunc_norm_mixture(0.0, 1.0, -np.inf, thr, kp)
        assert nb_m == pytest.approx(py_m, abs=1e-12)
        assert nb_v == pytest.approx(py_v, abs=1e-12)


class TestPairBasedKinship:
    """Verify pair-based kinship matches DP for known pair types."""

    def test_parent_offspring(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        df = pd.DataFrame(
            {
                "id": ids,
                "mother": mothers,
                "father": fathers,
                "twin": twins,
                "sex": [0, 1, 0, 1, 0, 1],
                "generation": [0, 0, 1, 1, 2, 2],
            }
        )
        kmat = build_kinship_from_pairs(df, ndegree=2)
        assert kmat[0, 2] == pytest.approx(0.25)  # mom → child
        assert kmat[1, 2] == pytest.approx(0.25)  # dad → child

    def test_full_siblings(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        df = pd.DataFrame(
            {
                "id": ids,
                "mother": mothers,
                "father": fathers,
                "twin": twins,
                "sex": [0, 1, 0, 1, 0, 1],
                "generation": [0, 0, 1, 1, 2, 2],
            }
        )
        kmat = build_kinship_from_pairs(df, ndegree=2)
        assert kmat[2, 3] == pytest.approx(0.25)

    def test_self_kinship(self):
        ids, mothers, fathers, twins = _make_simple_pedigree()
        df = pd.DataFrame(
            {
                "id": ids,
                "mother": mothers,
                "father": fathers,
                "twin": twins,
                "sex": [0, 1, 0, 1, 0, 1],
                "generation": [0, 0, 1, 1, 2, 2],
            }
        )
        kmat = build_kinship_from_pairs(df, ndegree=2)
        for i in range(len(ids)):
            assert kmat[i, i] == pytest.approx(0.5)
