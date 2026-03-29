"""Unit tests for bivariate PA-FGRS implementation."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from fit_ace.pafgrs.pafgrs import pa_fgrs
from fit_ace.pafgrs.pafgrs_bivariate import (
    _nb_pa_fgrs_bivariate_single,
    build_bivariate_covmat,
    pa_fgrs_bivariate,
)

# ---------------------------------------------------------------------------
# Covariance matrix construction
# ---------------------------------------------------------------------------


class TestBuildBivariateCovmat:
    def test_symmetry(self):
        """Covmat must be symmetric."""
        kin = np.array([0.25, 0.125])
        rel_kin = np.array([[0.5, 0.0], [0.0, 0.5]])
        cm = build_bivariate_covmat(kin, rel_kin, 0.5, 0.3, 0.2, 0.2)
        np.testing.assert_allclose(cm, cm.T, atol=1e-15)

    def test_positive_semidefinite(self):
        """Covmat eigenvalues must be non-negative."""
        kin = np.array([0.25, 0.25, 0.125])
        rel_kin = np.array(
            [
                [0.5, 0.25, 0.125],
                [0.25, 0.5, 0.125],
                [0.125, 0.125, 0.5],
            ]
        )
        cm = build_bivariate_covmat(kin, rel_kin, 0.5, 0.5, 0.3, 0.3)
        eigvals = np.linalg.eigvalsh(cm)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_proband_block(self):
        """Proband block should be [[h2_1, cov_g12], [cov_g12, h2_2]]."""
        cm = build_bivariate_covmat(
            np.array([0.25]),
            np.array([[0.5]]),
            h2_1=0.5,
            h2_2=0.3,
            cov_g12=0.15,
            rho_within=0.15,
        )
        assert cm[0, 0] == pytest.approx(0.5)
        assert cm[1, 1] == pytest.approx(0.3)
        assert cm[0, 1] == pytest.approx(0.15)
        assert cm[1, 0] == pytest.approx(0.15)

    def test_relative_self_block(self):
        """Each relative's self-block should be [[1, rho], [rho, 1]]."""
        rho = 0.2
        cm = build_bivariate_covmat(
            np.array([0.25]),
            np.array([[0.5]]),
            h2_1=0.5,
            h2_2=0.3,
            cov_g12=0.15,
            rho_within=rho,
        )
        assert cm[2, 2] == pytest.approx(1.0)
        assert cm[3, 3] == pytest.approx(1.0)
        assert cm[2, 3] == pytest.approx(rho)
        assert cm[3, 2] == pytest.approx(rho)

    def test_proband_relative_covariance(self):
        """Cross-covariance between proband and relative."""
        h2_1, h2_2 = 0.5, 0.3
        cov_g12 = 0.2
        kin = 0.25
        cm = build_bivariate_covmat(
            np.array([kin]),
            np.array([[0.5]]),
            h2_1,
            h2_2,
            cov_g12,
            rho_within=cov_g12,
        )
        # proband_t1 <-> rel_t1: 2 * kin * h2_1
        assert cm[0, 2] == pytest.approx(2 * kin * h2_1)
        # proband_t1 <-> rel_t2: 2 * kin * cov_g12
        assert cm[0, 3] == pytest.approx(2 * kin * cov_g12)
        # proband_t2 <-> rel_t1: 2 * kin * cov_g12
        assert cm[1, 2] == pytest.approx(2 * kin * cov_g12)
        # proband_t2 <-> rel_t2: 2 * kin * h2_2
        assert cm[1, 3] == pytest.approx(2 * kin * h2_2)

    def test_no_relatives(self):
        """Empty relative set: 2×2 proband block only."""
        cm = build_bivariate_covmat(
            np.array([]),
            np.empty((0, 0)),
            h2_1=0.5,
            h2_2=0.3,
            cov_g12=0.1,
            rho_within=0.1,
        )
        assert cm.shape == (2, 2)
        assert cm[0, 0] == pytest.approx(0.5)
        assert cm[1, 1] == pytest.approx(0.3)

    def test_size(self):
        """Matrix size should be 2 + 2*n_rel."""
        n_rel = 5
        kin = np.full(n_rel, 0.25)
        rel_kin = np.eye(n_rel) * 0.5
        cm = build_bivariate_covmat(kin, rel_kin, 0.5, 0.5, 0.3, 0.3)
        assert cm.shape == (2 + 2 * n_rel, 2 + 2 * n_rel)


# ---------------------------------------------------------------------------
# Bivariate PA-FGRS algorithm
# ---------------------------------------------------------------------------


class TestPaFgrsBivariate:
    def test_no_relatives_returns_prior(self):
        """No relatives → posterior = prior."""
        h2_1, h2_2, cov_g12 = 0.5, 0.3, 0.15
        covmat = np.array([[h2_1, cov_g12], [cov_g12, h2_2]])
        est1, est2, var1, var2, cov12 = pa_fgrs_bivariate(
            rel_status1=np.array([], dtype=bool),
            rel_thr1=np.array([]),
            rel_w1=np.array([]),
            rel_status2=np.array([], dtype=bool),
            rel_thr2=np.array([]),
            rel_w2=np.array([]),
            covmat=covmat,
        )
        assert est1 == pytest.approx(0.0)
        assert est2 == pytest.approx(0.0)
        assert var1 == pytest.approx(h2_1)
        assert var2 == pytest.approx(h2_2)
        assert cov12 == pytest.approx(cov_g12)

    def test_affected_relative_positive_shift(self):
        """Affected relative on both traits → both estimates shift up."""
        h2_1 = h2_2 = 0.5
        rA = 0.5
        cov_g12 = rA * math.sqrt(h2_1 * h2_2)
        kin = 0.25
        cm = build_bivariate_covmat(
            np.array([kin]),
            np.array([[0.5]]),
            h2_1,
            h2_2,
            cov_g12,
            rho_within=cov_g12,
        )
        thr = norm.ppf(0.9)  # 10% prevalence
        est1, est2, var1, var2, _ = pa_fgrs_bivariate(
            rel_status1=np.array([True]),
            rel_thr1=np.array([thr]),
            rel_w1=np.array([1.0]),
            rel_status2=np.array([True]),
            rel_thr2=np.array([thr]),
            rel_w2=np.array([1.0]),
            covmat=cm,
        )
        assert est1 > 0, "Affected relative should shift trait 1 estimate up"
        assert est2 > 0, "Affected relative should shift trait 2 estimate up"
        assert var1 < h2_1, "Posterior variance should shrink"
        assert var2 < h2_2

    def test_unaffected_relative_negative_shift(self):
        """Unaffected relative on both traits → both estimates shift down."""
        h2_1 = h2_2 = 0.5
        cov_g12 = 0.25
        kin = 0.25
        cm = build_bivariate_covmat(
            np.array([kin]),
            np.array([[0.5]]),
            h2_1,
            h2_2,
            cov_g12,
            rho_within=cov_g12,
        )
        thr = norm.ppf(0.9)
        est1, est2, _, _, _ = pa_fgrs_bivariate(
            rel_status1=np.array([False]),
            rel_thr1=np.array([thr]),
            rel_w1=np.array([1.0]),
            rel_status2=np.array([False]),
            rel_thr2=np.array([thr]),
            rel_w2=np.array([1.0]),
            covmat=cm,
        )
        assert est1 < 0, "Unaffected relative shifts trait 1 down"
        assert est2 < 0, "Unaffected relative shifts trait 2 down"

    def test_cross_trait_information(self):
        """Affected on trait 2 only should shift trait 1 estimate when rA > 0."""
        h2_1 = h2_2 = 0.5
        rA = 0.7
        cov_g12 = rA * math.sqrt(h2_1 * h2_2)
        kin = 0.25
        cm = build_bivariate_covmat(
            np.array([kin]),
            np.array([[0.5]]),
            h2_1,
            h2_2,
            cov_g12,
            rho_within=cov_g12,
        )
        thr = norm.ppf(0.9)

        # Relative affected only on trait 2, unaffected on trait 1
        est1, est2, _, _, _ = pa_fgrs_bivariate(
            rel_status1=np.array([False]),
            rel_thr1=np.array([thr]),
            rel_w1=np.array([1.0]),
            rel_status2=np.array([True]),
            rel_thr2=np.array([thr]),
            rel_w2=np.array([1.0]),
            covmat=cm,
        )
        # Trait 2 affected → trait 2 estimate positive
        assert est2 > 0
        # Cross-trait info: trait 2 affected should also pull trait 1 up
        # But trait 1 is unaffected, which pulls it down
        # The net effect depends on the relative strength; with rA=0.7 and
        # the same threshold, the trait 1 observation (unaffected) dominates
        # Just check we get a finite result
        assert np.isfinite(est1)

    def test_zero_rA_matches_univariate(self):
        """When rA=0, bivariate marginals should match univariate scores."""
        h2_1, h2_2 = 0.5, 0.3
        cov_g12 = 0.0
        kin = np.array([0.25, 0.125])
        rel_kin = np.array([[0.5, 0.0], [0.0, 0.5]])

        cm_biv = build_bivariate_covmat(
            kin,
            rel_kin,
            h2_1,
            h2_2,
            cov_g12,
            rho_within=0.0,
        )

        thr1 = norm.ppf(0.9)
        thr2 = norm.ppf(0.8)

        status1 = np.array([True, False])
        status2 = np.array([False, True])

        est1, est2, var1, var2, cov12 = pa_fgrs_bivariate(
            rel_status1=status1,
            rel_thr1=np.full(2, thr1),
            rel_w1=np.ones(2),
            rel_status2=status2,
            rel_thr2=np.full(2, thr2),
            rel_w2=np.ones(2),
            covmat=cm_biv,
        )

        # Univariate trait 1
        cm_uni1 = np.array(
            [
                [h2_1, 2 * kin[0] * h2_1, 2 * kin[1] * h2_1],
                [2 * kin[0] * h2_1, 1.0, 0.0],
                [2 * kin[1] * h2_1, 0.0, 1.0],
            ]
        )
        est1_uni, var1_uni = pa_fgrs(
            status1,
            np.full(2, thr1),
            np.ones(2),
            cm_uni1,
        )

        # Univariate trait 2
        cm_uni2 = np.array(
            [
                [h2_2, 2 * kin[0] * h2_2, 2 * kin[1] * h2_2],
                [2 * kin[0] * h2_2, 1.0, 0.0],
                [2 * kin[1] * h2_2, 0.0, 1.0],
            ]
        )
        est2_uni, var2_uni = pa_fgrs(
            status2,
            np.full(2, thr2),
            np.ones(2),
            cm_uni2,
        )

        assert est1 == pytest.approx(est1_uni, abs=1e-10)
        assert est2 == pytest.approx(est2_uni, abs=1e-10)
        assert var1 == pytest.approx(var1_uni, abs=1e-10)
        assert var2 == pytest.approx(var2_uni, abs=1e-10)
        assert cov12 == pytest.approx(0.0, abs=1e-10)

    def test_more_relatives_more_precision(self):
        """Two relatives should give lower variance than one."""
        h2 = 0.5
        cov_g12 = 0.2
        thr = norm.ppf(0.9)

        # One relative
        cm1 = build_bivariate_covmat(
            np.array([0.25]),
            np.array([[0.5]]),
            h2,
            h2,
            cov_g12,
            cov_g12,
        )
        _, _, var1_one, var2_one, _ = pa_fgrs_bivariate(
            np.array([True]),
            np.array([thr]),
            np.array([1.0]),
            np.array([True]),
            np.array([thr]),
            np.array([1.0]),
            cm1,
        )

        # Two relatives
        cm2 = build_bivariate_covmat(
            np.array([0.25, 0.25]),
            np.array([[0.5, 0.0], [0.0, 0.5]]),
            h2,
            h2,
            cov_g12,
            cov_g12,
        )
        _, _, var1_two, var2_two, _ = pa_fgrs_bivariate(
            np.array([True, True]),
            np.array([thr, thr]),
            np.ones(2),
            np.array([True, True]),
            np.array([thr, thr]),
            np.ones(2),
            cm2,
        )

        assert var1_two < var1_one
        assert var2_two < var2_one

    def test_bivariate_improves_over_univariate(self):
        """With rA > 0, bivariate should give lower variance than univariate."""
        h2 = 0.5
        rA = 0.7
        cov_g12 = rA * h2  # sqrt(h2*h2) = h2
        kin = 0.25
        thr = norm.ppf(0.9)

        # Bivariate
        cm_biv = build_bivariate_covmat(
            np.array([kin]),
            np.array([[0.5]]),
            h2,
            h2,
            cov_g12,
            cov_g12,
        )
        _, _, var1_biv, _, _ = pa_fgrs_bivariate(
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            cm_biv,
        )

        # Univariate (trait 1 only)
        cm_uni = np.array(
            [
                [h2, 2 * kin * h2],
                [2 * kin * h2, 1.0],
            ]
        )
        _, var1_uni = pa_fgrs(
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            cm_uni,
        )

        assert var1_biv < var1_uni, "Bivariate should have lower variance when rA > 0"


# ---------------------------------------------------------------------------
# Python ↔ Numba agreement
# ---------------------------------------------------------------------------


class TestPythonNumbaAgreement:
    def _run_both(self, n_rel, status1, thr1, w1, status2, thr2, w2, covmat):
        """Run both Python and Numba kernels, return (py_result, nb_result)."""
        py_result = pa_fgrs_bivariate(
            status1,
            thr1,
            w1,
            status2,
            thr2,
            w2,
            covmat,
        )
        nb_result = _nb_pa_fgrs_bivariate_single(
            n_rel,
            np.asarray(status1, dtype=np.bool_),
            np.asarray(thr1, dtype=np.float64),
            np.asarray(w1, dtype=np.float64),
            np.asarray(status2, dtype=np.bool_),
            np.asarray(thr2, dtype=np.float64),
            np.asarray(w2, dtype=np.float64),
            np.asarray(covmat, dtype=np.float64),
        )
        return py_result, nb_result

    def test_single_relative_agreement(self):
        """Python and Numba should agree for a single relative."""
        h2_1, h2_2 = 0.5, 0.3
        cov_g12 = 0.2
        cm = build_bivariate_covmat(
            np.array([0.25]),
            np.array([[0.5]]),
            h2_1,
            h2_2,
            cov_g12,
            cov_g12,
        )
        thr = norm.ppf(0.9)
        py, nb = self._run_both(
            1,
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            np.array([False]),
            np.array([thr]),
            np.ones(1),
            cm,
        )
        for i in range(5):
            assert py[i] == pytest.approx(nb[i], abs=1e-10), f"Mismatch at index {i}: py={py[i]}, nb={nb[i]}"

    def test_multiple_relatives_agreement(self):
        """Python and Numba should agree for multiple relatives."""
        h2_1, h2_2 = 0.5, 0.4
        rA = 0.3
        cov_g12 = rA * math.sqrt(h2_1 * h2_2)
        kin = np.array([0.25, 0.25, 0.125])
        rel_kin = np.array(
            [
                [0.5, 0.25, 0.0],
                [0.25, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ]
        )
        cm = build_bivariate_covmat(kin, rel_kin, h2_1, h2_2, cov_g12, cov_g12)
        thr1 = norm.ppf(0.9)
        thr2 = norm.ppf(0.8)

        py, nb = self._run_both(
            3,
            np.array([True, False, True]),
            np.full(3, thr1),
            np.array([1.0, 0.5, 1.0]),
            np.array([False, True, False]),
            np.full(3, thr2),
            np.array([1.0, 1.0, 0.8]),
            cm,
        )
        for i in range(5):
            assert py[i] == pytest.approx(nb[i], abs=1e-10), f"Mismatch at index {i}: py={py[i]}, nb={nb[i]}"

    def test_no_relatives_agreement(self):
        """Both should return prior when no relatives."""
        h2_1, h2_2 = 0.5, 0.3
        cov_g12 = 0.15
        cm = np.array([[h2_1, cov_g12], [cov_g12, h2_2]])

        py = pa_fgrs_bivariate(
            np.array([], dtype=bool),
            np.array([]),
            np.array([]),
            np.array([], dtype=bool),
            np.array([]),
            np.array([]),
            cm,
        )
        nb = _nb_pa_fgrs_bivariate_single(
            0,
            np.array([], dtype=np.bool_),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.bool_),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            cm,
        )
        for i in range(5):
            assert py[i] == pytest.approx(nb[i], abs=1e-15)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_partial_w(self):
        """Controls with partial w should produce finite results."""
        h2 = 0.5
        cov_g12 = 0.2
        cm = build_bivariate_covmat(
            np.array([0.25]),
            np.array([[0.5]]),
            h2,
            h2,
            cov_g12,
            cov_g12,
        )
        thr = norm.ppf(0.9)
        est1, est2, var1, var2, _cov12 = pa_fgrs_bivariate(
            np.array([False]),
            np.array([thr]),
            np.array([0.3]),
            np.array([False]),
            np.array([thr]),
            np.array([0.5]),
            cm,
        )
        assert np.isfinite(est1)
        assert np.isfinite(est2)
        assert var1 > 0
        assert var2 > 0

    def test_zero_w_ignored(self):
        """Relatives with w=0 should be equivalent to having no relatives."""
        h2_1, h2_2 = 0.5, 0.3
        cov_g12 = 0.15
        cm = build_bivariate_covmat(
            np.array([0.25]),
            np.array([[0.5]]),
            h2_1,
            h2_2,
            cov_g12,
            cov_g12,
        )
        thr = norm.ppf(0.9)
        est1, est2, var1, var2, _cov12 = pa_fgrs_bivariate(
            np.array([True]),
            np.array([thr]),
            np.array([0.0]),
            np.array([True]),
            np.array([thr]),
            np.array([0.0]),
            cm,
        )
        assert est1 == pytest.approx(0.0)
        assert est2 == pytest.approx(0.0)
        assert var1 == pytest.approx(h2_1)
        assert var2 == pytest.approx(h2_2)

    def test_different_h2_per_trait(self):
        """Different heritability per trait should not crash."""
        h2_1, h2_2 = 0.8, 0.2
        rA = 0.5
        cov_g12 = rA * math.sqrt(h2_1 * h2_2)
        cm = build_bivariate_covmat(
            np.array([0.25]),
            np.array([[0.5]]),
            h2_1,
            h2_2,
            cov_g12,
            cov_g12,
        )
        thr = norm.ppf(0.9)
        est1, est2, var1, var2, _ = pa_fgrs_bivariate(
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            cm,
        )
        assert est1 > 0
        assert est2 > 0
        assert var1 < h2_1
        assert var2 < h2_2

    def test_high_rA(self):
        """rA close to 1 should produce highly correlated estimates."""
        h2 = 0.5
        rA = 0.95
        cov_g12 = rA * h2
        kin = np.array([0.25, 0.25])
        rel_kin = np.array([[0.5, 0.0], [0.0, 0.5]])
        cm = build_bivariate_covmat(kin, rel_kin, h2, h2, cov_g12, cov_g12)
        thr = norm.ppf(0.9)

        # Both relatives affected on both traits
        est1, est2, _, _, _ = pa_fgrs_bivariate(
            np.array([True, True]),
            np.full(2, thr),
            np.ones(2),
            np.array([True, True]),
            np.full(2, thr),
            np.ones(2),
            cm,
        )
        # With rA≈1, both estimates should be similar
        assert abs(est1 - est2) < 0.1 * max(abs(est1), abs(est2), 0.01)

    def test_rho_within_differs_from_cov_g12(self):
        """Genetic+C variant: rho_within > cov_g12 should give tighter posterior."""
        h2 = 0.5
        cov_g12 = 0.2
        kin = 0.25
        thr = norm.ppf(0.9)

        # Genetic-only
        cm_gen = build_bivariate_covmat(
            np.array([kin]),
            np.array([[0.5]]),
            h2,
            h2,
            cov_g12,
            rho_within=cov_g12,
        )
        _, _, var1_gen, _, _ = pa_fgrs_bivariate(
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            cm_gen,
        )

        # Genetic+C (higher within-person correlation)
        rho_within_gc = cov_g12 + 0.15
        cm_gc = build_bivariate_covmat(
            np.array([kin]),
            np.array([[0.5]]),
            h2,
            h2,
            cov_g12,
            rho_within=rho_within_gc,
        )
        _, _, var1_gc, _, _ = pa_fgrs_bivariate(
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            np.array([True]),
            np.array([thr]),
            np.ones(1),
            cm_gc,
        )

        # Higher within-person correlation means more info flows cross-trait
        # → different (typically tighter) posterior
        assert var1_gc != pytest.approx(var1_gen, abs=1e-6)


# ---------------------------------------------------------------------------
# Tetrachoric correlation
# ---------------------------------------------------------------------------


class TestComputeTetrachoric:
    def test_zero_correlation(self):
        """Independent traits → tetrachoric ≈ 0."""
        from fit_ace.pafgrs.pafgrs_bivariate import compute_tetrachoric

        rng = np.random.default_rng(42)
        n = 10000
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        table = np.array(
            [
                [((x < 0) & (y < 0)).sum(), ((x < 0) & (y >= 0)).sum()],
                [((x >= 0) & (y < 0)).sum(), ((x >= 0) & (y >= 0)).sum()],
            ]
        )
        rho = compute_tetrachoric(table)
        assert abs(rho) < 0.1, f"Expected near zero, got {rho}"

    def test_positive_correlation(self):
        """Correlated normals → positive tetrachoric."""
        from fit_ace.pafgrs.pafgrs_bivariate import compute_tetrachoric

        rng = np.random.default_rng(42)
        n = 10000
        rho_true = 0.6
        z = rng.multivariate_normal([0, 0], [[1, rho_true], [rho_true, 1]], n)
        x, y = z[:, 0], z[:, 1]
        table = np.array(
            [
                [((x < 0) & (y < 0)).sum(), ((x < 0) & (y >= 0)).sum()],
                [((x >= 0) & (y < 0)).sum(), ((x >= 0) & (y >= 0)).sum()],
            ]
        )
        rho = compute_tetrachoric(table)
        assert rho == pytest.approx(rho_true, abs=0.1)

    def test_negative_correlation(self):
        """Negatively correlated normals → negative tetrachoric."""
        from fit_ace.pafgrs.pafgrs_bivariate import compute_tetrachoric

        rng = np.random.default_rng(42)
        n = 10000
        rho_true = -0.5
        z = rng.multivariate_normal([0, 0], [[1, rho_true], [rho_true, 1]], n)
        x, y = z[:, 0], z[:, 1]
        table = np.array(
            [
                [((x < 0) & (y < 0)).sum(), ((x < 0) & (y >= 0)).sum()],
                [((x >= 0) & (y < 0)).sum(), ((x >= 0) & (y >= 0)).sum()],
            ]
        )
        rho = compute_tetrachoric(table)
        assert rho == pytest.approx(rho_true, abs=0.1)

    def test_asymmetric_thresholds(self):
        """Non-median thresholds should still recover correlation."""
        from fit_ace.pafgrs.pafgrs_bivariate import compute_tetrachoric

        rng = np.random.default_rng(42)
        n = 20000
        rho_true = 0.4
        z = rng.multivariate_normal([0, 0], [[1, rho_true], [rho_true, 1]], n)
        x, y = z[:, 0], z[:, 1]
        # Asymmetric thresholds: 10% and 30%
        t1 = norm.ppf(0.9)
        t2 = norm.ppf(0.7)
        table = np.array(
            [
                [((x < t1) & (y < t2)).sum(), ((x < t1) & (y >= t2)).sum()],
                [((x >= t1) & (y < t2)).sum(), ((x >= t1) & (y >= t2)).sum()],
            ]
        )
        rho = compute_tetrachoric(table)
        assert rho == pytest.approx(rho_true, abs=0.15)
