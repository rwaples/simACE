"""Tests for mate correlation computation and expected value formula."""

import numpy as np
import pandas as pd

from simace.analysis.stats import compute_mate_correlation
from simace.simulation.mate_correlation import expected_mate_corr_matrix


class TestExpectedMateCorr:
    def test_both_zero_returns_zeros(self):
        """When assort1=assort2=0, expected matrix should be all zeros."""
        result = expected_mate_corr_matrix(0, 0, rA=0.5, rC=0.3, A1=0.5, C1=0.2, A2=0.4, C2=0.1)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))

    def test_single_trait_diagonal(self):
        """When assort1=0.3, assort2=0, diagonal should be (assort1, rho_w*assort1)."""
        a1, a2 = 0.3, 0.0
        # With only assort1, r_eff = |assort1| = 0.3
        # wf = [0.3, 0], wm = [0.3, 0]
        # Sigma @ wf = [0.3, 0.3*rho_w], norm = sqrt(0.3^2) = 0.3
        # lambda_f = [1, rho_w], lambda_m = [1, rho_w]
        # E[r] = 0.3 * outer([1, rho_w], [1, rho_w])
        # So E[r11] = 0.3, E[r12] = E[r21] = 0.3*rho_w, E[r22] = 0.3*rho_w^2
        result = expected_mate_corr_matrix(a1, a2, rA=0.5, rC=0.0, A1=0.5, C1=0.0, A2=0.5, C2=0.0)
        # rho_w = 0.5 * sqrt(0.5*0.5) + 0 = 0.25
        rho_w = 0.25
        assert abs(result[0, 0] - 0.3) < 1e-10
        assert abs(result[0, 1] - 0.3 * rho_w) < 1e-10
        assert abs(result[1, 0] - 0.3 * rho_w) < 1e-10
        assert abs(result[1, 1] - 0.3 * rho_w**2) < 1e-10

    def test_shape(self):
        """Result should be a 2x2 numpy array."""
        result = expected_mate_corr_matrix(0.3, 0.5, rA=0.5, rC=0.3, A1=0.5, C1=0.2, A2=0.4, C2=0.1)
        assert result.shape == (2, 2)

    def test_symmetric_assort_produces_symmetric(self):
        """With equal traits and equal assort, diagonal cells should be equal."""
        result = expected_mate_corr_matrix(0.3, 0.3, rA=1.0, rC=1.0, A1=0.5, C1=0.2, A2=0.5, C2=0.2)
        assert abs(result[0, 0] - result[1, 1]) < 1e-10

    def test_both_nonzero_diagonal_equals_targets(self):
        """With both traits, diagonal should equal target correlations."""
        result = expected_mate_corr_matrix(0.4, 0.3, rA=0.5, rC=0.3, A1=0.5, C1=0.2, A2=0.4, C2=0.1)
        assert abs(result[0, 0] - 0.4) < 1e-10
        assert abs(result[1, 1] - 0.3) < 1e-10

    def test_both_nonzero_offdiag_computed(self):
        """Off-diagonal should be computed from rho_w * sqrt(|r1*r2|)."""
        result = expected_mate_corr_matrix(0.4, 0.3, rA=0.5, rC=0.3, A1=0.5, C1=0.2, A2=0.4, C2=0.1)
        # rho_w = 0.5*sqrt(0.5*0.4) + 0.3*sqrt(0.2*0.1)
        rho_w = 0.5 * np.sqrt(0.2) + 0.3 * np.sqrt(0.02)
        c = rho_w * np.sqrt(abs(0.4 * 0.3)) * np.sign(0.4 * 0.3)
        assert not np.isnan(result[0, 1])
        assert abs(result[0, 1] - c) < 1e-10
        assert abs(result[1, 0] - c) < 1e-10
        # Symmetric
        assert abs(result[0, 1] - result[1, 0]) < 1e-10

    def test_high_targets_within_bounds(self):
        """Diagonal of high assort values should still be valid."""
        result = expected_mate_corr_matrix(0.9, 0.9, rA=0.5, rC=0.3, A1=0.5, C1=0.2, A2=0.4, C2=0.1)
        # Diagonal should equal targets
        assert abs(result[0, 0] - 0.9) < 1e-10
        assert abs(result[1, 1] - 0.9) < 1e-10
        # Off-diagonal should be computed, not NaN
        assert not np.isnan(result[0, 1])
        assert not np.isnan(result[1, 0])
        # Symmetric
        assert abs(result[0, 1] - result[1, 0]) < 1e-10

    def test_explicit_assort_matrix(self):
        """When assort_matrix is provided, it should be returned directly."""
        am = np.array([[0.4, 0.1], [0.1, 0.3]])
        result = expected_mate_corr_matrix(0, 0, rA=0, rC=0, A1=0, C1=0, A2=0, C2=0, assort_matrix=am)
        np.testing.assert_array_almost_equal(result, am)


class TestComputeMateCorrelation:
    def _make_pedigree(self, n_pairs=500, r=0.5, seed=42):
        """Create a simple pedigree with correlated mating."""
        rng = np.random.default_rng(seed)
        # Founders: mothers (even ids), fathers (odd ids)
        n_founders = 2 * n_pairs
        founders = pd.DataFrame(
            {
                "id": range(n_founders),
                "mother": -1,
                "father": -1,
                "liability1": rng.normal(size=n_founders),
                "liability2": rng.normal(size=n_founders),
            }
        )

        # Create correlated mating: sort mothers and fathers by liability1
        mother_ids = list(range(0, n_founders, 2))
        father_ids = list(range(1, n_founders, 2))

        # Sort by liability to induce positive correlation
        mother_liabs = founders.set_index("id").loc[mother_ids, "liability1"].values
        father_liabs = founders.set_index("id").loc[father_ids, "liability1"].values
        m_order = np.argsort(mother_liabs)
        f_order = np.argsort(father_liabs)
        sorted_mothers = np.array(mother_ids)[m_order]
        sorted_fathers = np.array(father_ids)[f_order]

        # Add some noise to pairing (not perfect sort)
        children = []
        child_id = n_founders
        for i in range(n_pairs):
            children.append(
                {
                    "id": child_id,
                    "mother": int(sorted_mothers[i]),
                    "father": int(sorted_fathers[i]),
                    "liability1": rng.normal(),
                    "liability2": rng.normal(),
                }
            )
            child_id += 1

        return pd.concat([founders, pd.DataFrame(children)], ignore_index=True)

    def test_returns_matrix_and_count(self):
        """Should return dict with 'matrix' and 'n_pairs' keys."""
        df = self._make_pedigree()
        result = compute_mate_correlation(df)
        assert "matrix" in result
        assert "n_pairs" in result
        assert result["n_pairs"] > 0
        assert len(result["matrix"]) == 2
        assert len(result["matrix"][0]) == 2

    def test_positive_assortment_detected(self):
        """Sorted mating should produce positive trait-1 diagonal correlation."""
        df = self._make_pedigree(n_pairs=2000)
        result = compute_mate_correlation(df)
        # Trait 1 self-correlation should be positive (sorted mating)
        assert result["matrix"][0][0] > 0.1

    def test_no_assortment_near_zero(self):
        """Random mating should produce near-zero correlations."""
        rng = np.random.default_rng(99)
        n = 1000
        founders = pd.DataFrame(
            {
                "id": range(2 * n),
                "mother": -1,
                "father": -1,
                "liability1": rng.normal(size=2 * n),
                "liability2": rng.normal(size=2 * n),
            }
        )
        children = pd.DataFrame(
            {
                "id": range(2 * n, 3 * n),
                "mother": rng.choice(range(0, 2 * n, 2), size=n),
                "father": rng.choice(range(1, 2 * n, 2), size=n),
                "liability1": rng.normal(size=n),
                "liability2": rng.normal(size=n),
            }
        )
        df = pd.concat([founders, children], ignore_index=True)
        result = compute_mate_correlation(df)
        for i in range(2):
            for j in range(2):
                assert abs(result["matrix"][i][j]) < 0.15

    def test_empty_pedigree(self):
        """All-founder pedigree should return nan matrix."""
        df = pd.DataFrame(
            {
                "id": [0, 1],
                "mother": [-1, -1],
                "father": [-1, -1],
                "liability1": [0.0, 0.0],
                "liability2": [0.0, 0.0],
            }
        )
        result = compute_mate_correlation(df)
        assert result["n_pairs"] == 0


class TestPlotMateCorrelation:
    def test_runs_without_error(self, tmp_path):
        """plot_mate_correlation should produce an output file."""
        from simace.plotting.plot_liability import plot_mate_correlation

        stats = [
            {"mate_correlation": {"matrix": [[0.3, 0.1], [0.1, 0.2]], "n_pairs": 100}},
            {"mate_correlation": {"matrix": [[0.25, 0.08], [0.12, 0.18]], "n_pairs": 100}},
        ]
        params = {"assort1": 0.3, "assort2": 0.0, "rA": 0.5, "rC": 0.0, "A1": 0.5, "C1": 0.0, "A2": 0.5, "C2": 0.0}
        out = tmp_path / "mate_correlation.png"
        plot_mate_correlation(stats, str(out), scenario="test", params=params)
        assert out.exists()

    def test_both_nonzero_no_crash(self, tmp_path):
        """plot_mate_correlation should handle NaN expected values without error."""
        from simace.plotting.plot_liability import plot_mate_correlation

        stats = [
            {"mate_correlation": {"matrix": [[0.3, 0.2], [0.15, 0.4]], "n_pairs": 100}},
        ]
        params = {"assort1": 0.3, "assort2": 0.5, "rA": 0.5, "rC": 0.3, "A1": 0.5, "C1": 0.2, "A2": 0.5, "C2": 0.2}
        out = tmp_path / "mate_correlation_both.png"
        plot_mate_correlation(stats, str(out), scenario="test", params=params)
        assert out.exists()

    def test_placeholder_without_data(self, tmp_path):
        """Should produce placeholder when no mate_correlation in stats."""
        from simace.plotting.plot_liability import plot_mate_correlation

        out = tmp_path / "mate_correlation.png"
        plot_mate_correlation([{}], str(out), scenario="test")
        assert out.exists()
