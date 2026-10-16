"""Tests for mate correlation computation and expected value formula."""

import numpy as np
import polars as pl
import pytest

from simace.analysis.mate_correlation import mate_correlation_by_generation, mate_correlation_matrix, mate_pedigree
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

    def test_single_trait2_diagonal(self):
        """When only assort2 is nonzero, trait-2 assortment propagates through rho_w."""
        result = expected_mate_corr_matrix(0.0, 0.4, rA=0.5, rC=0.0, A1=0.5, C1=0.0, A2=0.5, C2=0.0)
        rho_w = 0.25
        assert abs(result[0, 0] - 0.4 * rho_w**2) < 1e-10
        assert abs(result[0, 1] - 0.4 * rho_w) < 1e-10
        assert abs(result[1, 0] - 0.4 * rho_w) < 1e-10
        assert abs(result[1, 1] - 0.4) < 1e-10

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
        founders = pl.DataFrame(
            {
                "id": np.arange(n_founders),
                "mother": np.full(n_founders, -1),
                "father": np.full(n_founders, -1),
                "liability1": rng.normal(size=n_founders),
                "liability2": rng.normal(size=n_founders),
            }
        )

        # Create correlated mating: sort mothers and fathers by liability1
        mother_ids = list(range(0, n_founders, 2))
        father_ids = list(range(1, n_founders, 2))

        # Sort by liability to induce positive correlation. Founder ids equal
        # their row positions by construction, so id lookup is positional.
        liability1 = founders["liability1"].to_numpy()
        mother_liabs = liability1[mother_ids]
        father_liabs = liability1[father_ids]
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

        return pl.concat([founders, pl.DataFrame(children)], how="vertical")

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
        founders = pl.DataFrame(
            {
                "id": np.arange(2 * n),
                "mother": np.full(2 * n, -1),
                "father": np.full(2 * n, -1),
                "liability1": rng.normal(size=2 * n),
                "liability2": rng.normal(size=2 * n),
            }
        )
        children = pl.DataFrame(
            {
                "id": np.arange(2 * n, 3 * n),
                "mother": rng.choice(range(0, 2 * n, 2), size=n),
                "father": rng.choice(range(1, 2 * n, 2), size=n),
                "liability1": rng.normal(size=n),
                "liability2": rng.normal(size=n),
            }
        )
        df = pl.concat([founders, children], how="vertical")
        result = compute_mate_correlation(df)
        for i in range(2):
            for j in range(2):
                assert abs(result["matrix"][i][j]) < 0.15

    def test_empty_pedigree(self):
        """All-founder pedigree should return nan matrix."""
        df = pl.DataFrame(
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

    def test_dangling_parent_links_are_ignored(self):
        """Ascertainment dropout can sever parent links; skip incomplete mating pairs.

        Row 4 carries a mother but no father and is skipped; row 5 is a complete
        mating and is counted.  One pair is below the two needed for a
        correlation, so the matrix is NaN — but ``n_pairs`` reports the real
        count of valid matings rather than a hard-coded zero.
        """
        df = pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "mother": [-1, -1, -1, -1, 0, 2],
                "father": [-1, -1, -1, -1, -1, 3],
                "liability1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "liability2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            }
        )
        result = compute_mate_correlation(df)
        assert result["n_pairs"] == 1
        assert all(np.isnan(value) for row in result["matrix"] for value in row)


class TestPlotMateCorrelation:
    def test_runs_without_error(self, tmp_path):
        """plot_mate_correlation should produce an output file."""
        from simace.plotting.plot_liability import plot_mate_correlation

        stats = [
            {"mate_correlation": {"matrix": [[0.3, 0.1], [0.1, 0.2]], "n_pairs": 100}},
            {"mate_correlation": {"matrix": [[0.25, 0.08], [0.12, 0.18]], "n_pairs": 100}},
        ]
        params = {"G_ped": 3, "assort1": 0.3, "assort2": 0.0, "rA": 0.5, "rC": 0.0, "A1": 0.5, "C1": 0.0, "A2": 0.5}
        params |= {"C2": 0.0}
        out = tmp_path / "mate_correlation.png"
        plot_mate_correlation(stats, str(out), scenario="test", params=params)
        assert out.exists()

    def test_both_nonzero_no_crash(self, tmp_path):
        """plot_mate_correlation should handle NaN expected values without error."""
        from simace.plotting.plot_liability import plot_mate_correlation

        stats = [
            {"mate_correlation": {"matrix": [[0.3, 0.2], [0.15, 0.4]], "n_pairs": 100}},
        ]
        params = {"G_ped": 3, "assort1": 0.3, "assort2": 0.5, "rA": 0.5, "rC": 0.3, "A1": 0.5, "C1": 0.2, "A2": 0.5}
        params |= {"C2": 0.2}
        out = tmp_path / "mate_correlation_both.png"
        plot_mate_correlation(stats, str(out), scenario="test", params=params)
        assert out.exists()

    def test_expected_panel_spans_a_target_that_varies_by_generation(self, tmp_path, monkeypatch):
        from simace.plotting import plot_liability

        heatmaps = []
        real = plot_liability.sns.heatmap
        monkeypatch.setattr(
            plot_liability.sns, "heatmap", lambda data, **kw: heatmaps.append((data, kw)) or real(data, **kw)
        )
        stats = [{"mate_correlation": {"matrix": [[0.3, 0.0], [0.0, 0.0]], "n_pairs": 100}}]
        params = {"G_sim": 4, "G_ped": 4, "assort1": {0: 0.2, 2: 0.4}, "assort2": 0.0, "rA": 0.0, "A1": 0.5, "A2": 0.5}
        params |= {"C1": 0.0, "C2": 0.0, "E1": 0.5, "E2": 0.5}
        out = tmp_path / "mate_correlation_varying.png"
        plot_liability.plot_mate_correlation(stats, str(out), scenario="test", params=params)
        expected, kw = heatmaps[0]
        assert expected[0, 0] == pytest.approx(0.3)
        assert kw["annot"][0, 0] == "0.20\u20130.40"
        assert kw["annot"][1, 1] == "0.00"
        assert "per-generation" in heatmaps[1][1]["ax"].get_title()
        assert out.exists()

    def test_observed_panel_flags_cross_cells_under_a_scale_change(self, tmp_path, monkeypatch):
        """A constant cross target does not make the pooled cross Pearson comparable when E changes scale."""
        from simace.plotting import plot_liability

        axes = []
        real = plot_liability.sns.heatmap
        monkeypatch.setattr(plot_liability.sns, "heatmap", lambda data, **kw: axes.append(kw["ax"]) or real(data, **kw))
        stats = [{"mate_correlation": {"matrix": [[0.5, 0.13], [0.13, 0.5]], "n_pairs": 100}}]
        params = {"G_ped": 4, "assort1": 0.5, "assort2": 0.5, "rA": 0.8, "A1": 0.5, "A2": 0.5, "C1": 0.0, "C2": 0.0}
        params |= {"rE": 0.0, "E1": {0: 0.1, 2: 5.0}, "E2": {0: 5.0, 2: 0.1}}
        plot_liability.plot_mate_correlation(stats, str(tmp_path / "mc.png"), scenario="test", params=params)
        assert "varies" not in axes[0].get_title()
        assert "per-generation" in axes[1].get_title()

    def test_placeholder_without_data(self, tmp_path):
        """Should produce placeholder when no mate_correlation in stats."""
        from simace.plotting.plot_liability import plot_mate_correlation

        out = tmp_path / "mate_correlation.png"
        plot_mate_correlation([{}], str(out), scenario="test")
        assert out.exists()


def _oracle(df: pl.DataFrame, columns: list[str]) -> tuple[np.ndarray, int]:
    """Pearson over distinct (mother, father) pairs whose parents are both rows.

    Each cell uses the pairs whose two values are finite.
    """
    ids = df["id"].to_numpy()
    row = {int(i): r for r, i in enumerate(ids)}
    parents = zip(df["mother"].to_numpy(), df["father"].to_numpy(), strict=True)
    pairs = sorted({(int(m), int(f)) for m, f in parents if int(m) in row and int(f) in row})
    m_rows = np.array([row[m] for m, _ in pairs])
    f_rows = np.array([row[f] for _, f in pairs])
    values = [df[c].to_numpy() for c in columns]

    def cell(vi: np.ndarray, vj: np.ndarray) -> float:
        m, f = vi[m_rows], vj[f_rows]
        keep = np.isfinite(m) & np.isfinite(f)
        return np.corrcoef(m[keep], f[keep])[0, 1]

    return np.array([[cell(vi, vj) for vj in values] for vi in values]), len(pairs)


class TestMateCorrelationMatrix:
    """pg-phenotype's mate correlation against a NumPy oracle."""

    @staticmethod
    def _pedigree(seed: int = 5, n: int = 400) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        founders = 2 * n
        mothers = rng.choice(np.arange(0, founders, 2), size=n)
        fathers = rng.choice(np.arange(1, founders, 2), size=n)
        return pl.DataFrame(
            {
                "id": np.arange(founders + n),
                "mother": np.concatenate([np.full(founders, -1), mothers]),
                "father": np.concatenate([np.full(founders, -1), fathers]),
                "liability1": rng.normal(size=founders + n),
                "liability2": rng.normal(size=founders + n),
            }
        )

    def _check(self, df: pl.DataFrame) -> None:
        matrix, n_pairs = mate_correlation_matrix(mate_pedigree(df), df, ["liability1", "liability2"])
        expected, n_expected = _oracle(df, ["liability1", "liability2"])
        assert n_pairs == n_expected
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-12)

    def test_matches_oracle(self):
        self._check(self._pedigree())

    def test_matches_oracle_with_dangling_parent_ids(self):
        """Dropped rows leave parent ids that name no row; their children make no pair."""
        df = self._pedigree()
        keep = np.random.default_rng(1).random(len(df)) > 0.3
        self._check(df.filter(pl.Series(keep)))

    def test_matches_oracle_with_severed_parents(self):
        df = self._pedigree()
        sever = np.random.default_rng(2).random(len(df)) < 0.2
        self._check(df.with_columns(pl.when(pl.Series(sever)).then(-1).otherwise(pl.col("father")).alias("father")))

    def test_constant_column_is_nan_and_spares_the_other(self):
        df = self._pedigree().with_columns(pl.lit(0.0).alias("liability2"))
        matrix, n_pairs = mate_correlation_matrix(mate_pedigree(df), df, ["liability1", "liability2"])
        expected, n_expected = _oracle(df, ["liability1"])
        assert n_pairs == n_expected
        assert matrix[0, 0] == pytest.approx(expected[0, 0], abs=1e-12)
        assert np.isnan(matrix[[0, 1, 1], [1, 0, 1]]).all()

    def test_every_column_constant_still_counts_pairs(self):
        df = self._pedigree().with_columns(pl.lit(0.0).alias("liability1"))
        matrix, n_pairs = mate_correlation_matrix(mate_pedigree(df), df, ["liability1"])
        assert n_pairs == _oracle(df, ["liability2"])[1]
        assert np.isnan(matrix).all()

    def test_infinite_values_are_missing(self):
        """A pair with an infinite value leaves the cells that read it, as NaN would."""
        df = self._pedigree()
        mother = int(df["mother"].to_numpy()[-1])
        father = int(df["father"].to_numpy()[-2])
        df = df.with_columns(
            pl.when(pl.col("id") == mother)
            .then(float("inf"))
            .when(pl.col("id") == father)
            .then(float("-inf"))
            .otherwise(pl.col("liability1"))
            .alias("liability1")
        )
        self._check(df)

    def test_all_infinite_column_is_nan(self):
        df = self._pedigree().with_columns(pl.lit(float("inf")).alias("liability1"))
        matrix, n_pairs = mate_correlation_matrix(mate_pedigree(df), df, ["liability1", "liability2"])
        assert n_pairs == _oracle(df, ["liability2"])[1]
        assert np.isnan(matrix[[0, 0, 1], [0, 1, 0]]).all()
        assert not np.isnan(matrix[1, 1])


class TestMateCorrelationByGeneration:
    """Each generation's pairs against the NumPy oracle on that generation's rows."""

    @staticmethod
    def _three_generations(seed: int = 9, n: int = 300) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        ids = np.arange(3 * n)
        mother = np.full(3 * n, -1)
        father = np.full(3 * n, -1)
        for g in (1, 2):
            parents = ids[(g - 1) * n : g * n]
            mother[g * n : (g + 1) * n] = rng.choice(parents[0::2], size=n)
            father[g * n : (g + 1) * n] = rng.choice(parents[1::2], size=n)
        return pl.DataFrame(
            {
                "id": ids,
                "mother": mother,
                "father": father,
                "generation": np.repeat([0, 1, 2], n),
                "liability1": rng.normal(size=3 * n),
                "liability2": rng.normal(size=3 * n),
            }
        )

    def test_matches_oracle_per_parents_generation(self):
        df = self._three_generations()
        cols = ["liability1", "liability2"]
        by_gen = mate_correlation_by_generation(df, cols, [0, 1])
        for g in (0, 1):
            expected, n_expected = _oracle(df.filter(pl.col("generation").is_in([g, g + 1])), cols)
            matrix, n_pairs = by_gen[g]
            assert n_pairs == n_expected
            np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-12)
        assert by_gen[0][1] + by_gen[1][1] == mate_correlation_matrix(mate_pedigree(df), df, cols)[1]
