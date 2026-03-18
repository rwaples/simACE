"""Tests for plot utility helpers and refactored plotting functions.

Covers save_placeholder_plot, finalize_plot, and the deduplicated
_plot_joint_grid in plot_liability.  Also smoke-tests representative
placeholder early-return paths across plot modules.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sim_ace.utils import finalize_plot, save_placeholder_plot

# ---------------------------------------------------------------------------
# save_placeholder_plot
# ---------------------------------------------------------------------------


class TestSavePlaceholderPlot:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "placeholder.png"
        save_placeholder_plot(out, "Test message")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_figsize(self, tmp_path):
        out = tmp_path / "placeholder_big.png"
        save_placeholder_plot(out, "Big figure", figsize=(10, 8))
        assert out.exists()

    def test_multiline_message(self, tmp_path):
        out = tmp_path / "placeholder_multi.png"
        save_placeholder_plot(out, "Line one\nLine two\nLine three")
        assert out.exists()

    def test_closes_figure(self, tmp_path):
        """Ensure no figure is left open after the call."""
        n_before = len(plt.get_fignums())
        save_placeholder_plot(tmp_path / "a.png", "msg")
        assert len(plt.get_fignums()) == n_before


# ---------------------------------------------------------------------------
# finalize_plot
# ---------------------------------------------------------------------------


class TestFinalizePlot:
    def test_saves_and_closes(self, tmp_path):
        out = tmp_path / "final.png"
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        finalize_plot(out)
        assert out.exists()
        assert out.stat().st_size > 0
        # Figure should be closed
        assert fig.number not in plt.get_fignums()

    def test_tight_rect(self, tmp_path):
        out = tmp_path / "final_rect.png"
        _fig, _ax = plt.subplots()
        _ax.plot([0, 1], [0, 1])
        finalize_plot(out, tight_rect=[0, 0, 1, 0.93])
        assert out.exists()

    def test_custom_dpi(self, tmp_path):
        out_lo = tmp_path / "lo.png"
        out_hi = tmp_path / "hi.png"
        _fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        finalize_plot(out_lo, dpi=50)
        _fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        finalize_plot(out_hi, dpi=300)
        # Higher DPI should produce a larger file
        assert out_hi.stat().st_size > out_lo.stat().st_size

    def test_no_open_figures_leak(self, tmp_path):
        """Call finalize_plot several times; no figures should accumulate."""
        n_before = len(plt.get_fignums())
        for i in range(5):
            _fig, ax = plt.subplots()
            ax.bar([1, 2], [3, 4])
            finalize_plot(tmp_path / f"leak_{i}.png")
        assert len(plt.get_fignums()) == n_before


# ---------------------------------------------------------------------------
# Fixtures for plot smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """Minimal DataFrame that satisfies plot_liability functions."""
    rng = np.random.default_rng(99)
    n = 200
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "sex": rng.integers(0, 2, n),
            "generation": np.repeat([0, 1, 2], [40, 80, 80]),
            "mother": np.concatenate([np.full(40, -1), rng.integers(0, 40, 160)]),
            "father": np.concatenate([np.full(40, -1), rng.integers(0, 40, 160)]),
            "liability1": rng.normal(size=n),
            "liability2": rng.normal(size=n),
            "A1": rng.normal(size=n) * 0.5,
            "A2": rng.normal(size=n) * 0.5,
            "C1": rng.normal(size=n) * 0.3,
            "C2": rng.normal(size=n) * 0.3,
            "E1": rng.normal(size=n) * 0.2,
            "E2": rng.normal(size=n) * 0.2,
            "affected1": rng.random(n) < 0.1,
            "affected2": rng.random(n) < 0.1,
            "t1": rng.exponential(50, n),
            "t2": rng.exponential(50, n),
            "t_observed1": rng.exponential(50, n),
            "t_observed2": rng.exponential(50, n),
            "death_age": rng.uniform(50, 90, n),
            "death_censored1": rng.random(n) < 0.3,
            "death_censored2": rng.random(n) < 0.3,
        }
    )


@pytest.fixture
def minimal_stats():
    """Minimal stats dict list (1 rep) for correlation plot functions."""
    return [
        {
            "tetrachoric": {
                "trait1": {
                    "MZ twin": {"r": 0.8, "n_pairs": 20},
                    "Full sib": {"r": 0.5, "n_pairs": 100},
                },
                "trait2": {
                    "MZ twin": {"r": 0.7, "n_pairs": 20},
                    "Full sib": {"r": 0.4, "n_pairs": 100},
                },
            },
            "liability_correlations": {"trait1": {}, "trait2": {}},
            "prevalence": {"trait1": 0.1, "trait2": 0.1},
            "regression": {"trait1": {"r2": 0.3, "slope": -5, "intercept": 80}, "trait2": None},
            "mortality": {"rates": [0.01] * 10, "decade_labels": [f"{i}0s" for i in range(10)]},
            "cumulative_incidence": {
                "trait1": {"ages": list(range(100)), "values": [i / 1000 for i in range(100)]},
                "trait2": {"ages": list(range(100)), "values": [i / 1000 for i in range(100)]},
            },
        }
    ]


# ---------------------------------------------------------------------------
# plot_liability: _plot_joint_grid via public wrappers
# ---------------------------------------------------------------------------


class TestPlotLiabilityJointGrid:
    """Smoke tests for the refactored _plot_joint_grid function."""

    def test_joint_plain(self, tmp_path, sample_df):
        from sim_ace.plot_liability import plot_liability_joint

        out = tmp_path / "joint.png"
        plot_liability_joint(sample_df, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_joint_affected(self, tmp_path, sample_df):
        from sim_ace.plot_liability import plot_liability_joint_affected

        out = tmp_path / "joint_aff.png"
        plot_liability_joint_affected(sample_df, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_joint_missing_columns(self, tmp_path, sample_df):
        """Works when some component columns (C, E) are absent."""
        from sim_ace.plot_liability import plot_liability_joint

        df = sample_df.drop(columns=["C1", "C2", "E1", "E2"])
        out = tmp_path / "joint_partial.png"
        plot_liability_joint(df, out, scenario="test")
        assert out.exists()

    def test_no_figure_leak(self, tmp_path, sample_df):
        from sim_ace.plot_liability import plot_liability_joint

        n_before = len(plt.get_fignums())
        plot_liability_joint(sample_df, tmp_path / "leak.png")
        assert len(plt.get_fignums()) == n_before


# ---------------------------------------------------------------------------
# Placeholder early-return paths (smoke tests)
# ---------------------------------------------------------------------------


class TestPlaceholderPaths:
    """Verify that placeholder early-return paths produce valid files."""

    def test_tetrachoric_by_gen_no_data(self, tmp_path):
        from sim_ace.plot_correlations import plot_tetrachoric_by_generation

        out = tmp_path / "tet_gen.png"
        plot_tetrachoric_by_generation([{}], out, scenario="test")
        assert out.exists()

    def test_heritability_no_data(self, tmp_path):
        from sim_ace.plot_correlations import plot_heritability_by_generation

        out = tmp_path / "h2.png"
        plot_heritability_by_generation([{}], out, scenario="test")
        assert out.exists()

    def test_broad_heritability_no_data(self, tmp_path):
        from sim_ace.plot_correlations import plot_broad_heritability_by_generation

        out = tmp_path / "H2.png"
        plot_broad_heritability_by_generation([{}], out, scenario="test")
        assert out.exists()

    def test_cross_trait_frailty_no_data(self, tmp_path):
        from sim_ace.plot_correlations import plot_cross_trait_frailty_by_generation

        out = tmp_path / "ct.png"
        plot_cross_trait_frailty_by_generation([{}], out, scenario="test")
        assert out.exists()

    def test_parent_offspring_no_generation(self, tmp_path, sample_df):
        from sim_ace.plot_correlations import plot_parent_offspring_liability

        df = sample_df.drop(columns=["generation"])
        out = tmp_path / "po.png"
        plot_parent_offspring_liability(df, [{}], out, scenario="test")
        assert out.exists()

    def test_cumulative_by_sex_no_data(self, tmp_path):
        from sim_ace.plot_distributions import plot_cumulative_incidence_by_sex

        out = tmp_path / "ci_sex.png"
        plot_cumulative_incidence_by_sex([{}], out, scenario="test")
        assert out.exists()

    def test_cumulative_by_sex_gen_no_data(self, tmp_path):
        from sim_ace.plot_distributions import plot_cumulative_incidence_by_sex_generation

        out = tmp_path / "ci_sg.png"
        plot_cumulative_incidence_by_sex_generation([{}], out, scenario="test")
        assert out.exists()

    def test_censoring_windows_no_data(self, tmp_path):
        from sim_ace.plot_distributions import plot_censoring_windows

        out = tmp_path / "cw.png"
        plot_censoring_windows([{"censoring": None}], out, scenario="test")
        assert out.exists()

    def test_liability_violin_by_gen_no_gen(self, tmp_path, sample_df):
        from sim_ace.plot_liability import plot_liability_violin_by_generation

        df = sample_df.drop(columns=["generation"])
        out = tmp_path / "lv_gen.png"
        plot_liability_violin_by_generation(df, [{}], out, scenario="test")
        assert out.exists()

    def test_censoring_confusion_no_data(self, tmp_path):
        from sim_ace.plot_liability import plot_censoring_confusion

        out = tmp_path / "cc.png"
        plot_censoring_confusion([{}], out, scenario="test")
        assert out.exists()

    def test_censoring_cascade_no_data(self, tmp_path):
        from sim_ace.plot_liability import plot_censoring_cascade

        out = tmp_path / "cascade.png"
        plot_censoring_cascade([{}], out, scenario="test")
        assert out.exists()

    def test_simple_ltm_violin_by_gen_no_gen(self, tmp_path, sample_df):
        from sim_ace.plot_simple_ltm import plot_liability_violin_by_generation

        df = sample_df.drop(columns=["generation"])
        out = tmp_path / "tv_gen.png"
        plot_liability_violin_by_generation(df, [{}], 0.1, 0.1, out, scenario="test")
        assert out.exists()

    def test_pedigree_counts_no_data(self, tmp_path):
        from sim_ace.plot_pedigree_counts import plot_pedigree_relationship_counts

        out = tmp_path / "ped.png"
        plot_pedigree_relationship_counts([{}], out, scenario="test")
        assert out.exists()


# ---------------------------------------------------------------------------
# Finalize paths in full plot functions (non-placeholder)
# ---------------------------------------------------------------------------


class TestFinalizePaths:
    """Smoke tests that full (non-placeholder) plot functions complete and close."""

    def test_liability_violin(self, tmp_path, sample_df, minimal_stats):
        from sim_ace.plot_liability import plot_liability_violin

        out = tmp_path / "violin.png"
        plot_liability_violin(sample_df, minimal_stats, out, scenario="test")
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_joint_affection(self, tmp_path, sample_df, minimal_stats):
        from sim_ace.plot_liability import plot_joint_affection

        # Add joint_affection and cross_trait_tetrachoric to stats
        stats = minimal_stats[0].copy()
        stats["joint_affection"] = {
            "counts": {"both": 5, "trait1_only": 15, "trait2_only": 15, "neither": 165},
            "proportions": {"both": 0.025, "trait1_only": 0.075, "trait2_only": 0.075, "neither": 0.825},
            "n": 200,
        }
        stats["cross_trait_tetrachoric"] = {"same_person": {"r": 0.3, "se": 0.05, "n": 200}}
        out = tmp_path / "joint_aff.png"
        plot_joint_affection([stats], out, scenario="test")
        assert out.exists()

    def test_censoring_confusion_full(self, tmp_path):
        from sim_ace.plot_liability import plot_censoring_confusion

        stats = [
            {
                "censoring_confusion": {
                    "trait1": {"tp": 50, "fn": 10, "fp": 2, "tn": 138, "n": 200},
                    "trait2": {"tp": 40, "fn": 15, "fp": 1, "tn": 144, "n": 200},
                },
            }
        ]
        out = tmp_path / "cc_full.png"
        plot_censoring_confusion(stats, out, scenario="test")
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_censoring_cascade_full(self, tmp_path):
        from sim_ace.plot_liability import plot_censoring_cascade

        stats = [
            {
                "censoring_cascade": {
                    "trait1": {
                        "gen1": {
                            "observed": 30,
                            "death_censored": 5,
                            "right_censored": 10,
                            "left_truncated": 5,
                            "true_affected": 50,
                            "n_gen": 100,
                            "sensitivity": 0.6,
                            "window": [20, 80],
                        },
                    },
                    "trait2": {
                        "gen1": {
                            "observed": 25,
                            "death_censored": 8,
                            "right_censored": 12,
                            "left_truncated": 5,
                            "true_affected": 50,
                            "n_gen": 100,
                            "sensitivity": 0.5,
                            "window": [20, 80],
                        },
                    },
                },
            }
        ]
        out = tmp_path / "cascade_full.png"
        plot_censoring_cascade(stats, out, scenario="test")
        assert out.exists()

    def test_death_age_distribution(self, tmp_path, minimal_stats):
        from sim_ace.plot_distributions import plot_death_age_distribution

        out = tmp_path / "mortality.png"
        plot_death_age_distribution(minimal_stats, 100.0, out, scenario="test")
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_trait_phenotype(self, tmp_path, sample_df):
        from sim_ace.plot_distributions import plot_trait_phenotype

        out = tmp_path / "phenotype.png"
        plot_trait_phenotype(sample_df, out, scenario="test")
        assert out.exists()

    def test_cumulative_incidence(self, tmp_path, minimal_stats):
        from sim_ace.plot_distributions import plot_cumulative_incidence

        out = tmp_path / "ci.png"
        plot_cumulative_incidence(minimal_stats, 100.0, out, scenario="test")
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_cumulative_incidence_by_sex(self, tmp_path):
        from sim_ace.plot_distributions import plot_cumulative_incidence_by_sex

        ages = list(range(100))
        values = [i / 1000 for i in range(100)]
        stats = [
            {
                "cumulative_incidence_by_sex": {
                    "trait1": {
                        "female": {"ages": ages, "values": values, "n": 50, "prevalence": 0.1},
                        "male": {"ages": ages, "values": values, "n": 50, "prevalence": 0.12},
                    },
                    "trait2": {
                        "female": {"ages": ages, "values": values, "n": 50, "prevalence": 0.08},
                        "male": {"ages": ages, "values": values, "n": 50, "prevalence": 0.09},
                    },
                },
            }
        ]
        out = tmp_path / "ci_sex.png"
        plot_cumulative_incidence_by_sex(stats, out, scenario="test")
        assert out.exists()
