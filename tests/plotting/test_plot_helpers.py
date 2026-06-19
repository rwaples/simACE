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

from simace.plotting.plot_utils import finalize_plot, save_placeholder_plot

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
                    "MZ": {"r": 0.8, "n_pairs": 20},
                    "FS": {"r": 0.5, "n_pairs": 100},
                },
                "trait2": {
                    "MZ": {"r": 0.7, "n_pairs": 20},
                    "FS": {"r": 0.4, "n_pairs": 100},
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
        from simace.plotting.plot_liability import plot_liability_joint

        out = tmp_path / "joint.png"
        plot_liability_joint(sample_df, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_joint_affected(self, tmp_path, sample_df):
        from simace.plotting.plot_liability import plot_liability_joint_affected

        out = tmp_path / "joint_aff.png"
        plot_liability_joint_affected(sample_df, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_joint_missing_columns(self, tmp_path, sample_df):
        """Works when some component columns (C, E) are absent."""
        from simace.plotting.plot_liability import plot_liability_joint

        df = sample_df.drop(columns=["C1", "C2", "E1", "E2"])
        out = tmp_path / "joint_partial.png"
        plot_liability_joint(df, out, scenario="test")
        assert out.exists()

    def test_no_figure_leak(self, tmp_path, sample_df):
        from simace.plotting.plot_liability import plot_liability_joint

        n_before = len(plt.get_fignums())
        plot_liability_joint(sample_df, tmp_path / "leak.png")
        assert len(plt.get_fignums()) == n_before


# ---------------------------------------------------------------------------
# Placeholder early-return paths (smoke tests)
# ---------------------------------------------------------------------------


class TestPlaceholderPaths:
    """Verify that placeholder early-return paths produce valid files."""

    def test_tetrachoric_by_gen_no_data(self, tmp_path):
        from simace.plotting.plot_correlations import plot_tetrachoric_by_generation

        out = tmp_path / "tet_gen.png"
        plot_tetrachoric_by_generation([{}], out, scenario="test")
        assert out.exists()

    def test_heritability_no_data(self, tmp_path):
        from simace.plotting.plot_heritability import plot_heritability_by_generation

        out = tmp_path / "h2.png"
        plot_heritability_by_generation([{}], out, scenario="test")
        assert out.exists()

    def test_broad_heritability_no_data(self, tmp_path):
        from simace.plotting.plot_heritability import plot_broad_heritability_by_generation

        out = tmp_path / "H2.png"
        plot_broad_heritability_by_generation([{}], out, scenario="test")
        assert out.exists()

    def test_parent_offspring_no_generation(self, tmp_path, sample_df):
        from simace.plotting.plot_correlations import plot_parent_offspring_liability

        df = sample_df.drop(columns=["generation"])
        out = tmp_path / "po.png"
        plot_parent_offspring_liability(df, [{}], out, scenario="test")
        assert out.exists()

    def test_parent_offspring_with_params(self, tmp_path, sample_df):
        from simace.plotting.plot_correlations import plot_parent_offspring_liability

        out = tmp_path / "po_params.png"
        params = {"A1": 0.5, "C1": 0.3, "A2": 0.4, "C2": 0.2}
        plot_parent_offspring_liability(sample_df, [{}], out, scenario="test", params=params)
        assert out.exists()

    def test_cumulative_by_sex_no_data(self, tmp_path):
        from simace.plotting.plot_distributions import plot_cumulative_incidence_by_sex

        out = tmp_path / "ci_sex.png"
        plot_cumulative_incidence_by_sex([{}], out, scenario="test")
        assert out.exists()

    def test_cumulative_by_sex_gen_no_data(self, tmp_path):
        from simace.plotting.plot_distributions import plot_cumulative_incidence_by_sex_generation

        out = tmp_path / "ci_sg.png"
        plot_cumulative_incidence_by_sex_generation([{}], out, scenario="test")
        assert out.exists()

    def test_censoring_windows_no_data(self, tmp_path):
        from simace.plotting.plot_distributions import plot_censoring_windows

        out = tmp_path / "cw.png"
        plot_censoring_windows([{"censoring": None}], out, scenario="test")
        assert out.exists()

    def test_liability_violin_by_gen_no_gen(self, tmp_path, sample_df):
        from simace.plotting.plot_liability import plot_liability_violin_by_generation

        df = sample_df.drop(columns=["generation"])
        out = tmp_path / "lv_gen.png"
        plot_liability_violin_by_generation(df, [{}], out, scenario="test")
        assert out.exists()

    def test_censoring_confusion_no_data(self, tmp_path):
        from simace.plotting.plot_liability import plot_censoring_confusion

        out = tmp_path / "cc.png"
        plot_censoring_confusion([{}], out, scenario="test")
        assert out.exists()

    def test_censoring_cascade_no_data(self, tmp_path):
        from simace.plotting.plot_liability import plot_censoring_cascade

        out = tmp_path / "cascade.png"
        plot_censoring_cascade([{}], out, scenario="test")
        assert out.exists()

    def test_pedigree_counts_no_data(self, tmp_path):
        from simace.plotting.plot_pedigree_counts import plot_pedigree_relationship_counts

        out = tmp_path / "ped.png"
        plot_pedigree_relationship_counts([{}], out, scenario="test")
        assert out.exists()


# ---------------------------------------------------------------------------
# Finalize paths in full plot functions (non-placeholder)
# ---------------------------------------------------------------------------


class TestFinalizePaths:
    """Smoke tests that full (non-placeholder) plot functions complete and close."""

    def test_liability_violin(self, tmp_path, sample_df, minimal_stats):
        from simace.plotting.plot_liability import plot_liability_violin

        out = tmp_path / "violin.png"
        plot_liability_violin(sample_df, minimal_stats, out, scenario="test")
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_joint_affection(self, tmp_path, sample_df, minimal_stats):
        from simace.plotting.plot_liability import plot_joint_affection

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

    def test_mate_correlation_heatmaps_have_cell_borders(self, tmp_path, monkeypatch):
        import simace.plotting.plot_liability as plot_liability

        heatmap_kwargs = []

        def fake_heatmap(*_args, **kwargs):
            heatmap_kwargs.append(kwargs)

        monkeypatch.setattr(plot_liability.sns, "heatmap", fake_heatmap)
        stats = [{"mate_correlation": {"matrix": [[0.1, 0.2], [0.3, 0.4]]}}]
        out = tmp_path / "mate_correlation.png"

        plot_liability.plot_mate_correlation(stats, out, scenario="test")

        assert out.exists()
        assert len(heatmap_kwargs) == 2
        assert all(kwargs["linewidths"] == 0.5 for kwargs in heatmap_kwargs)
        assert all(kwargs["linecolor"] == "black" for kwargs in heatmap_kwargs)
        assert len(plt.get_fignums()) == 0

    def test_censoring_confusion_heatmaps_are_unshaded_with_cell_borders(self, tmp_path, monkeypatch):
        import simace.plotting.plot_liability as plot_liability

        heatmap_kwargs = []

        def fake_heatmap(*_args, **kwargs):
            heatmap_kwargs.append(kwargs)

        monkeypatch.setattr(plot_liability.sns, "heatmap", fake_heatmap)
        stats = [
            {
                "censoring_confusion": {
                    "trait1": {"tp": 50, "fn": 10, "fp": 2, "tn": 138, "n": 200},
                    "trait2": {"tp": 40, "fn": 15, "fp": 1, "tn": 144, "n": 200},
                },
            }
        ]
        out = tmp_path / "cc_borders.png"

        plot_liability.plot_censoring_confusion(stats, out, scenario="test")

        assert out.exists()
        assert len(heatmap_kwargs) == 2
        assert all(kwargs["cbar"] is False for kwargs in heatmap_kwargs)
        assert all(kwargs["cmap"].colors == ["white"] for kwargs in heatmap_kwargs)
        assert all(kwargs["linewidths"] == 0.5 for kwargs in heatmap_kwargs)
        assert all(kwargs["linecolor"] == "black" for kwargs in heatmap_kwargs)
        assert len(plt.get_fignums()) == 0

    def test_censoring_confusion_full(self, tmp_path):
        from simace.plotting.plot_liability import plot_censoring_confusion

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

    def test_censoring_cascade_reserves_top_space_for_legend(self, tmp_path, monkeypatch):
        import simace.plotting.plot_liability as plot_liability

        finalize_kwargs = {}

        def fake_finalize(output_path, **kwargs):
            finalize_kwargs.update(kwargs)
            output_path.write_bytes(b"placeholder")
            plt.close(plt.gcf())

        monkeypatch.setattr(plot_liability, "finalize_plot", fake_finalize)
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
                            "window": [20, 80],
                        },
                    },
                },
            }
        ]
        out = tmp_path / "cascade_spacing.png"

        plot_liability.plot_censoring_cascade(stats, out, scenario="test")

        assert out.exists()
        assert finalize_kwargs["tight_rect"] == [0, 0, 1, 0.90]
        assert len(plt.get_fignums()) == 0

    def test_censoring_cascade_full(self, tmp_path):
        from simace.plotting.plot_liability import plot_censoring_cascade

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
        from simace.plotting.plot_distributions import plot_death_age_distribution

        out = tmp_path / "mortality.png"
        plot_death_age_distribution(minimal_stats, 100.0, out, scenario="test")
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_death_age_distribution_with_death_age_histograms(self, tmp_path, minimal_stats, sample_df):
        from simace.plotting.plot_distributions import plot_death_age_distribution

        out = tmp_path / "mortality_with_death_ages.png"
        plot_death_age_distribution(minimal_stats, 100.0, out, scenario="test", df_samples=sample_df)
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_death_age_distribution_tiny_rates(self, tmp_path):
        # Regression: a fixed +0.01 data-coord offset on bar labels used to
        # expand the saved PNG to ~120M pixels when rates were ~1e-5
        # (death_scale=1e6 scenarios). bar_label(padding=) fixes it.
        from PIL import Image

        from simace.plotting.plot_distributions import plot_death_age_distribution

        stats = [{"mortality": {"rates": [1e-5] * 8, "decade_labels": [f"{i}0s" for i in range(8)]}}]
        out = tmp_path / "mortality.png"
        plot_death_age_distribution(stats, 80.0, out, scenario="test")
        with Image.open(out) as im:
            assert im.width * im.height < 5_000_000, (
                f"image exploded to {im.width}x{im.height} ({im.width * im.height / 1e6:.0f}M px)"
            )

    def test_trait_phenotype(self, tmp_path, sample_df):
        from simace.plotting.plot_distributions import plot_trait_phenotype

        out = tmp_path / "phenotype.png"
        plot_trait_phenotype(sample_df, out, scenario="test")
        assert out.exists()

    def test_cumulative_incidence(self, tmp_path, minimal_stats):
        from simace.plotting.plot_distributions import plot_cumulative_incidence

        out = tmp_path / "ci.png"
        plot_cumulative_incidence(minimal_stats, 100.0, out, scenario="test")
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_cumulative_incidence_by_sex(self, tmp_path):
        from simace.plotting.plot_distributions import plot_cumulative_incidence_by_sex

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

    def test_cumulative_incidence_aj(self, tmp_path, minimal_stats):
        from simace.plotting.plot_distributions import plot_cumulative_incidence_aj

        ages = list(range(100))
        aj_vals = [i / 1000 for i in range(100)]
        aj_death = [i / 2000 for i in range(100)]
        aj_surv = [1.0 - aj_vals[i] - aj_death[i] for i in range(100)]
        stats = [
            {
                **minimal_stats[0],
                "cumulative_incidence_aj": {
                    "trait1": {
                        "ages": ages,
                        "aj_values": aj_vals,
                        "aj_death_values": aj_death,
                        "aj_survival": aj_surv,
                        "n": 100,
                        "n_events_disease": 10,
                        "n_events_death": 5,
                        "half_target_age": 50.0,
                    },
                    "trait2": {
                        "ages": ages,
                        "aj_values": aj_vals,
                        "aj_death_values": aj_death,
                        "aj_survival": aj_surv,
                        "n": 100,
                        "n_events_disease": 8,
                        "n_events_death": 5,
                        "half_target_age": 55.0,
                    },
                },
            }
        ]
        out = tmp_path / "ci_aj.png"
        plot_cumulative_incidence_aj(stats, 100.0, out, scenario="test")
        assert out.exists()
        assert len(plt.get_fignums()) == 0

    def test_cumulative_incidence_aj_includes_true_cif_line_without_death_line(self, tmp_path, monkeypatch):
        import simace.plotting.plot_distributions as plot_distributions

        ages = [0, 1, 2, 3]
        aj_vals = [0.0, 0.1, 0.2, 0.3]
        aj_death = [0.0, 0.02, 0.04, 0.06]
        aj_surv = [1.0 - aj_vals[i] - aj_death[i] for i in range(len(ages))]
        true_vals = [0.0, 0.12, 0.24, 0.36]
        observed_vals = [0.0, 0.08, 0.16, 0.24]
        seen_labels = []

        def fake_finalize(output_path, **_kwargs):
            fig = plt.gcf()
            for ax in fig.axes:
                seen_labels.extend(line.get_label() for line in ax.lines)
            output_path.write_bytes(b"placeholder")
            plt.close(fig)

        monkeypatch.setattr(plot_distributions, "finalize_plot", fake_finalize)
        stats = [
            {
                "cumulative_incidence": {
                    "trait1": {"ages": ages, "observed_values": observed_vals, "true_values": true_vals},
                    "trait2": {"ages": ages, "observed_values": observed_vals, "true_values": true_vals},
                },
                "cumulative_incidence_aj": {
                    "trait1": {
                        "ages": ages,
                        "aj_values": aj_vals,
                        "aj_death_values": aj_death,
                        "aj_survival": aj_surv,
                        "n": 100,
                        "n_events_disease": 10,
                        "n_events_death": 5,
                        "half_target_age": 2.0,
                    },
                    "trait2": {
                        "ages": ages,
                        "aj_values": aj_vals,
                        "aj_death_values": aj_death,
                        "aj_survival": aj_surv,
                        "n": 100,
                        "n_events_disease": 8,
                        "n_events_death": 5,
                        "half_target_age": 2.0,
                    },
                },
            }
        ]
        out = tmp_path / "ci_aj_true.png"

        plot_distributions.plot_cumulative_incidence_aj(stats, 3.0, out, scenario="test")

        assert out.exists()
        assert seen_labels.count("True CIF") == 2
        assert "AJ death" not in seen_labels
        assert len(plt.get_fignums()) == 0

    def test_cumulative_incidence_aj_by_sex(self, tmp_path):
        from simace.plotting.plot_distributions import plot_cumulative_incidence_aj_by_sex

        ages = list(range(100))
        vals = [i / 1000 for i in range(100)]
        deaths = [i / 2000 for i in range(100)]
        surv = [1.0 - vals[i] - deaths[i] for i in range(100)]

        def per_sex():
            return {
                "ages": ages,
                "aj_values": vals,
                "aj_death_values": deaths,
                "aj_survival": surv,
                "n": 50,
                "n_events_disease": 5,
                "n_events_death": 2,
                "prevalence": 0.10,
            }

        stats = [
            {
                "cumulative_incidence_aj_by_sex": {
                    "trait1": {"female": per_sex(), "male": per_sex()},
                    "trait2": {"female": per_sex(), "male": per_sex()},
                }
            }
        ]
        out = tmp_path / "ci_aj_sex.png"
        plot_cumulative_incidence_aj_by_sex(stats, out, scenario="test")
        assert out.exists()

    def test_cumulative_incidence_aj_by_sex_generation(self, tmp_path):
        from simace.plotting.plot_distributions import plot_cumulative_incidence_aj_by_sex_generation

        ages = list(range(100))
        vals = [i / 1000 for i in range(100)]
        deaths = [i / 2000 for i in range(100)]
        surv = [1.0 - vals[i] - deaths[i] for i in range(100)]

        def per_sex():
            return {
                "ages": ages,
                "aj_values": vals,
                "aj_death_values": deaths,
                "aj_survival": surv,
                "n": 25,
                "n_events_disease": 2,
                "n_events_death": 1,
                "prevalence": 0.08,
            }

        stats = [
            {
                "cumulative_incidence_aj_by_sex_generation": {
                    "trait1": {
                        "gen0": {"female": per_sex(), "male": per_sex()},
                        "gen1": {"female": per_sex(), "male": per_sex()},
                    },
                    "trait2": {
                        "gen0": {"female": per_sex(), "male": per_sex()},
                        "gen1": {"female": per_sex(), "male": per_sex()},
                    },
                }
            }
        ]
        out = tmp_path / "ci_aj_sex_gen.png"
        plot_cumulative_incidence_aj_by_sex_generation(stats, out, scenario="test")
        assert out.exists()

    def test_cumulative_incidence_aj_missing_keys_placeholder(self, tmp_path, minimal_stats):
        """When stats lack the AJ keys, the plot must produce a placeholder, not crash."""
        from simace.plotting.plot_distributions import (
            plot_cumulative_incidence_aj,
            plot_cumulative_incidence_aj_by_sex,
            plot_cumulative_incidence_aj_by_sex_generation,
        )

        out1 = tmp_path / "aj_missing.png"
        plot_cumulative_incidence_aj(minimal_stats, 100.0, out1, scenario="test")
        assert out1.exists()

        out2 = tmp_path / "aj_sex_missing.png"
        plot_cumulative_incidence_aj_by_sex([{}], out2, scenario="test")
        assert out2.exists()

        out3 = tmp_path / "aj_sex_gen_missing.png"
        plot_cumulative_incidence_aj_by_sex_generation([{}], out3, scenario="test")
        assert out3.exists()
