"""Smoke tests for untested plot modules.

Each test calls the render/plot function with minimal inputs and asserts
a matplotlib Figure is returned and no figures leak.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_params():
    """Minimal scenario parameter dict for pipeline/table1 rendering."""
    return {
        "seed": 42,
        "replicates": 1,
        "N": 1000,
        "G_ped": 3,
        "G_sim": 4,
        "G_pheno": 3,
        "mating_lambda": 0.5,
        "p_mztwin": 0.02,
        "A1": 0.5,
        "C1": 0.2,
        "A2": 0.5,
        "C2": 0.2,
        "rA": 0.3,
        "rC": 0.5,
        "assort1": 0.0,
        "assort2": 0.0,
        "beta1": 1.0,
        "beta2": 1.5,
        "beta_sex1": 0.0,
        "beta_sex2": 0.0,
        "phenotype_model1": "frailty_weibull",
        "phenotype_model2": "frailty_weibull",
        "phenotype_params1": {"scale": 2160, "rho": 0.8},
        "phenotype_params2": {"scale": 333, "rho": 1.2},
        "censor_age": 80,
        "gen_censoring": {0: [80, 80], 1: [80, 80], 2: [0, 80]},
        "death_scale": 164,
        "death_rho": 2.73,
        "standardize": True,
        "N_sample": 0,
        "case_ascertainment_ratio": 1,
        "pedigree_dropout_rate": 0,
        "prevalence1": 0.10,
        "prevalence2": 0.20,
    }


@pytest.fixture
def minimal_stats():
    """Minimal phenotype_stats dict for Table 1 rendering."""
    return {
        "n_individuals": 500,
        "n_males": 250,
        "n_females": 250,
        "prevalence": {"trait1": 0.10, "trait2": 0.20},
        "joint_affection": {
            "counts": {"both": 10, "trait1_only": 40, "trait2_only": 90, "neither": 360},
            "proportions": {"both": 0.02, "trait1_only": 0.08, "trait2_only": 0.18, "neither": 0.72},
            "n": 500,
            "by_sex": {"female": 0.02, "male": 0.02},
        },
        "person_years": {"total": 20000.0, "deaths": 50, "trait1": 18000.0, "trait2": 16000.0},
        "family_size": {
            "mean": 2.3,
            "median": 2.0,
            "q1": 1.0,
            "q3": 3.0,
            "n_families": 200,
            "frac_with_full_sib": 0.7,
            "size_dist": {"1": 0.25, "2": 0.35, "3": 0.25, "4+": 0.15},
            "person_offspring_dist": {"0": 0.4, "1": 0.2, "2": 0.2, "3": 0.1, "4+": 0.1},
            "mates_by_sex": {
                "female_mean": 1.1,
                "male_mean": 1.1,
                "female_1": 0.8,
                "female_2+": 0.2,
                "male_1": 0.8,
                "male_2+": 0.2,
            },
        },
        "parent_status": {"phenotyped": {"0": 100, "1": 200, "2": 200}},
        "censoring": {
            "generations": {
                "gen2": {
                    "n": 500,
                    "trait1": {
                        "pct_affected": 0.10,
                        "left_censored": 0.0,
                        "right_censored": 0.05,
                        "death_censored": 0.02,
                    },
                    "trait2": {
                        "pct_affected": 0.20,
                        "left_censored": 0.0,
                        "right_censored": 0.03,
                        "death_censored": 0.01,
                    },
                },
            },
            "censoring_ages": list(np.linspace(0, 80, 10)),
            "censor_age": 80,
        },
        "censoring_cascade": {
            "trait1": {
                "gen2": {
                    "observed": 45,
                    "death_censored": 3,
                    "right_censored": 2,
                    "left_truncated": 0,
                    "true_affected": 50,
                    "n_gen": 500,
                    "sensitivity": 0.90,
                    "window": [0, 80],
                },
            },
            "trait2": {
                "gen2": {
                    "observed": 95,
                    "death_censored": 3,
                    "right_censored": 2,
                    "left_truncated": 0,
                    "true_affected": 100,
                    "n_gen": 500,
                    "sensitivity": 0.95,
                    "window": [0, 80],
                },
            },
        },
    }


@pytest.fixture
def validation_df():
    """Minimal validation summary DataFrame for plot_validation functions."""
    rng = np.random.default_rng(42)
    n = 6  # 2 scenarios × 3 reps
    return pd.DataFrame(
        {
            "scenario": ["scA"] * 3 + ["scB"] * 3,
            "rep": [1, 2, 3, 1, 2, 3],
            "N": [1000] * n,
            "G_ped": [3] * n,
            "G_sim": [4] * n,
            "A1": [0.5] * n,
            "C1": [0.2] * n,
            "E1": [0.3] * n,
            "A2": [0.5] * n,
            "C2": [0.2] * n,
            "E2": [0.3] * n,
            "rA": [0.3] * n,
            "rC": [0.5] * n,
            "p_mztwin": [0.02] * n,
            "mating_lambda": [0.5] * n,
            "assort1": [0.0] * n,
            "assort2": [0.0] * n,
            "checks_failed": [0] * n,
            "observed_twin_rate": rng.normal(0.02, 0.002, n),
            "variance_A1": rng.normal(0.5, 0.02, n),
            "variance_C1": rng.normal(0.2, 0.02, n),
            "variance_E1": rng.normal(0.3, 0.02, n),
            "variance_A2": rng.normal(0.5, 0.02, n),
            "variance_C2": rng.normal(0.2, 0.02, n),
            "variance_E2": rng.normal(0.3, 0.02, n),
            "observed_rA": rng.normal(0.3, 0.02, n),
            "observed_rC": rng.normal(0.5, 0.02, n),
            "observed_rE": rng.normal(0.0, 0.02, n),
            "mz_twin_A1_corr": rng.normal(1.0, 0.01, n),
            "mz_twin_liability1_corr": rng.normal(0.7, 0.02, n),
            "mz_twin_A2_corr": rng.normal(1.0, 0.01, n),
            "mz_twin_liability2_corr": rng.normal(0.7, 0.02, n),
            "dz_sibling_A1_corr": rng.normal(0.5, 0.02, n),
            "dz_sibling_liability1_corr": rng.normal(0.5, 0.02, n),
            "dz_sibling_A2_corr": rng.normal(0.5, 0.02, n),
            "dz_sibling_liability2_corr": rng.normal(0.5, 0.02, n),
            "half_sib_prop_expected": [0.1] * n,
            "half_sib_prop_observed": rng.normal(0.1, 0.01, n),
            "offspring_with_half_sib_expected": [0.15] * n,
            "offspring_with_half_sib_observed": rng.normal(0.15, 0.01, n),
            "half_sib_A1_corr": rng.normal(0.25, 0.02, n),
            "half_sib_liability1_corr": rng.normal(0.25, 0.02, n),
            "half_sib_shared_C1": rng.normal(0.0, 0.01, n),
            "mate_corr_liability1": rng.normal(0.0, 0.01, n),
            "mate_corr_liability2": rng.normal(0.0, 0.01, n),
            "simulate_seconds": rng.uniform(1, 5, n),
            "simulate_max_rss_mb": rng.uniform(100, 500, n),
            "mother_mean_offspring": rng.normal(2.3, 0.1, n),
            "father_mean_offspring": rng.normal(2.3, 0.1, n),
            "falconer_h2_trait1": rng.normal(0.5, 0.02, n),
            "falconer_h2_trait2": rng.normal(0.5, 0.02, n),
            "parent_offspring_A1_slope": rng.normal(0.5, 0.02, n),
            "parent_offspring_A1_r2": rng.normal(0.5, 0.02, n),
            "parent_offspring_liability1_slope": rng.normal(0.5, 0.02, n),
            "parent_offspring_liability1_r2": rng.normal(0.5, 0.02, n),
            "parent_offspring_A2_slope": rng.normal(0.5, 0.02, n),
            "parent_offspring_A2_r2": rng.normal(0.5, 0.02, n),
            "parent_offspring_liability2_slope": rng.normal(0.5, 0.02, n),
            "parent_offspring_liability2_r2": rng.normal(0.5, 0.02, n),
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_no_leaked_figures(before: list[int], fig):
    """Close the returned figure and verify no leaked figure numbers."""
    plt.close(fig)
    after = plt.get_fignums()
    assert after == before, f"Leaked figures: {set(after) - set(before)}"


# ---------------------------------------------------------------------------
# plot_pipeline
# ---------------------------------------------------------------------------


class TestPlotPipeline:
    def test_render_pipeline_figure_returns_figure(self, minimal_params):
        from sim_ace.plotting.plot_pipeline import render_pipeline_figure

        before = plt.get_fignums()
        fig = render_pipeline_figure(minimal_params, scenario="test_scenario")
        assert isinstance(fig, plt.Figure)
        _assert_no_leaked_figures(before, fig)

    def test_render_pipeline_figure_no_scenario(self, minimal_params):
        from sim_ace.plotting.plot_pipeline import render_pipeline_figure

        before = plt.get_fignums()
        fig = render_pipeline_figure(minimal_params, scenario="")
        assert isinstance(fig, plt.Figure)
        _assert_no_leaked_figures(before, fig)


# ---------------------------------------------------------------------------
# plot_table1
# ---------------------------------------------------------------------------


class TestPlotTable1:
    def test_render_table1_figure_returns_figure(self, minimal_stats, minimal_params):
        from sim_ace.plotting.plot_table1 import render_table1_figure

        before = plt.get_fignums()
        fig = render_table1_figure([minimal_stats], minimal_params, scenario="test")
        assert isinstance(fig, plt.Figure)
        _assert_no_leaked_figures(before, fig)

    def test_render_table1_figure_multiple_reps(self, minimal_stats, minimal_params):
        from sim_ace.plotting.plot_table1 import render_table1_figure

        before = plt.get_fignums()
        fig = render_table1_figure([minimal_stats, minimal_stats], minimal_params, scenario="test")
        assert isinstance(fig, plt.Figure)
        _assert_no_leaked_figures(before, fig)


# ---------------------------------------------------------------------------
# plot_atlas helpers
# ---------------------------------------------------------------------------


class TestPlotAtlasHelpers:
    def test_get_model_equation_weibull(self, minimal_params):
        from sim_ace.plotting.plot_atlas import get_model_equation

        lines = get_model_equation(minimal_params)
        assert isinstance(lines, list)
        assert len(lines) >= 1

    def test_get_model_equation_different_models(self):
        from sim_ace.plotting.plot_atlas import get_model_equation

        params = {"phenotype_model1": "frailty_weibull", "phenotype_model2": "frailty_gompertz"}
        lines = get_model_equation(params)
        assert len(lines) >= 2  # one set per model

    def test_get_model_family_same_model(self, minimal_params):
        from sim_ace.plotting.plot_atlas import get_model_family

        name, desc = get_model_family(minimal_params)
        assert isinstance(name, str)
        assert isinstance(desc, str)
        assert "Weibull" in name

    def test_get_model_family_different_models(self):
        from sim_ace.plotting.plot_atlas import get_model_family

        params = {"phenotype_model1": "frailty_weibull", "phenotype_model2": "frailty_gompertz"}
        name, _desc = get_model_family(params)
        assert isinstance(name, str)

    def test_model_family_all_known_models(self):
        from sim_ace.plotting.plot_atlas import MODEL_FAMILY

        expected_models = {
            "frailty_weibull",
            "frailty_exponential",
            "frailty_gompertz",
            "frailty_lognormal",
            "frailty_loglogistic",
            "frailty_gamma",
            "cure_frailty",
            "adult_ltm",
            "adult_cox",
            "first_passage",
        }
        assert set(MODEL_FAMILY.keys()) == expected_models


# ---------------------------------------------------------------------------
# plot_validation
# ---------------------------------------------------------------------------


class TestPlotValidation:
    def test_plot_variance_components(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_variance_components

        before = plt.get_fignums()
        plot_variance_components(validation_df, tmp_path, ext="png")
        assert (tmp_path / "variance_components.png").exists()
        assert plt.get_fignums() == before

    def test_plot_twin_rate(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_twin_rate

        before = plt.get_fignums()
        plot_twin_rate(validation_df, tmp_path, ext="png")
        assert (tmp_path / "twin_rate.png").exists()
        assert plt.get_fignums() == before

    def test_plot_A_correlations(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_A_correlations

        before = plt.get_fignums()
        plot_A_correlations(validation_df, tmp_path, ext="png")
        assert (tmp_path / "correlations_A.png").exists()
        assert plt.get_fignums() == before

    def test_plot_cross_trait_correlations(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_cross_trait_correlations

        before = plt.get_fignums()
        plot_cross_trait_correlations(validation_df, tmp_path, ext="png")
        assert (tmp_path / "cross_trait_correlations.png").exists()
        assert plt.get_fignums() == before

    def test_plot_heritability_estimates(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_heritability_estimates

        before = plt.get_fignums()
        plot_heritability_estimates(validation_df, tmp_path, ext="png")
        assert (tmp_path / "heritability_estimates.png").exists()
        assert plt.get_fignums() == before

    def test_plot_half_sib_proportions(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_half_sib_proportions

        before = plt.get_fignums()
        plot_half_sib_proportions(validation_df, tmp_path, ext="png")
        assert (tmp_path / "half_sib_proportions.png").exists()
        assert plt.get_fignums() == before

    def test_plot_family_size(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_family_size

        before = plt.get_fignums()
        plot_family_size(validation_df, tmp_path, ext="png")
        assert (tmp_path / "family_size.png").exists()
        assert plt.get_fignums() == before

    def test_plot_summary_bias(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_summary_bias

        before = plt.get_fignums()
        plot_summary_bias(validation_df, tmp_path, ext="png")
        assert (tmp_path / "summary_bias.png").exists()
        assert plt.get_fignums() == before

    def test_plot_runtime(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_runtime

        before = plt.get_fignums()
        plot_runtime(validation_df, tmp_path, ext="png")
        assert (tmp_path / "runtime.png").exists()
        assert plt.get_fignums() == before

    def test_plot_memory(self, validation_df, tmp_path):
        from sim_ace.plotting.plot_validation import plot_memory

        before = plt.get_fignums()
        plot_memory(validation_df, tmp_path, ext="png")
        assert (tmp_path / "memory.png").exists()
        assert plt.get_fignums() == before


# ---------------------------------------------------------------------------
# Shared fixtures for new smoke tests
# ---------------------------------------------------------------------------

PAIR_TYPES = ["MZ", "FS", "MO", "FO", "MHS", "PHS", "1C"]


def _make_pair_data(r=0.5, n=100):
    return {"r": r, "se": 0.05, "n_pairs": n}


def _make_pair_data_with_liab(r=0.5, n=100, liab_r=0.4):
    return {"r": r, "se": 0.05, "n_pairs": n, "liability_r": liab_r}


@pytest.fixture
def simple_ltm_stats():
    """Minimal stats dict for plot_simple_ltm functions."""
    return [
        {
            "prevalence": {
                "trait1": {"overall": 0.10, "generations": [0, 1, 2], "prevalence": [0.09, 0.10, 0.11]},
                "trait2": {"overall": 0.20, "generations": [0, 1, 2], "prevalence": [0.19, 0.20, 0.21]},
            },
            "tetrachoric": {
                "trait1": {pt: _make_pair_data(0.5 - i * 0.05, 100 - i * 10) for i, pt in enumerate(PAIR_TYPES)},
                "trait2": {pt: _make_pair_data(0.5 - i * 0.05, 100 - i * 10) for i, pt in enumerate(PAIR_TYPES)},
            },
            "liability_correlations": {
                "trait1": {pt: 0.6 - i * 0.05 for i, pt in enumerate(PAIR_TYPES)},
                "trait2": {pt: 0.6 - i * 0.05 for i, pt in enumerate(PAIR_TYPES)},
            },
            "joint_affection": {
                "counts": {"both": 10, "trait1_only": 40, "trait2_only": 90, "neither": 360},
                "proportions": {"both": 0.02, "trait1_only": 0.08, "trait2_only": 0.18, "neither": 0.72},
                "n": 500,
                "by_sex": {"female": 0.02, "male": 0.02},
            },
            "cross_trait_tetrachoric": {
                "same_person": {"r": 0.3, "se": 0.04, "n": 500},
                "same_person_by_generation": {
                    "gen0": {"r": 0.28, "se": 0.06, "n": 160},
                    "gen1": {"r": 0.32, "se": 0.05, "n": 170},
                    "gen2": {"r": 0.30, "se": 0.05, "n": 170},
                },
                "cross_person": {pt: _make_pair_data(0.2, 80) for pt in PAIR_TYPES},
            },
            "regression": {
                "trait1": {
                    "slope": -12.0,
                    "intercept": 50.0,
                    "r": -0.4,
                    "r2": 0.16,
                    "stderr": 2.0,
                    "pvalue": 0.001,
                    "n": 50,
                },
                "trait2": {
                    "slope": -10.0,
                    "intercept": 45.0,
                    "r": -0.3,
                    "r2": 0.09,
                    "stderr": 2.5,
                    "pvalue": 0.01,
                    "n": 100,
                },
            },
            "family_size": {
                "mean": 2.3,
                "median": 2.0,
                "q1": 1.0,
                "q3": 3.0,
                "n_families": 200,
                "frac_with_full_sib": 0.7,
                "size_dist": {"1": 0.25, "2": 0.35, "3": 0.25, "4+": 0.15},
                "person_offspring_dist": {"0": 0.4, "1": 0.2, "2": 0.2, "3": 0.1, "4+": 0.1},
                "mates_by_sex": {
                    "female_mean": 1.1,
                    "male_mean": 1.1,
                    "female_1": 0.8,
                    "female_2+": 0.2,
                    "male_1": 0.8,
                    "male_2+": 0.2,
                },
            },
            "tetrachoric_by_sex": {
                sex: {f"trait{t}": {pt: _make_pair_data_with_liab(0.4, 50) for pt in PAIR_TYPES} for t in [1, 2]}
                for sex in ["female", "male"]
            },
            "parent_offspring_corr_by_sex": {
                sex: {
                    f"trait{t}": {
                        f"gen{g}": {
                            "slope": 0.45,
                            "r": 0.4,
                            "r2": 0.16,
                            "intercept": 0.0,
                            "stderr": 0.05,
                            "pvalue": 0.01,
                            "n_pairs": 100,
                        }
                        for g in [1, 2]
                    }
                    for t in [1, 2]
                }
                for sex in ["female", "male"]
            },
        }
    ]


@pytest.fixture
def simple_ltm_samples():
    """Minimal sample DataFrame for plot_simple_ltm functions."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "liability1": rng.normal(size=n),
            "liability2": rng.normal(size=n),
            "A1": rng.normal(0, 0.7, n),
            "A2": rng.normal(0, 0.7, n),
            "C1": rng.normal(0, 0.4, n),
            "C2": rng.normal(0, 0.4, n),
            "E1": rng.normal(0, 0.5, n),
            "E2": rng.normal(0, 0.5, n),
            "affected1": rng.random(n) < 0.1,
            "affected2": rng.random(n) < 0.2,
            "generation": np.repeat([0, 1, 2, 3], 50),
            "sex": rng.binomial(1, 0.5, n),
            "t_observed1": rng.uniform(10, 80, n),
            "t_observed2": rng.uniform(10, 80, n),
        }
    )


@pytest.fixture
def broad_h2_validations():
    """Minimal all_validations list for plot_broad_heritability_by_generation."""
    return [
        {
            "per_generation": {
                f"generation_{g}": {
                    "A1_var": 0.48,
                    "C1_var": 0.19,
                    "E1_var": 0.30,
                    "A2_var": 0.48,
                    "C2_var": 0.19,
                    "E2_var": 0.30,
                }
                for g in [1, 2, 3]
            },
            "parameters": {"A1": 0.5, "C1": 0.2, "A2": 0.5, "C2": 0.2},
        }
    ]


# ---------------------------------------------------------------------------
# plot_simple_ltm smoke tests
# ---------------------------------------------------------------------------


class TestPlotSimpleLtm:
    def test_plot_prevalence_by_generation(self, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_simple_ltm import plot_prevalence_by_generation

        before = plt.get_fignums()
        out = tmp_path / "prev_gen.png"
        plot_prevalence_by_generation(simple_ltm_stats, 0.10, 0.20, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_liability_violin(self, simple_ltm_samples, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_simple_ltm import plot_liability_violin

        before = plt.get_fignums()
        out = tmp_path / "liab_violin.png"
        plot_liability_violin(simple_ltm_samples, simple_ltm_stats, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_liability_violin_by_generation(self, simple_ltm_samples, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_simple_ltm import plot_liability_violin_by_generation

        before = plt.get_fignums()
        out = tmp_path / "liab_violin_gen.png"
        plot_liability_violin_by_generation(
            simple_ltm_samples,
            simple_ltm_stats,
            0.10,
            0.20,
            out,
            scenario="test",
        )
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_tetrachoric(self, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_simple_ltm import plot_tetrachoric

        before = plt.get_fignums()
        out = tmp_path / "tetrachoric.png"
        plot_tetrachoric(simple_ltm_stats, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_joint_affection(self, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_simple_ltm import plot_joint_affection

        before = plt.get_fignums()
        out = tmp_path / "joint_aff.png"
        plot_joint_affection(simple_ltm_stats, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_liability_joint(self, simple_ltm_samples, tmp_path):
        from sim_ace.plotting.plot_simple_ltm import plot_liability_joint

        before = plt.get_fignums()
        out = tmp_path / "liab_joint.png"
        plot_liability_joint(simple_ltm_samples, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before


# ---------------------------------------------------------------------------
# plot_correlations expanded smoke tests
# ---------------------------------------------------------------------------


class TestPlotCorrelationsExpanded:
    def test_plot_tetrachoric_by_sex(self, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_correlations import plot_tetrachoric_by_sex

        before = plt.get_fignums()
        out = tmp_path / "tet_sex.png"
        plot_tetrachoric_by_sex(simple_ltm_stats, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_tetrachoric_by_sex_no_data(self, tmp_path):
        from sim_ace.plotting.plot_correlations import plot_tetrachoric_by_sex

        before = plt.get_fignums()
        out = tmp_path / "tet_sex_empty.png"
        plot_tetrachoric_by_sex([{}], out, scenario="test")
        assert out.exists()
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_heritability_by_sex_generation(self, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_correlations import plot_heritability_by_sex_generation

        before = plt.get_fignums()
        out = tmp_path / "h2_sex_gen.png"
        plot_heritability_by_sex_generation(simple_ltm_stats, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_heritability_by_sex_generation_no_data(self, tmp_path):
        from sim_ace.plotting.plot_correlations import plot_heritability_by_sex_generation

        before = plt.get_fignums()
        out = tmp_path / "h2_sex_gen_empty.png"
        plot_heritability_by_sex_generation([{}], out, scenario="test")
        assert out.exists()
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_broad_heritability_by_generation(self, broad_h2_validations, tmp_path):
        from sim_ace.plotting.plot_correlations import plot_broad_heritability_by_generation

        before = plt.get_fignums()
        out = tmp_path / "broad_h2_gen.png"
        plot_broad_heritability_by_generation(broad_h2_validations, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_broad_heritability_by_generation_no_data(self, tmp_path):
        from sim_ace.plotting.plot_correlations import plot_broad_heritability_by_generation

        before = plt.get_fignums()
        out = tmp_path / "broad_h2_gen_empty.png"
        plot_broad_heritability_by_generation([{}], out, scenario="test")
        assert out.exists()
        plt.close("all")
        assert plt.get_fignums() == before


# ---------------------------------------------------------------------------
# plot_distributions expanded smoke tests
# ---------------------------------------------------------------------------


class TestPlotDistributionsExpanded:
    def test_plot_trait_regression(self, simple_ltm_samples, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_distributions import plot_trait_regression

        # Ensure affected samples have liability columns
        df = simple_ltm_samples.copy()
        before = plt.get_fignums()
        out = tmp_path / "trait_reg.png"
        plot_trait_regression(df, simple_ltm_stats, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_family_structure(self, simple_ltm_stats, tmp_path):
        from sim_ace.plotting.plot_distributions import plot_family_structure

        before = plt.get_fignums()
        out = tmp_path / "family_struct.png"
        plot_family_structure(simple_ltm_stats, out, scenario="test")
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")
        assert plt.get_fignums() == before

    def test_plot_family_structure_no_data(self, tmp_path):
        from sim_ace.plotting.plot_distributions import plot_family_structure

        before = plt.get_fignums()
        out = tmp_path / "family_struct_empty.png"
        plot_family_structure([{}], out, scenario="test")
        assert out.exists()
        plt.close("all")
        assert plt.get_fignums() == before
