"""Tests for EPIMIGHT bias analysis plots."""

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fit_ace.constants import KIND_ORDER
from fit_ace.plotting.plot_epimight_bias import (
    CENSOR_ORDER,
    _analytical_dilution,
    compute_dilution_corrected_h2,
    generate_all_plots,
    plot_epimight_attenuation_summary,
    plot_epimight_bias_by_censoring,
    plot_epimight_bias_heatmap,
    plot_epimight_bias_vs_prevalence,
    plot_epimight_c_effect,
    plot_epimight_corrected_h2,
    plot_epimight_dilution_ratio,
    plot_epimight_forest,
    plot_epimight_model_comparison,
)


@pytest.fixture(scope="module")
def bias_df():
    """Minimal bias summary DataFrame covering the columns all plot functions use."""
    rows = [
        {
            "scenario": f"ebias_ltm_K{int(prev * 100):02d}_C{'02' if c_val else '0'}_{censor.replace('_only', '')}",
            "kind": kind,
            "phenotype_model": "adult",
            "model_label": "adult_ltm",
            "prevalence": prev,
            "C": c_val,
            "censor_label": censor,
            "h2_epimight": 0.50 * (1 - prev * 0.5),
            "h2_se": 0.03,
            "h2_l95": 0.50 * (1 - prev * 0.5) - 0.06,
            "h2_u95": 0.50 * (1 - prev * 0.5) + 0.06,
            "h2_true_liability": 0.50,
            "attenuation_ratio": 1 - prev * 0.5,
            "abs_bias_liability": -0.25 * prev,
            "rel_bias_liability": -0.5 * prev,
            "ci_covers_liability": 1.0,
            "h2_ltm_falconer": 0.475,
            "has_death_censor": "death" in censor,
            "has_window_censor": "window" in censor,
        }
        for prev in [0.01, 0.05, 0.10, 0.20, 0.40]
        for kind in KIND_ORDER
        for censor in CENSOR_ORDER
        for c_val in [0.0, 0.2]
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _analytical_dilution
# ---------------------------------------------------------------------------


class TestAnalyticalDilution:
    def test_low_prevalence_near_one(self):
        d = _analytical_dilution(K=0.01, h2=0.5, kinship=0.25, n_rel=2)
        assert d > 0.9

    def test_high_prevalence_lower(self):
        d = _analytical_dilution(K=0.40, h2=0.5, kinship=0.25, n_rel=2)
        assert d < 0.95

    def test_more_relatives_more_dilution(self):
        d_few = _analytical_dilution(K=0.20, h2=0.5, kinship=0.25, n_rel=2)
        d_many = _analytical_dilution(K=0.20, h2=0.5, kinship=0.25, n_rel=13)
        assert d_many < d_few

    def test_zero_kinship_returns_one(self):
        d = _analytical_dilution(K=0.10, h2=0.5, kinship=0.0, n_rel=2)
        assert d == pytest.approx(1.0)

    def test_extreme_low_K_returns_one(self):
        d = _analytical_dilution(K=1e-12, h2=0.5, kinship=0.25, n_rel=2)
        assert d == pytest.approx(1.0)

    def test_extreme_high_K_returns_one(self):
        d = _analytical_dilution(K=1 - 1e-12, h2=0.5, kinship=0.25, n_rel=2)
        assert d == pytest.approx(1.0)

    def test_monotonic_in_prevalence(self):
        prevs = [0.01, 0.05, 0.10, 0.20, 0.40]
        dilutions = [_analytical_dilution(K=k, h2=0.5, kinship=0.25, n_rel=4) for k in prevs]
        # Dilution should decrease (more dilution) with higher prevalence
        for i in range(len(dilutions) - 1):
            assert dilutions[i + 1] <= dilutions[i] + 1e-6

    def test_result_in_valid_range(self):
        d = _analytical_dilution(K=0.10, h2=0.5, kinship=0.25, n_rel=5)
        assert 0.05 <= d <= 1.0


# ---------------------------------------------------------------------------
# compute_dilution_corrected_h2
# ---------------------------------------------------------------------------


class TestComputeDilutionCorrectedH2:
    def test_adds_expected_columns(self):
        df = pd.DataFrame(
            {
                "kind": ["PO", "FS"],
                "prevalence": [0.10, 0.10],
                "h2_epimight": [0.30, 0.25],
            }
        )
        result = compute_dilution_corrected_h2(df)
        assert "dilution_ratio" in result.columns
        assert "h2_corrected" in result.columns

    def test_corrected_at_least_raw(self):
        df = pd.DataFrame(
            {
                "kind": ["PO", "FS", "HS"],
                "prevalence": [0.20, 0.20, 0.20],
                "h2_epimight": [0.30, 0.25, 0.20],
            }
        )
        result = compute_dilution_corrected_h2(df)
        for i in range(len(result)):
            if not np.isnan(result.iloc[i]["h2_corrected"]):
                assert result.iloc[i]["h2_corrected"] >= result.iloc[i]["h2_epimight"] - 0.01

    def test_low_prevalence_minimal_correction(self):
        df = pd.DataFrame(
            {
                "kind": ["PO"],
                "prevalence": [0.01],
                "h2_epimight": [0.45],
            }
        )
        result = compute_dilution_corrected_h2(df)
        # At very low prevalence, correction should be minimal
        assert abs(result.iloc[0]["h2_corrected"] - 0.45) < 0.05

    def test_does_not_modify_original(self):
        df = pd.DataFrame(
            {
                "kind": ["PO"],
                "prevalence": [0.10],
                "h2_epimight": [0.30],
            }
        )
        compute_dilution_corrected_h2(df)
        assert "dilution_ratio" not in df.columns


# ---------------------------------------------------------------------------
# Plot smoke tests
# ---------------------------------------------------------------------------


def _assert_plot_created(func, df, tmp_path, name="test.png"):
    path = tmp_path / name
    before = plt.get_fignums()
    func(df, path)
    assert path.exists(), f"{name} not created"
    assert path.stat().st_size > 0, f"{name} is empty"
    assert plt.get_fignums() == before, f"{name} leaked a figure"


class TestPlotSmoke:
    def test_bias_vs_prevalence(self, bias_df, tmp_path):
        _assert_plot_created(plot_epimight_bias_vs_prevalence, bias_df, tmp_path)

    def test_bias_by_censoring(self, bias_df, tmp_path):
        _assert_plot_created(plot_epimight_bias_by_censoring, bias_df, tmp_path)

    def test_bias_heatmap(self, bias_df, tmp_path):
        _assert_plot_created(plot_epimight_bias_heatmap, bias_df, tmp_path)

    def test_c_effect(self, bias_df, tmp_path):
        _assert_plot_created(plot_epimight_c_effect, bias_df, tmp_path)

    def test_model_comparison(self, bias_df, tmp_path):
        _assert_plot_created(plot_epimight_model_comparison, bias_df, tmp_path)

    def test_forest(self, bias_df, tmp_path):
        _assert_plot_created(plot_epimight_forest, bias_df, tmp_path)

    def test_attenuation_summary(self, bias_df, tmp_path):
        _assert_plot_created(plot_epimight_attenuation_summary, bias_df, tmp_path)

    def test_dilution_ratio(self, bias_df, tmp_path):
        # Filter to the subset the plot function uses to avoid running
        # compute_dilution_corrected_h2 on all 320 rows (~400 quad calls)
        small = bias_df[
            (bias_df["C"] == 0) & (bias_df["phenotype_model"] == "adult") & (bias_df["censor_label"] == "none")
        ].copy()
        small = compute_dilution_corrected_h2(small)
        _assert_plot_created(plot_epimight_dilution_ratio, small, tmp_path)

    def test_corrected_h2(self, bias_df, tmp_path):
        small = bias_df[
            (bias_df["C"] == 0) & (bias_df["phenotype_model"] == "adult") & (bias_df["censor_label"] == "none")
        ].copy()
        small = compute_dilution_corrected_h2(small)
        _assert_plot_created(plot_epimight_corrected_h2, small, tmp_path)


class TestGenerateAllPlots:
    def test_returns_paths(self, bias_df, tmp_path):
        # Skip dilution correction plots — they're tested individually above
        # and each triggers ~400 scipy.quad calls (~3 min)
        paths = generate_all_plots(bias_df, tmp_path, include_dilution_correction=False)
        assert len(paths) > 0
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
