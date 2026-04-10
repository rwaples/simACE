"""Tests for EPIMIGHT bias analysis functions."""

import json

import numpy as np
import pytest

from fit_ace.epimight.epimight_bias_analysis import (
    compute_bias_metrics,
    load_epimight_meta,
    load_true_h2,
    parse_scenario_name,
)

# ---------------------------------------------------------------------------
# parse_scenario_name
# ---------------------------------------------------------------------------


class TestParseScenarioName:
    def test_ltm_scenario(self):
        r = parse_scenario_name("ebias_ltm_K10_C0_nocensor")
        assert r["phenotype_model"] == "adult"
        assert r["model_label"] == "adult_ltm"
        assert r["prevalence"] == pytest.approx(0.10)
        assert r["C"] == 0.0
        assert r["censor_label"] == "none"
        assert r["has_death_censor"] is False
        assert r["has_window_censor"] is False

    def test_weibull_scenario(self):
        r = parse_scenario_name("ebias_weibull_K05_C02_death")
        assert r["phenotype_model"] == "frailty"
        assert r["model_label"] == "frailty_weibull"
        assert r["prevalence"] == pytest.approx(0.05)
        assert r["C"] == 0.2
        assert r["censor_label"] == "death_only"
        assert r["has_death_censor"] is True

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("K01", 0.01),
            ("K05", 0.05),
            ("K10", 0.10),
            ("K20", 0.20),
            ("K40", 0.40),
        ],
    )
    def test_all_prevalences(self, key, expected):
        r = parse_scenario_name(f"ebias_ltm_{key}_C0_nocensor")
        assert r["prevalence"] == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("nocensor", "none"),
            ("death", "death_only"),
            ("window", "window_only"),
            ("both", "both"),
        ],
    )
    def test_all_censor_labels(self, key, expected):
        r = parse_scenario_name(f"ebias_ltm_K10_C0_{key}")
        assert r["censor_label"] == expected

    def test_c02_flag(self):
        r = parse_scenario_name("ebias_ltm_K10_C02_nocensor")
        assert r["C"] == 0.2

    def test_both_censoring_flags(self):
        r = parse_scenario_name("ebias_ltm_K10_C0_both")
        assert r["has_death_censor"] is True
        assert r["has_window_censor"] is True

    def test_unknown_model_passthrough(self):
        r = parse_scenario_name("ebias_newmodel_K10_C0_nocensor")
        assert r["phenotype_model"] == "newmodel"


# ---------------------------------------------------------------------------
# compute_bias_metrics
# ---------------------------------------------------------------------------


class TestComputeBiasMetrics:
    def test_zero_bias(self):
        m = compute_bias_metrics(0.50, 0.05, 0.40, 0.60, 0.50, 0.50)
        assert m["abs_bias_liability"] == pytest.approx(0.0)
        assert m["rel_bias_liability"] == pytest.approx(0.0)
        assert m["attenuation_ratio"] == pytest.approx(1.0)

    def test_downward_bias(self):
        m = compute_bias_metrics(0.30, 0.05, 0.20, 0.40, 0.50, None)
        assert m["abs_bias_liability"] == pytest.approx(-0.20)
        assert m["rel_bias_liability"] == pytest.approx(-0.40)
        assert m["attenuation_ratio"] == pytest.approx(0.60)

    def test_ci_covers_true(self):
        m = compute_bias_metrics(0.50, 0.05, 0.40, 0.60, 0.50, None)
        assert m["ci_covers_liability"] == 1.0

    def test_ci_misses_true(self):
        m = compute_bias_metrics(0.30, 0.02, 0.26, 0.34, 0.50, None)
        assert m["ci_covers_liability"] == 0.0

    def test_ltm_comparison(self):
        m = compute_bias_metrics(0.30, 0.05, 0.20, 0.40, 0.50, 0.45)
        assert m["abs_bias_ltm"] == pytest.approx(-0.15)
        assert m["rel_bias_ltm"] == pytest.approx(-0.15 / 0.45)
        # CI [0.20, 0.40] does not cover 0.45
        assert m["ci_covers_ltm"] == 0.0

    def test_ltm_none(self):
        m = compute_bias_metrics(0.30, 0.05, 0.20, 0.40, 0.50, None)
        assert np.isnan(m["abs_bias_ltm"])
        assert np.isnan(m["rel_bias_ltm"])
        assert np.isnan(m["ci_covers_ltm"])

    def test_h2_true_zero(self):
        m = compute_bias_metrics(0.30, 0.05, 0.20, 0.40, 0.0, None)
        assert np.isnan(m["rel_bias_liability"])
        assert np.isnan(m["attenuation_ratio"])


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


class TestLoadTrueH2:
    def test_reads_correct_trait(self, tmp_path):
        truth = {"h2_trait1_true": 0.45, "h2_trait2_true": 0.30}
        (tmp_path / "true_parameters.json").write_text(json.dumps(truth))
        assert load_true_h2(tmp_path, trait=1) == pytest.approx(0.45)
        assert load_true_h2(tmp_path, trait=2) == pytest.approx(0.30)

    def test_missing_file_returns_none(self, tmp_path):
        assert load_true_h2(tmp_path) is None


class TestLoadEpimightMeta:
    def test_reads_first_row(self, tmp_path):
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "fixed_meta": 0.42,
                    "fixed_se": 0.03,
                    "fixed_l95": 0.36,
                    "fixed_u95": 0.48,
                }
            ]
        ).to_csv(tsv_dir / "h2_d1_meta_PO.tsv", sep="\t", index=False)
        result = load_epimight_meta(tmp_path, "PO")
        assert result["fixed_meta"] == pytest.approx(0.42)

    def test_missing_file_returns_none(self, tmp_path):
        assert load_epimight_meta(tmp_path, "PO") is None

    def test_empty_data_returns_none(self, tmp_path):
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()
        import pandas as pd

        # File with headers but no data rows
        pd.DataFrame(columns=["fixed_meta", "fixed_se"]).to_csv(tsv_dir / "h2_d1_meta_PO.tsv", sep="\t", index=False)
        assert load_epimight_meta(tmp_path, "PO") is None
