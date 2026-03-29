"""Tests for EPIMIGHT plot atlas data loaders and helpers."""

import pandas as pd
import pytest

from fit_ace.epimight.plot_epimight import (
    discover_kinds,
    load_cif,
    load_cohort_sizes,
    load_h2,
    load_meta,
    load_true_params,
    tmax_rows,
)

# ---------------------------------------------------------------------------
# discover_kinds
# ---------------------------------------------------------------------------


class TestDiscoverKinds:
    def test_discovers_from_filenames(self, tmp_path):
        (tmp_path / "h2_d1_PO.tsv").touch()
        (tmp_path / "h2_d1_FS.tsv").touch()
        (tmp_path / "h2_d1_1C.tsv").touch()
        kinds = discover_kinds(tmp_path)
        assert set(kinds) == {"PO", "FS", "1C"}

    def test_skips_meta_files(self, tmp_path):
        (tmp_path / "h2_d1_PO.tsv").touch()
        (tmp_path / "h2_d1_meta_PO.tsv").touch()
        kinds = discover_kinds(tmp_path)
        assert kinds == ["PO"]

    def test_respects_kind_order(self, tmp_path):
        # Create in reverse order
        for kind in ["1C", "1G", "Av", "HS", "FS", "PO"]:
            (tmp_path / f"h2_d1_{kind}.tsv").touch()
        kinds = discover_kinds(tmp_path)
        # Should follow KIND_ORDER: PO, FS, HS, ..., 1C
        assert kinds[0] == "PO"
        assert kinds[1] == "FS"
        assert kinds[2] == "HS"

    def test_empty_dir(self, tmp_path):
        assert discover_kinds(tmp_path) == []


# ---------------------------------------------------------------------------
# tmax_rows
# ---------------------------------------------------------------------------


class TestTmaxRows:
    def test_selects_max_time_per_year(self):
        df = pd.DataFrame({
            "born_at_year": [1980, 1980, 1981, 1981],
            "time": [40, 60, 50, 70],
            "h2": [0.3, 0.4, 0.2, 0.35],
        })
        result = tmax_rows(df)
        assert len(result) == 2
        assert result.iloc[0]["time"] == 60
        assert result.iloc[1]["time"] == 70

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["born_at_year", "time", "h2"])
        result = tmax_rows(df)
        assert result.empty

    def test_single_row(self):
        df = pd.DataFrame({
            "born_at_year": [1980],
            "time": [50],
            "h2": [0.3],
        })
        result = tmax_rows(df)
        assert len(result) == 1
        assert result.iloc[0]["time"] == 50


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------


class TestLoadCif:
    def test_loads_existing(self, tmp_path):
        pd.DataFrame({"time": [1, 2], "cif": [0.01, 0.02]}).to_csv(
            tmp_path / "cif_d1_c1_PO.tsv", sep="\t", index=False
        )
        df = load_cif(tmp_path, "d1", "c1", "PO")
        assert len(df) == 2
        assert "cif" in df.columns

    def test_missing_returns_empty(self, tmp_path):
        df = load_cif(tmp_path, "d1", "c1", "PO")
        assert df.empty


class TestLoadH2:
    def test_loads_existing(self, tmp_path):
        pd.DataFrame({"time": [10, 20], "h2": [0.3, 0.4]}).to_csv(
            tmp_path / "h2_d1_FS.tsv", sep="\t", index=False
        )
        df = load_h2(tmp_path, "d1", "FS")
        assert len(df) == 2

    def test_missing_returns_empty(self, tmp_path):
        df = load_h2(tmp_path, "d1", "FS")
        assert df.empty


class TestLoadMeta:
    def test_reads_from_tsv(self, tmp_path):
        pd.DataFrame([{
            "fixed_meta": 0.42, "fixed_se": 0.03,
            "fixed_l95": 0.36, "fixed_u95": 0.48,
        }]).to_csv(tmp_path / "h2_d1_meta_PO.tsv", sep="\t", index=False)
        result = load_meta(tmp_path, "h2_d1_meta", "PO")
        assert result is not None
        assert result["fixed_meta"] == pytest.approx(0.42)

    def test_missing_returns_none(self, tmp_path):
        result = load_meta(tmp_path, "h2_d1_meta", "PO")
        assert result is None


class TestLoadTrueParams:
    def test_reads_json(self, tmp_path):
        import json
        truth = {"h2_trait1_true": 0.45}
        (tmp_path / "true_parameters.json").write_text(json.dumps(truth))
        result = load_true_params(tmp_path)
        assert result["h2_trait1_true"] == pytest.approx(0.45)

    def test_missing_returns_none(self, tmp_path):
        assert load_true_params(tmp_path) is None


class TestLoadCohortSizes:
    def test_parses_markdown(self, tmp_path):
        md = (
            "| Cohort | Description | N |\n"
            "|--------|-------------|---|\n"
            "| c1 | Base pop | 60000 |\n"
            "| c2 | Exposed | 19096 |\n"
            "| c3 | Trait 2 exp | 37784 |\n"
        )
        (tmp_path / "results_PO.md").write_text(md)
        sizes = load_cohort_sizes(tmp_path, "PO")
        assert sizes["c1"] == 60000
        assert sizes["c2"] == 19096
        assert sizes["c3"] == 37784

    def test_missing_returns_empty(self, tmp_path):
        assert load_cohort_sizes(tmp_path, "PO") == {}
