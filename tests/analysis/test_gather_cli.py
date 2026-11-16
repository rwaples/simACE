"""Tests for gather.py and cli_base.py."""

import argparse
import logging

import pytest
import yaml

from simace.analysis.quality import FamilyResult, PairCorrelation, QualityCheck, QualityReport
from simace.analysis.quality.assortative_mating import AssortativeMatingMeasurements
from simace.analysis.quality.consanguinity import ConsanguinityMeasurements
from simace.analysis.quality.half_sibs import HalfSibMeasurements, HalfSibTrait
from simace.analysis.quality.heritability import HeritabilityMeasurements, HeritabilityTrait, MidparentRegression
from simace.analysis.quality.population import PopulationMeasurements
from simace.analysis.quality.statistical import FounderVariances, StatisticalMeasurements
from simace.analysis.quality.structural import StructuralMeasurements
from simace.analysis.quality.twins import TwinMeasurements
from simace.analysis.report import assemble_report
from simace.analysis.sample_stats import AnalysisPedigreeStats, AnalysisSampleStats, SampleStats, ScopeSize
from tests.conftest import stats_tables

# ---------------------------------------------------------------------------
# cli_base
# ---------------------------------------------------------------------------


class TestCliBase:
    def test_add_logging_args_adds_verbose_and_quiet(self):
        from simace.core.cli_base import add_logging_args

        parser = argparse.ArgumentParser()
        add_logging_args(parser)
        # Verify -v and -q are registered
        args = parser.parse_args([])
        assert hasattr(args, "verbose")
        assert hasattr(args, "quiet")
        assert args.verbose is False
        assert args.quiet is False

    def test_add_logging_args_verbose(self):
        from simace.core.cli_base import add_logging_args

        parser = argparse.ArgumentParser()
        add_logging_args(parser)
        args = parser.parse_args(["-v"])
        assert args.verbose is True
        assert args.quiet is False

    def test_add_logging_args_quiet(self):
        from simace.core.cli_base import add_logging_args

        parser = argparse.ArgumentParser()
        add_logging_args(parser)
        args = parser.parse_args(["-q"])
        assert args.quiet is True
        assert args.verbose is False

    def test_init_logging_verbose(self):
        from simace.core.cli_base import init_logging

        args = argparse.Namespace(verbose=True, quiet=False)
        init_logging(args)
        pkg = logging.getLogger("simace")
        assert pkg.level == logging.DEBUG

    def test_init_logging_quiet(self):
        from simace.core.cli_base import init_logging

        args = argparse.Namespace(verbose=False, quiet=True)
        init_logging(args)
        pkg = logging.getLogger("simace")
        assert pkg.level == logging.WARNING

    def test_init_logging_default(self):
        from simace.core.cli_base import init_logging

        args = argparse.Namespace(verbose=False, quiet=False)
        init_logging(args)
        pkg = logging.getLogger("simace")
        assert pkg.level == logging.INFO


# ---------------------------------------------------------------------------
# gather — extract_metrics
# ---------------------------------------------------------------------------

_MINIMAL_PARAMS = {
    "N": 1000,
    "G_ped": 3,
    "G_sim": 4,
    "A1": 0.5,
    "C1": 0.2,
    "E1": 0.3,
    "A2": 0.5,
    "C2": 0.2,
    "E2": 0.3,
    "rA": 0.3,
    "rC": 0.5,
    "p_mztwin": 0.02,
    "mating_lambda": 0.5,
    "assort1": 0.0,
    "assort2": 0.0,
    "seed": 42,
}

_HALF_SIB_TRAIT = HalfSibTrait(
    A_corr=PairCorrelation(0.25, 100), liability_corr=PairCorrelation(0.25, 100), shared_C=PairCorrelation(0.0, 100)
)
_HERITABILITY_TRAIT = HeritabilityTrait(
    mz_A_corr=PairCorrelation(0.99, 20),
    mz_liability_corr=PairCorrelation(0.70, 20),
    dz_A_corr=PairCorrelation(0.49, 300),
    dz_liability_corr=PairCorrelation(0.50, 300),
    falconer=0.50,
    po_A=MidparentRegression(slope=0.50, r_squared=0.50),
    po_liability=MidparentRegression(slope=0.50, r_squared=0.50),
)

# A quality report with every measurement extract_metrics reads.
_MINIMAL_QUALITY = QualityReport(
    structural=FamilyResult(StructuralMeasurements(sex_ratio_male=0.5), ()),
    twins=FamilyResult(TwinMeasurements(twin_rate=0.019, expected_twin_rate=0.0392), ()),
    statistical=FamilyResult(
        StatisticalMeasurements(
            traits=(FounderVariances(0.48, 0.21, 0.31, 1.0), FounderVariances(0.49, 0.19, 0.32, 1.0)),
            rA=0.29,
            rC=0.51,
            rE=0.01,
        ),
        (),
    ),
    population=FamilyResult(PopulationMeasurements(generation_sizes=[1000, 1000, 1000], n_generations=3), ()),
    consanguineous_matings=FamilyResult(
        ConsanguinityMeasurements(n_half_sib_matings=0, n_full_sib_matings=0, total_missing_gp_links=0),
        (QualityCheck("grandparent_reconciliation", True, "reconciled"),),
    ),
    half_sibs=FamilyResult(
        HalfSibMeasurements(
            pair_proportion=0.11,
            offspring_with_half_sib=0.14,
            traits=(_HALF_SIB_TRAIT, _HALF_SIB_TRAIT),
        ),
        (),
    ),
    c_transmission=FamilyResult(None, ()),
    heritability=FamilyResult(HeritabilityMeasurements(traits=(_HERITABILITY_TRAIT, _HERITABILITY_TRAIT)), ()),
    assortative_mating=FamilyResult(
        AssortativeMatingMeasurements(liability=(0.01, -0.01), genetic=(0.0, 0.0)),
        (),
    ),
    am_equilibrium=FamilyResult(None, ()),
    per_generation={},
    family_size_distribution={"mother": {"mean": 2.3}, "father": {"mean": 2.3}},
)

# extract_metrics reads a curated v2 report (ADR 0008). Funnel the fixture
# through the real builder so registry paths line up.
_MINIMAL_REPORT, _ = assemble_report(
    replicate={"folder": "base", "scenario": "minimal", "rep": 1, "seed": 42},
    params=_MINIMAL_PARAMS,
    case_ascertainment_ratio=1.0,
    quality=_MINIMAL_QUALITY,
    stats=SampleStats(
        sample_size=ScopeSize(n_individuals=0, n_generations=0),
        pedigree_size=ScopeSize(n_individuals=0, n_generations=0),
        analysis_sample=stats_tables(AnalysisSampleStats, {}),
        analysis_pedigree=stats_tables(AnalysisPedigreeStats, {}),
        observed_h2_estimators={},
    ),
    prevalence_phenotyped={},
    scope_counts={},
)


class TestExtractMetrics:
    def test_extracts_scenario_and_rep_from_path(self, tmp_path):
        from simace.analysis.gather import extract_metrics

        # Create path that matches the expected pattern
        val_dir = tmp_path / "results" / "base" / "my_scenario" / "rep2"
        val_dir.mkdir(parents=True)
        val_path = val_dir / "report.yaml"
        val_path.write_text(yaml.dump(_MINIMAL_REPORT))

        row = extract_metrics(str(val_path))
        assert row["scenario"] == "my_scenario"
        assert row["rep"] == 2

    def test_extracts_parameter_values(self, tmp_path):
        from simace.analysis.gather import extract_metrics

        val_dir = tmp_path / "results" / "base" / "scA" / "rep1"
        val_dir.mkdir(parents=True)
        val_path = val_dir / "report.yaml"
        val_path.write_text(yaml.dump(_MINIMAL_REPORT))

        row = extract_metrics(str(val_path))
        assert row["N"] == 1000
        assert row["A1"] == 0.5
        assert row["checks_failed"] == 0
        assert row["observed_twin_rate"] == pytest.approx(0.019)
        assert row["variance_A1"] == pytest.approx(0.48)


class TestGatherMain:
    def test_sorted_by_scenario_and_rep(self, tmp_path):
        from simace.analysis.gather import main

        for sc, rep in [("scB", 2), ("scA", 1), ("scB", 1)]:
            val_dir = tmp_path / "results" / "base" / sc / f"rep{rep}"
            val_dir.mkdir(parents=True)
            val_path = val_dir / "report.yaml"
            val_path.write_text(yaml.dump(_MINIMAL_REPORT))

        out_path = tmp_path / "summary.tsv"
        files = [
            str(tmp_path / "results" / "base" / "scB" / "rep2" / "report.yaml"),
            str(tmp_path / "results" / "base" / "scA" / "rep1" / "report.yaml"),
            str(tmp_path / "results" / "base" / "scB" / "rep1" / "report.yaml"),
        ]
        main(files, str(out_path))

        lines = out_path.read_text().strip().split("\n")
        header = lines[0].split("\t")
        sc_idx = header.index("scenario")
        scenarios = [row.split("\t")[sc_idx] for row in lines[1:]]
        assert scenarios == ["scA", "scB", "scB"]

    def test_empty_input_no_crash(self, tmp_path):
        from simace.analysis.gather import main

        out_path = tmp_path / "empty.tsv"
        main([], str(out_path))
        # No file written for empty input
        assert not out_path.exists()

    def test_nan_is_written_as_the_empty_field(self, tmp_path, monkeypatch):
        """A NaN metric must not leave a literal ``nan`` token in the TSV.

        polars infers any column holding ``nan`` as ``String`` (pandas read it
        as missing), which silently turns a numeric metric into text and breaks
        every downstream reader. ±inf still round-trips as a float.
        """
        import polars as pl

        from simace.analysis import gather

        rows = [
            {"scenario": "scA", "rep": 1, "metric": float("nan"), "big": float("inf"), "ok": 0.5},
            {"scenario": "scA", "rep": 2, "metric": 0.25, "big": float("-inf"), "ok": 0.75},
        ]
        monkeypatch.setattr(gather, "extract_metrics", lambda path: rows.pop(0))

        out_path = tmp_path / "summary.tsv"
        gather.main(["a.yaml", "b.yaml"], str(out_path))

        text = out_path.read_text()
        assert "nan" not in text
        df = pl.read_csv(out_path, separator="\t", null_values=["NA", ""], infer_schema_length=None)
        assert df["metric"].dtype == pl.Float64
        assert df["metric"].to_list() == [None, 0.25]
        assert df["big"].dtype == pl.Float64


class TestExtractMetricsBranches:
    """Branches in ``extract_metrics`` not exercised elsewhere."""

    def test_path_without_results_pattern_defaults_to_unknown(self, tmp_path):
        """A report YAML whose path doesn't match ``results/{folder}/{scenario}/rep{N}``
        falls through to ``scenario='unknown'`` / ``rep=1``."""
        from simace.analysis.gather import extract_metrics

        val_path = tmp_path / "stray_report.yaml"
        val_path.write_text(yaml.dump(_MINIMAL_REPORT))

        row = extract_metrics(str(val_path))
        assert row["scenario"] == "unknown"
        assert row["rep"] == 1
        assert row["simulate_seconds"] is None
        assert row["simulate_max_rss_mb"] is None

    @pytest.mark.parametrize(
        "timing",
        [
            "stage\twall_s\tmax_rss_mb\texit_code\nsimulate\t1.23\t456.7\t0\nphenotype\t9.9\t999.0\t0\n",
            "stage\twall_s\tmax_rss_mb\ttree_peak_mb\texit_code\nsimulate\t1.23\t456.7\t500.1\t0\n",
        ],
        ids=["before-tree-peak", "with-tree-peak"],
    )
    def test_timing_tsv_populates_timing_fields(self, tmp_path, timing):
        """The rep's ``timing.tsv`` simulate row fills ``simulate_seconds`` and ``simulate_max_rss_mb``."""
        from simace.analysis.gather import extract_metrics

        val_dir = tmp_path / "any_root" / "base" / "myscenario" / "rep2"
        val_dir.mkdir(parents=True)
        val_path = val_dir / "report.yaml"
        val_path.write_text(yaml.dump(_MINIMAL_REPORT))
        (val_dir / "timing.tsv").write_text(timing)

        row = extract_metrics(str(val_path))
        assert (row["folder"], row["scenario"], row["rep"]) == ("base", "myscenario", 2)
        assert row["simulate_seconds"] == pytest.approx(1.23)
        assert row["simulate_max_rss_mb"] == pytest.approx(456.7)


class TestGatherCli:
    """End-to-end CLI smoke for ``simace.analysis.gather:cli``."""

    def test_cli_writes_tsv(self, tmp_path):
        from simace.analysis.gather import cli as gather_cli

        # Two scenario YAMLs.
        files = []
        for sc in ("scA", "scB"):
            val_dir = tmp_path / "results" / "base" / sc / "rep1"
            val_dir.mkdir(parents=True)
            val_path = val_dir / "report.yaml"
            val_path.write_text(yaml.dump(_MINIMAL_REPORT))
            files.append(str(val_path))

        out_path = tmp_path / "summary.tsv"
        gather_cli([*files, "--output", str(out_path)])

        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        header = lines[0].split("\t")
        assert {"scenario", "rep", "N", "A1"}.issubset(set(header))
