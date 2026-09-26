"""``simace gather <folder>`` summarizes every rep report on disk, then plots."""

from __future__ import annotations

import polars as pl
import pytest
import yaml

import simace.plotting.plot_validation as plot_validation_mod
from simace.cli.gather import cli
from tests.analysis.test_gather_cli import _MINIMAL_REPORT


@pytest.fixture
def plotted(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(plot_validation_mod, "main", lambda *a, **k: calls.append((a, k)))
    return calls


def test_refuses_a_folder_without_reports(tmp_path, plotted, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli(["empty", "--results", str(tmp_path)])
    assert exc.value.code == 1
    assert "no finished reps" in capsys.readouterr().err
    assert plotted == []


def test_gathers_every_finished_rep_with_its_timing_then_plots(tmp_path, plotted, capsys) -> None:
    for scenario, rep in [("scA", 1), ("scA", 2), ("scB", 1), ("scB", 2)]:
        rep_dir = tmp_path / "fold" / scenario / f"rep{rep}"
        rep_dir.mkdir(parents=True)
        (rep_dir / "report.yaml").write_text(yaml.dump(_MINIMAL_REPORT))
        (rep_dir / "timing.tsv").write_text(f"stage\twall_s\tmax_rss_mb\texit_code\nsimulate\t{rep}.5\t100\t0\n")
        if (scenario, rep) != ("scB", 2):
            (rep_dir / "run.yaml").write_text("finished: now\n")

    cli(["fold", "--results", str(tmp_path), "--format", "pdf"])

    assert "skipping" in capsys.readouterr().err

    summary = pl.read_csv(tmp_path / "fold" / "report_summary.tsv", separator="\t")
    assert summary.select("scenario", "rep", "simulate_seconds").rows() == [
        ("scA", 1, 1.5),
        ("scA", 2, 2.5),
        ("scB", 1, 1.5),
    ]
    (args, kwargs) = plotted[0]
    assert args[1] == tmp_path / "fold" / "plots"
    assert kwargs["atlas_name"] == "atlas.pdf"
