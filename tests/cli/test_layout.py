"""Layout pins the path contract fitACE reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from simace.cli.inspect import ls_cli, show_cli
from simace.cli.layout import Layout, RepArtifact
from simace.cli.run import cli as run_cli
from simace.cli.stages import REP_LAYOUT, RETIRED_OUTPUTS
from simace.core.cohort import LAYOUT

if TYPE_CHECKING:
    from pathlib import Path

F, S = "fold", "scen"


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        (RepArtifact.PEDIGREE, "results/fold/scen/rep3/pedigree.parquet"),
        (RepArtifact.COHORT, "results/fold/scen/rep3/cohort.parquet"),
        (RepArtifact.REPORT, "results/fold/scen/rep3/report.yaml"),
        (RepArtifact.PARAMS, "results/fold/scen/rep3/params.yaml"),
    ],
)
def test_fitace_contract_paths_are_frozen(artifact: RepArtifact, expected: str) -> None:
    assert Layout().rep(F, S, 3, artifact).as_posix() == expected


def test_retired_basenames_are_no_artifact() -> None:
    assert set(RETIRED_OUTPUTS).isdisjoint(RepArtifact)


def test_rep_layout_mirrors_the_parquet_marker() -> None:
    assert REP_LAYOUT == LAYOUT


def test_scenario_and_folder_paths() -> None:
    layout = Layout()
    assert layout.scenario_plots(F, S).as_posix() == "results/fold/scen/plots"
    assert layout.scenario_lock(F, S).as_posix() == "results/fold/scen/.run.lock"
    assert layout.folder_summary(F).as_posix() == "results/fold/report_summary.tsv"
    assert layout.folder_plots(F).as_posix() == "results/fold/plots"
    assert layout.scenario_log(F, S, "plot").as_posix() == "logs/fold/scen/plot.log"


def test_layout_honours_custom_roots(tmp_path: Path) -> None:
    layout = Layout(root=tmp_path / "r", logs=tmp_path / "l")
    assert layout.rep(F, S, 2, RepArtifact.TIMING) == tmp_path / "r" / F / S / "rep2" / "timing.tsv"
    assert layout.log(F, S, 2, "censor") == tmp_path / "l" / F / S / "rep2" / "censor.log"


@pytest.mark.parametrize(
    ("command", "argv"),
    [("run", ["small_test", "--dry-run"]), ("show", ["small_test"]), ("ls", [])],
)
def test_config_commands_outside_a_checkout_exit_2_with_a_hint(
    command: str, argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = {"run": run_cli, "show": show_cli, "ls": ls_cli}[command]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        entry(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith(f"simace {command}: no scenario config at config/_default.yaml")
    assert "simACE checkout" in err
