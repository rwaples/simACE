"""``simace gather <folder>`` summarizes the configured folder's complete reps (or, with ``--all``, the disk), then plots."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
import yaml

import simace.plotting.plot_validation as plot_validation_mod
from simace.cli.gather import cli
from simace.cli.layout import Layout
from simace.cli.manifest import write_manifest
from simace.cli.run import expected_manifest, load_scenario, rep_outputs
from simace.cli.stages import REP_OUTPUTS, ResolvedRep
from tests.analysis.test_gather_cli import _MINIMAL_REPORT

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


@pytest.fixture
def plotted(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(plot_validation_mod, "main", lambda *a, **k: calls.append((a, k)))
    return calls


def _no_config(tmp_path: Path) -> list[str]:
    """Roots for a results tree whose config lists no scenarios."""
    (tmp_path / "no-config").mkdir(exist_ok=True)
    shutil.copy(REPO_CONFIG / "_default.yaml", tmp_path / "no-config" / "_default.yaml")
    return ["--results", str(tmp_path), "--config-dir", str(tmp_path / "no-config")]


def test_refuses_a_folder_without_reports(tmp_path, plotted, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli(["empty", "--all", *_no_config(tmp_path)])
    assert exc.value.code == 1
    assert "no complete reps" in capsys.readouterr().err
    assert plotted == []


def test_gathers_every_finished_rep_with_its_timing_then_plots(tmp_path, plotted, capsys) -> None:
    for scenario, rep in [("scA", 1), ("scA", 2), ("scB", 1), ("scB", 2)]:
        rep_dir = tmp_path / "fold" / scenario / f"rep{rep}"
        rep_dir.mkdir(parents=True)
        (rep_dir / "report.yaml").write_text(yaml.dump(_MINIMAL_REPORT))
        (rep_dir / "timing.tsv").write_text(f"stage\twall_s\tmax_rss_mb\texit_code\nsimulate\t{rep}.5\t100\t0\n")
        if (scenario, rep) != ("scB", 2):
            (rep_dir / "run.yaml").write_text("finished: now\n")

    cli(["fold", "--all", *_no_config(tmp_path), "--format", "pdf"])

    assert "skipping" in capsys.readouterr().err

    summary = pl.read_csv(tmp_path / "fold" / "report_summary.tsv", separator="\t")
    assert summary.select("scenario", "rep", "simulate_seconds").rows() == [
        ("scA", 1, 1.5),
        ("scA", 2, 2.5),
        ("scB", 1, 1.5),
    ]
    (args, kwargs) = plotted[0]
    assert args[1] == tmp_path / "fold" / "plots"
    assert kwargs["atlas_names"] == ("atlas.html", "atlas.pdf")


def _tiny_folder(tmp_path: Path) -> tuple[Path, Layout]:
    """A configured ``fold/tiny`` with rep1 complete and rep2 stale, plus reps and scenarios the config does not list."""
    config = tmp_path / "config"
    config.mkdir()
    shutil.copy(REPO_CONFIG / "_default.yaml", config / "_default.yaml")
    tiny = {"N": 300, "G_ped": 3, "G_sim": 3, "G_pheno": 2, "replicates": 2, "seed": 100}
    (config / "fold.yaml").write_text(yaml.safe_dump({"tiny": tiny}))
    layout = Layout(root=tmp_path / "results")
    params = load_scenario(config, "tiny")
    for r in (1, 2):
        rep = ResolvedRep("fold", "tiny", r, params)
        for artifact in REP_OUTPUTS:
            path = layout.rep("fold", "tiny", r, artifact)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        layout.rep("fold", "tiny", r, "report.yaml").write_text(yaml.dump(_MINIMAL_REPORT))
        layout.rep("fold", "tiny", r, "timing.tsv").write_text("stage\twall_s\tmax_rss_mb\texit_code\n")
        manifest = expected_manifest(rep)
        if r == 2:
            manifest = type(manifest)(**{**manifest.__dict__, "resolved": {**manifest.resolved, "N": 999}})
        write_manifest(layout.rep("fold", "tiny", r, "run.yaml"), manifest, rep_outputs(rep, layout))
    # A scenario the config no longer lists is kept as long as it finished.
    gone = tmp_path / "results" / "fold" / "gone" / "rep1"
    gone.mkdir(parents=True)
    (gone / "report.yaml").write_text(yaml.dump(_MINIMAL_REPORT))
    (gone / "run.yaml").write_text("finished: now\n")
    # A rep beyond the current replicates is still checked against the config.
    shutil.copytree(layout.rep_dir("fold", "tiny", 2), layout.rep_dir("fold", "tiny", 3))
    # A hand-made directory under a rep* name is not a rep.
    backup = tmp_path / "results" / "fold" / "tiny" / "rep_old"
    backup.mkdir()
    (backup / "report.yaml").write_text(yaml.dump(_MINIMAL_REPORT))
    return config, layout


def test_gathers_the_configured_folder_naming_every_other_rep(tmp_path, plotted, capsys) -> None:
    config, layout = _tiny_folder(tmp_path)
    (config / "fold.yaml").write_text(
        yaml.safe_dump({"tiny": yaml.safe_load((config / "fold.yaml").read_text())["tiny"], "never": {"replicates": 1}})
    )

    cli(["fold", "--results", str(tmp_path / "results"), "--config-dir", str(config)])

    err = capsys.readouterr().err
    assert f"skipping {layout.rep_dir('fold', 'tiny', 2)} (stale: N: 999 -> 300)" in err
    assert "fold/tiny: 1 of 2 reps" in err
    assert f"skipping {layout.rep_dir('fold', 'never', 1)} (absent)" in err
    assert "fold/never: 0 of 1 reps" in err
    assert f"skipping {layout.rep_dir('fold', 'tiny', 3)} (beyond replicates: 2; --all includes it)" in err
    assert f"skipping {tmp_path / 'results' / 'fold' / 'gone'} (not in config/fold.yaml; --all includes it)" in err
    assert "rep_old" not in err
    summary = pl.read_csv(tmp_path / "results" / "fold" / "report_summary.tsv", separator="\t")
    assert summary.select("scenario", "rep").rows() == [("tiny", 1)]


def test_all_gathers_every_finished_rep_on_disk(tmp_path, plotted, capsys) -> None:
    config, layout = _tiny_folder(tmp_path)

    cli(["fold", "--all", "--results", str(tmp_path / "results"), "--config-dir", str(config)])

    err = capsys.readouterr().err
    assert f"skipping {layout.rep_dir('fold', 'tiny', 2)} (stale: N: 999 -> 300)" in err
    assert f"skipping {layout.rep_dir('fold', 'tiny', 3)} (stale: rep: 2 -> 3, seed: 101 -> 102, N: 999 -> 300)" in err
    assert f"skipping {tmp_path / 'results' / 'fold' / 'tiny' / 'rep_old'} (not a rep directory)" in err
    summary = pl.read_csv(tmp_path / "results" / "fold" / "report_summary.tsv", separator="\t")
    assert summary.select("scenario", "rep").rows() == [("gone", 1), ("tiny", 1)]


def test_a_folder_the_config_does_not_list_needs_all(tmp_path, plotted, capsys) -> None:
    config, _ = _tiny_folder(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli(["archive", "--results", str(tmp_path / "results"), "--config-dir", str(config)])
    assert exc.value.code == 2
    assert "no configured scenario has folder 'archive'; --all" in capsys.readouterr().err
    assert plotted == []


def test_explicit_missing_config_dir_is_an_error(tmp_path, plotted, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli(["fold", "--results", str(tmp_path), "--config-dir", str(tmp_path / "confg")])
    assert exc.value.code == 2
    assert "no config directory at" in capsys.readouterr().err
    assert plotted == []
