"""``simace run``: skip, refuse, force, failure, dry run, and params.yaml."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

import simace
import simace.cli.run as run_mod
from simace.cli.inspect import ls_cli, show_cli
from simace.cli.layout import Layout, RepArtifact
from simace.cli.manifest import RepState, write_manifest
from simace.cli.run import cli, expected_manifest, load_scenario, status_on_disk
from simace.cli.stages import REP_OUTPUTS, STAGES, ResolvedRep

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
TINY = {"N": 300, "G_ped": 3, "G_sim": 3, "G_pheno": 2, "replicates": 2, "seed": 100}


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    cfg = tmp_path / "config"
    cfg.mkdir()
    shutil.copy(REPO_CONFIG / "_default.yaml", cfg / "_default.yaml")
    scenarios = {
        "tiny": TINY,
        "tiny_wf": {**TINY, "replicates": 1, "pedigree": {"mating_model": "wright_fisher"}},
        "tiny_am": {**TINY, "replicates": 1, "pedigree": {"assort_matrix": [[0.2, 0.05], [0.05, 0.1]]}},
        "broken": {**TINY, "replicates": 1, "G_pheno": 9},
        "dropped": {**TINY, "use_gene_drop": True},
    }
    (cfg / "t.yaml").write_text(yaml.safe_dump(scenarios))
    return cfg


@pytest.fixture
def roots(tmp_path: Path) -> list[str]:
    return ["--results", str(tmp_path / "results"), "--logs", str(tmp_path / "logs")]


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    return Layout(root=tmp_path / "results", logs=tmp_path / "logs")


def _run(config_dir: Path, roots: list[str], *argv: str) -> int:
    try:
        cli([*argv, "--config-dir", str(config_dir), *roots])
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _rep(config_dir: Path, rep: int, scenario: str = "tiny") -> ResolvedRep:
    params = load_scenario(config_dir, scenario)
    return ResolvedRep(params["folder"], scenario, rep, params)


@pytest.fixture
def no_plots(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(run_mod, "_build_plots", lambda reps, *a: calls.append(reps) or True)
    return calls


@pytest.fixture
def recorded(monkeypatch) -> list[int]:
    calls: list[int] = []

    def fake_recompute(rep, layout, env, console):
        calls.append(rep.rep)
        _finish(layout, rep)
        return True

    monkeypatch.setattr(run_mod, "_recompute", fake_recompute)
    return calls


def _manifest(layout: Layout, rep: ResolvedRep) -> Path:
    return layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST)


def _finish(layout: Layout, rep: ResolvedRep) -> None:
    """Leave ``rep`` on disk as a finished run does: every output, then ``run.yaml``."""
    for artifact in REP_OUTPUTS:
        path = layout.rep(rep.folder, rep.scenario, rep.rep, artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    write_manifest(_manifest(layout, rep), expected_manifest(rep))


def test_unknown_scenario_exits_2_listing_known_names(config_dir, roots, capsys) -> None:
    assert _run(config_dir, roots, "nope") == 2
    assert "tiny_wf" in capsys.readouterr().err


def test_gene_drop_scenario_exits_2(config_dir, roots, capsys) -> None:
    assert _run(config_dir, roots, "dropped") == 2
    assert "scripts/gene_drop" in capsys.readouterr().err


def test_dry_run_prints_every_stage_and_writes_nothing(config_dir, roots, tmp_path, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "--dry-run") == 0
    lines = capsys.readouterr().out.splitlines()
    commands = [line.split(" -m simace ")[1].split()[0] for line in lines if " -m simace " in line]
    assert commands == [s.name for s in STAGES] * 2 + ["plot", "atlas"]
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "logs").exists()


def test_complete_reps_are_skipped_and_plots_rebuilt(config_dir, roots, layout, recorded, no_plots) -> None:
    for r in (1, 2):
        _finish(layout, _rep(config_dir, r))
    assert _run(config_dir, roots, "tiny") == 0
    assert recorded == []
    assert [rep.rep for rep in no_plots[0]] == [1, 2]


def test_stale_rep_is_refused_naming_the_changed_keys(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    stale = _rep(config_dir, 1)
    old = expected_manifest(stale)
    write_manifest(_manifest(layout, stale), type(old)(**{**old.__dict__, "resolved": {**old.resolved, "N": 999}}))
    assert _run(config_dir, roots, "tiny") == 1
    assert "differs in N" in capsys.readouterr().out
    assert recorded == [2]
    assert no_plots == []


def test_second_run_of_a_locked_scenario_refuses(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    with run_mod._scenario_lock(layout.scenario_lock("t", "tiny")):
        assert _run(config_dir, roots, "tiny") == 1
        assert f"(pid {os.getpid()}) is running tiny" in capsys.readouterr().err
        assert _run(config_dir, roots, "tiny", "--dry-run") == 0
        assert _run(config_dir, roots, "tiny_wf") == 0
    assert recorded == [1]
    assert _run(config_dir, roots, "tiny") == 0
    assert recorded == [1, 1, 2]


@pytest.mark.parametrize("reps", [["1", "1"], ["0"], ["3"], ["-1"]])
def test_rep_values_must_be_distinct_and_in_range(config_dir, roots, recorded, reps, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "--rep", *reps) == 2
    assert "--rep values must be distinct and in 1..2" in capsys.readouterr().err
    assert recorded == []


def test_keys_no_stage_reads_do_not_make_a_rep_stale(config_dir, layout) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    changed = ResolvedRep(
        rep.folder,
        rep.scenario,
        1,
        {**rep.params, "blended_diagnosis": {"x": 1}, "replicates": 9, "plot_format": "pdf"},
    )
    assert status_on_disk(changed, layout).state is RepState.COMPLETE
    touched = ResolvedRep(rep.folder, rep.scenario, 1, {**rep.params, "death_rho": 3.0})
    assert status_on_disk(touched, layout).reasons == ("death_rho",)


def test_a_manifest_copied_from_another_rep_is_stale(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    rep1, rep2 = _rep(config_dir, 1), _rep(config_dir, 2)
    _finish(layout, rep1)
    shutil.copytree(layout.rep_dir("t", "tiny", 1), layout.rep_dir("t", "tiny", 2))
    assert status_on_disk(rep2, layout).state is RepState.STALE
    assert status_on_disk(rep2, layout).reasons == ("rep", "seed")
    assert _run(config_dir, roots, "tiny") == 1
    assert "rep2] refused: run.yaml differs in rep, seed" in capsys.readouterr().out
    assert recorded == []


def test_a_rep_with_a_missing_output_is_recomputed(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    for r in (1, 2):
        _finish(layout, _rep(config_dir, r))
    layout.rep("t", "tiny", 2, RepArtifact.TRAIT).unlink()
    assert status_on_disk(_rep(config_dir, 2), layout).state is RepState.INCOMPLETE
    assert status_on_disk(_rep(config_dir, 2), layout).reasons == ("trait.parquet",)
    assert _run(config_dir, roots, "tiny") == 0
    assert "rep2] recompute: trait.parquet missing" in capsys.readouterr().out
    assert recorded == [2]
    assert [rep.rep for rep in no_plots[0]] == [1, 2]


def test_force_recomputes_complete_and_stale_reps(config_dir, roots, layout, recorded, no_plots) -> None:
    _finish(layout, _rep(config_dir, 1))
    assert _run(config_dir, roots, "tiny", "--force") == 0
    assert recorded == [1, 2]


def test_plots_wait_for_every_rep(config_dir, roots, recorded, no_plots) -> None:
    assert _run(config_dir, roots, "tiny", "--rep", "2") == 0
    assert recorded == [2]
    assert no_plots == []


def test_fail_fast_cancels_pending_reps(config_dir, roots, monkeypatch, no_plots) -> None:
    attempted: list[int] = []
    monkeypatch.setattr(run_mod, "_recompute", lambda rep, *a: attempted.append(rep.rep) and False)
    assert _run(config_dir, roots, "tiny", "--fail-fast") == 1
    assert attempted == [1]


def test_blas_is_always_pinned_and_kernels_only_for_concurrent_reps(monkeypatch) -> None:
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    one, many = run_mod._child_env(1), run_mod._child_env(2)
    assert one["OMP_NUM_THREADS"] == many["OMP_NUM_THREADS"] == "1"
    assert "NUMBA_NUM_THREADS" not in one
    assert many["NUMBA_NUM_THREADS"] == many["POLARS_MAX_THREADS"] == "1"


def test_pedigree_graph_gets_every_core_only_for_one_rep_at_a_time(monkeypatch) -> None:
    monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
    assert run_mod._child_env(1)["PEDIGREE_GRAPH_THREADS"] == str(len(os.sched_getaffinity(0)))
    assert run_mod._child_env(2)["PEDIGREE_GRAPH_THREADS"] == "1"
    monkeypatch.setenv("PEDIGREE_GRAPH_THREADS", "3")
    assert run_mod._child_env(1)["PEDIGREE_GRAPH_THREADS"] == "3"
    assert run_mod._child_env(2)["PEDIGREE_GRAPH_THREADS"] == "1"


@pytest.mark.parametrize(
    ("text", "size"),
    [("512M", 512 * 2**20), ("8G", 8 * 2**30), ("8gb", 8 * 2**30), ("1.5G", 3 * 2**29), ("4096", 4096)],
)
def test_parse_size(text, size) -> None:
    assert run_mod.parse_size(text) == size


@pytest.mark.parametrize("text", ["", "lots", "0", "-1G", "8X"])
def test_parse_size_rejects(text) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        run_mod.parse_size(text)


def test_launcher_kills_a_stage_over_the_memory_cap(tmp_path) -> None:
    hog = [sys.executable, "-c", "import time; b = b'x' * (400 * 2**20); time.sleep(60)"]
    log = tmp_path / "hog.log"
    result = run_mod._Launcher(dict(os.environ), max_rss=200 * 2**20).run(hog, log)
    assert (result.exit_code, result.over_memory) == (-9, True)
    assert result.wall_s < 30
    assert "went over --max-memory (200 MB)" in log.read_text()


def test_launcher_leaves_a_stage_under_the_cap_alone(tmp_path) -> None:
    result = run_mod._Launcher(dict(os.environ), max_rss=2**30).run([sys.executable, "-c", "pass"], tmp_path / "ok.log")
    assert (result.exit_code, result.over_memory) == (0, False)


def test_max_memory_fails_the_rep_and_names_the_cap(config_dir, roots, layout, no_plots, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "--rep", "1", "--max-memory", "20M") == 1
    assert "simulate FAILED (exit -9, killed for going over --max-memory)" in capsys.readouterr().out
    rep_dir = layout.rep_dir("t", "tiny", 1)
    assert (rep_dir / "timing.tsv").read_text().splitlines()[-1].endswith("\t-9")
    assert not (rep_dir / "run.yaml").exists()


def test_recompute_runs_every_stage_and_records_it(config_dir, roots, layout, no_plots) -> None:
    rep = _rep(config_dir, 1)
    rep_dir = layout.rep_dir(rep.folder, rep.scenario, 1)
    rep_dir.mkdir(parents=True)
    (rep_dir / "trait.raw.parquet.tmp").write_text("left by a killed stage")

    assert _run(config_dir, roots, "tiny", "--rep", "1") == 0

    timing = (rep_dir / "timing.tsv").read_text().splitlines()
    assert [row.split("\t")[0] for row in timing[1:]] == [s.name for s in STAGES]
    assert all(row.endswith("\t0") for row in timing[1:])
    assert not list(rep_dir.glob("*.tmp"))
    for stage in STAGES:
        assert all(layout.rep(rep.folder, rep.scenario, 1, out).exists() for out in stage.outputs)
        assert layout.log(rep.folder, rep.scenario, 1, stage.name).exists()
    manifest = yaml.safe_load((rep_dir / "run.yaml").read_text())
    assert manifest["stages"] == [s.name for s in STAGES]
    assert manifest["seed"] == 100
    params = yaml.safe_load((rep_dir / "params.yaml").read_text())
    assert (params["seed"], params["rep"], params["G_pheno"]) == (100, 1, 2)


def test_failing_stage_leaves_no_manifest(config_dir, roots, layout, no_plots, capsys) -> None:
    rep = _rep(config_dir, 1, "broken")
    rep_dir = layout.rep_dir(rep.folder, "broken", 1)
    rep_dir.mkdir(parents=True)
    old_outputs = ["pedigree.parquet", "trait.parquet", "report.yaml", "trait.full.parquet", "run.yaml"]
    for name in old_outputs:
        (rep_dir / name).write_text("from the previous parameters")

    assert _run(config_dir, roots, "broken", "--force") == 1

    assert [name for name in old_outputs if (rep_dir / name).exists()] == []
    assert (rep_dir / "pedigree.full.parquet").exists()
    last = (rep_dir / "timing.tsv").read_text().splitlines()[-1].split("\t")
    assert (last[0], last[-1]) == ("phenotype", "1")
    assert not (rep_dir / "run.yaml").exists()
    assert "phenotype FAILED" in capsys.readouterr().out
    assert "G_pheno" in layout.log(rep.folder, "broken", 1, "phenotype").read_text()


def test_params_yaml_seed_offsets_by_rep(config_dir, layout) -> None:
    run_mod._write_params(_rep(config_dir, 3), layout)
    params = yaml.safe_load(layout.rep("t", "tiny", 3, RepArtifact.PARAMS).read_text())
    assert (params["rep"], params["seed"]) == (3, 102)
    assert "assort_matrix" not in params
    assert isinstance(params["simace_version"], str)


def test_params_yaml_echoes_wright_fisher_knobs_as_set(config_dir, layout) -> None:
    rep = _rep(config_dir, 1, "tiny_wf")
    run_mod._write_params(rep, layout)
    params = yaml.safe_load(layout.rep("t", "tiny_wf", 1, RepArtifact.PARAMS).read_text())
    assert params["mating_model"] == "wright_fisher"
    assert params["mating_lambda"] == rep.params["mating_lambda"]


def test_params_yaml_carries_assort_matrix(config_dir, layout) -> None:
    run_mod._write_params(_rep(config_dir, 1, "tiny_am"), layout)
    params = yaml.safe_load(layout.rep("t", "tiny_am", 1, RepArtifact.PARAMS).read_text())
    assert params["assort_matrix"] == [[0.2, 0.05], [0.05, 0.1]]


def test_show_prints_resolved_params_and_rep_seeds(config_dir, capsys) -> None:
    show_cli(["tiny", "--config-dir", str(config_dir)])
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["params"]["N"] == 300
    assert {name: rep["seed"] for name, rep in shown["reps"].items()} == {"rep1": 100, "rep2": 101}


def test_ls_reports_each_rep_state(config_dir, layout, tmp_path, capsys) -> None:
    _finish(layout, _rep(config_dir, 1))
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    lines = dict(line.split("  ", 1) for line in capsys.readouterr().out.splitlines())
    assert lines["t/tiny"] == "rep1 complete  rep2 absent"
    assert lines["t/dropped"] == "gene drop (not run by simace run)"


def test_ls_flags_reps_built_by_another_version(config_dir, layout, tmp_path, capsys) -> None:
    for r in (1, 2):
        _finish(layout, _rep(config_dir, r))
    old = _manifest(layout, _rep(config_dir, 2))
    old.write_text(old.read_text().replace(f"simace_version: {simace.__version__}", "simace_version: 2020.1.0"))
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    lines = dict(line.split("  ", 1) for line in capsys.readouterr().out.splitlines())
    assert lines["t/tiny"] == "rep1 complete  rep2 complete (built by simace 2020.1.0)"


@pytest.mark.slow
def test_end_to_end_run_builds_plots_and_atlas(config_dir, roots, layout) -> None:
    assert _run(config_dir, roots, "tiny_wf") == 0
    plots = layout.scenario_plots("t", "tiny_wf")
    assert (plots / "atlas.html").exists()
    assert [row.split("\t")[0] for row in (plots / "timing.tsv").read_text().splitlines()[1:]] == ["plot", "atlas"]
    assert _run(config_dir, roots, "tiny_wf") == 0
