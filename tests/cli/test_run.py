"""``simace run``: skip, refuse, force, failure, dry run, and params.yaml."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

import simace
import simace.cli.launch as launch_mod
import simace.cli.run as run_mod
from simace.cli.cgroups import CgroupRoot, CgroupUnavailable
from simace.cli.inspect import ls_cli, show_cli
from simace.cli.layout import Layout, RepArtifact
from simace.cli.manifest import PlotsState, RepState, write_manifest, write_plots_manifest
from simace.cli.run import cli
from simace.cli.stages import REP_OUTPUTS, RETIRED_OUTPUTS, STAGES, ResolvedRep
from simace.cli.status import (
    expected_manifest,
    load_scenario,
    rep_outputs,
    rep_ranges,
    scenario_plots_status,
    status_on_disk,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
TINY = {"N": 300, "G_ped": 3, "G_sim": 3, "G_pheno": 2, "replicates": 2, "seed": 100}


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    cfg = tmp_path / "config"
    cfg.mkdir()
    shutil.copy(REPO_CONFIG / "_default.yaml", cfg / "_default.yaml")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "simace"\n')
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

    def fake_recompute(rep, layout, env, console, *_):
        calls.append(rep.rep)
        _finish(layout, rep)

    monkeypatch.setattr(run_mod, "_recompute", fake_recompute)
    return calls


def _manifest(layout: Layout, rep: ResolvedRep) -> Path:
    return layout.rep(rep.folder, rep.scenario, rep.rep, RepArtifact.RUN_MANIFEST)


def _finish(layout: Layout, rep: ResolvedRep) -> None:
    """Leave ``rep`` on disk as a finished run does: every output, then ``run.yaml``."""
    for artifact in REP_OUTPUTS:
        path = layout.rep(rep.folder, rep.scenario, rep.rep, artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("")
    _write_manifest(layout, rep, expected_manifest(rep))


def _write_manifest(layout: Layout, rep: ResolvedRep, manifest) -> None:
    write_manifest(_manifest(layout, rep), manifest, rep_outputs(rep, layout))


def _write_manifest_through(layout: Layout, rep: ResolvedRep, n_stages: int) -> None:
    """Rewrite ``rep``'s manifest as ``simace run --until`` leaves it after the first ``n_stages`` stages."""
    stages = STAGES[:n_stages]
    write_manifest(_manifest(layout, rep), expected_manifest(rep, stages), rep_outputs(rep, layout, stages))


def _stale(layout: Layout, rep: ResolvedRep) -> None:
    """Rewrite ``rep``'s manifest as if it had been computed with ``N: 999``."""
    old = expected_manifest(rep)
    _write_manifest(layout, rep, type(old)(**{**old.__dict__, "resolved": {**old.resolved, "N": 999}}))


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


def test_dry_run_pdf_builds_the_pdf_atlas_beside_the_html(config_dir, roots, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "--dry-run", "--format", "pdf") == 0
    atlases = [line for line in capsys.readouterr().out.splitlines() if " -m simace atlas " in line]
    assert [line.rsplit("/", 1)[1] for line in atlases] == ["atlas.html", "atlas.pdf"]


def test_dry_run_subset_skips_plots_until_other_reps_are_complete(config_dir, roots, layout, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "--rep", "1", "--dry-run") == 0
    assert " -m simace plot " not in capsys.readouterr().out

    _finish(layout, _rep(config_dir, 2))
    assert _run(config_dir, roots, "tiny", "--rep", "1", "--dry-run") == 0
    out = capsys.readouterr().out
    assert " -m simace plot " in out
    assert " -m simace atlas " in out


def test_dry_run_refusal_skips_plots(config_dir, roots, layout, capsys) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    _stale(layout, rep)
    assert _run(config_dir, roots, "tiny", "--dry-run") == 1
    assert " -m simace plot " not in capsys.readouterr().out


def test_complete_reps_are_skipped_and_plots_rebuilt(config_dir, roots, layout, recorded, no_plots) -> None:
    for r in (1, 2):
        _finish(layout, _rep(config_dir, r))
    assert _run(config_dir, roots, "tiny") == 0
    assert recorded == []
    assert [rep.rep for rep in no_plots[0]] == [1, 2]


def test_stale_rep_is_refused_naming_the_changed_keys(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    stale = _rep(config_dir, 1)
    _finish(layout, stale)
    _stale(layout, stale)
    assert _run(config_dir, roots, "tiny") == 1
    out = capsys.readouterr().out
    assert "[tiny/rep1] refused: run.yaml differs in N: 999 -> 300; --force recomputes it" in out
    assert out.endswith(
        "[tiny] summary: 1 computed, 1 refused\n"
        "[tiny]   rep1 refused: run.yaml differs in N: 999 -> 300\n"
        "[tiny] plots skipped: rep1 not complete\n"
        "[tiny] plots: absent\n"
    )
    assert recorded == [2]
    assert no_plots == []


def test_second_run_of_a_locked_scenario_refuses(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    with run_mod._scenario_lock(layout.scenario_lock("t", "tiny")):
        assert _run(config_dir, roots, "tiny") == 1
        err = capsys.readouterr().err
        assert f"tiny is locked ({layout.scenario_lock('t', 'tiny')}) by simace run pid {os.getpid()}, or by" in err
        assert _run(config_dir, roots, "tiny", "--dry-run") == 0
        assert _run(config_dir, roots, "tiny_wf") == 0
        # A folder run takes every lock before starting anything.
        assert _run(config_dir, roots, "t") == 1
    assert recorded == [1]
    assert _run(config_dir, roots, "tiny") == 0
    assert recorded == [1, 1, 2]


def test_stage_keeps_scenario_locked_if_orchestrator_is_killed(tmp_path) -> None:
    lock = tmp_path / "scenario" / ".run.lock"
    ready, release = tmp_path / "ready", tmp_path / "release"
    child_code = (
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text('ready')\n"
        "deadline = time.monotonic() + 10\n"
        "while not pathlib.Path(sys.argv[2]).exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
    )
    parent_code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from simace.cli.launch import Launcher\nfrom simace.cli.run import _scenario_lock\n"
        "with _scenario_lock(Path(sys.argv[1])) as fd:\n"
        "    Launcher(dict(os.environ), lock_fds=(fd,)).run("
        "[sys.executable, '-c', sys.argv[4], sys.argv[2], sys.argv[3]], Path(sys.argv[5]))\n"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent_code,
            str(lock),
            str(ready),
            str(release),
            child_code,
            str(tmp_path / "stage.log"),
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), f"stage did not start; parent exited {parent.poll()}"
        parent.kill()
        parent.wait(timeout=5)
        with pytest.raises(run_mod.ScenarioBusy), run_mod._scenario_lock(lock):
            pass
    finally:
        release.write_text("release")
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with run_mod._scenario_lock(lock):
                return
        except run_mod.ScenarioBusy:
            time.sleep(0.01)
    pytest.fail("scenario lock was not released after the stage exited")


@pytest.mark.parametrize("reps", [["1", "1"], ["0"], ["3"], ["1-2", "2"]])
def test_rep_values_must_be_distinct_and_in_range(config_dir, roots, recorded, reps, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "--rep", *reps) == 2
    assert "--rep values must be distinct and in 1..2" in capsys.readouterr().err
    assert recorded == []


@pytest.mark.parametrize(("text", "reps"), [("7", [7]), ("3-5", [3, 4, 5]), ("2-2", [2])])
def test_rep_spec_parses_numbers_and_ranges(text, reps) -> None:
    assert run_mod.rep_spec(text) == reps


@pytest.mark.parametrize("text", ["", "x", "5-3", "-1", "1-"])
def test_rep_spec_rejects(text) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        run_mod.rep_spec(text)


def test_rep_range_selects_those_reps(config_dir, roots, recorded, no_plots) -> None:
    (config_dir / "t.yaml").write_text(yaml.safe_dump({"many": {**TINY, "replicates": 6}}))
    assert _run(config_dir, roots, "many", "--rep", "2-3", "6") == 0
    assert recorded == [2, 3, 6]


def test_rep_needs_exactly_one_scenario(config_dir, roots, recorded, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "tiny_wf", "--rep", "1") == 2
    assert "--rep needs exactly one scenario; the targets name 2" in capsys.readouterr().err
    assert recorded == []


def test_folder_target_runs_every_runnable_scenario_once(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    _finish(layout, _rep(config_dir, 1, "tiny_wf"))
    assert _run(config_dir, roots, "t", "tiny") == 0  # tiny, named twice, runs once
    out = capsys.readouterr().out
    assert recorded == [1, 1, 2, 1]  # broken, tiny x2, tiny_am; tiny_wf skipped; dropped left out
    assert "[tiny_wf/rep1] skip (run.yaml matches)" in out
    assert "[total] 4 scenarios: 1 skipped, 4 computed\n" in out
    assert [rep.scenario for reps in no_plots for rep in reps[:1]] == ["broken", "tiny", "tiny_am", "tiny_wf"]


def test_folder_target_with_a_failure_still_plots_the_complete_scenarios(
    config_dir, roots, monkeypatch, no_plots, capsys
) -> None:
    def recompute(rep, layout, env, console, *_):
        if rep.scenario == "broken":
            return "simulate FAILED (exit 1)"
        _finish(layout, rep)
        return None

    monkeypatch.setattr(run_mod, "_recompute", recompute)
    assert _run(config_dir, roots, "t") == 1
    out = capsys.readouterr().out
    assert "[broken] summary: 1 failed\n[broken]   rep1 failed: simulate FAILED (exit 1)\n" in out
    assert "[broken] plots skipped: rep1 not complete" in out
    assert "[total] 4 scenarios: 4 computed, 1 failed\n" in out
    assert sorted(reps[0].scenario for reps in no_plots) == ["tiny", "tiny_am", "tiny_wf"]


def test_unknown_target_lists_folders_and_scenarios(config_dir, roots, capsys) -> None:
    assert _run(config_dir, roots, "nope") == 2
    err = capsys.readouterr().err
    assert "unknown target 'nope'; folders: t" in err
    assert "tiny_wf" in err


def test_no_plots_skips_the_plot_pass(config_dir, roots, recorded, no_plots, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "--no-plots") == 0
    assert recorded == [1, 2]
    assert no_plots == []
    assert _run(config_dir, roots, "tiny", "--no-plots", "--dry-run") == 0
    assert " -m simace plot " not in capsys.readouterr().out


def test_roots_are_found_by_walking_up_from_cwd(config_dir, tmp_path, monkeypatch, recorded, no_plots) -> None:
    nested = tmp_path / "results" / "deep" / "er"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    cli(["tiny", "--rep", "1"])
    assert recorded == [1]
    assert (tmp_path / "results" / "t" / "tiny" / "rep1" / "run.yaml").exists()


def test_root_search_skips_a_nested_checkout_with_its_own_default_config(config_dir, tmp_path, monkeypatch) -> None:
    from simace.cli.layout import project_root

    nested = tmp_path / "fitACE" / "config"
    nested.mkdir(parents=True)
    (nested / "_default.yaml").write_text("fit: {}\n")
    (tmp_path / "fitACE" / "pyproject.toml").write_text('[project]\nname = "fitace"\n')
    monkeypatch.chdir(nested)
    assert project_root() == tmp_path.resolve()


def test_roots_stay_relative_when_cwd_is_the_root(config_dir, tmp_path, monkeypatch) -> None:
    from simace.cli.layout import resolve_roots

    monkeypatch.chdir(tmp_path)
    found_config, layout = resolve_roots(argparse.Namespace(config_dir=None, results=None, logs=None))
    assert (found_config, layout.root, layout.logs) == (Path("config"), Path("results"), Path("logs"))
    monkeypatch.chdir(tmp_path / "config")
    found_config, layout = resolve_roots(argparse.Namespace(config_dir=None, results=None, logs=None))
    assert (found_config, layout.root, layout.logs) == (tmp_path / "config", tmp_path / "results", tmp_path / "logs")


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
    assert "rep2] refused: run.yaml differs in rep: 1 -> 2, seed: 100 -> 101" in capsys.readouterr().out
    assert recorded == []


def test_a_rep_with_a_missing_output_is_recomputed(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    for r in (1, 2):
        _finish(layout, _rep(config_dir, r))
    layout.rep("t", "tiny", 2, RepArtifact.COHORT).unlink()
    assert status_on_disk(_rep(config_dir, 2), layout).state is RepState.INCOMPLETE
    assert status_on_disk(_rep(config_dir, 2), layout).reasons == ("cohort.parquet missing",)
    assert _run(config_dir, roots, "tiny") == 0
    assert "rep2] recompute: cohort.parquet missing" in capsys.readouterr().out
    assert recorded == [2]
    assert [rep.rep for rep in no_plots[0]] == [1, 2]


def test_an_output_rewritten_after_the_manifest_is_recomputed(
    config_dir, roots, layout, recorded, no_plots, capsys
) -> None:
    for r in (1, 2):
        _finish(layout, _rep(config_dir, r))
    cohort = layout.rep("t", "tiny", 2, RepArtifact.COHORT)
    os.utime(cohort, ns=(cohort.stat().st_atime_ns, cohort.stat().st_mtime_ns + 1))
    layout.rep("t", "tiny", 2, RepArtifact.REPORT).write_text("rewritten by hand\n")
    assert status_on_disk(_rep(config_dir, 2), layout).reasons == ("cohort.parquet changed", "report.yaml changed")
    assert _run(config_dir, roots, "tiny") == 0
    assert "rep2] recompute: cohort.parquet changed, report.yaml changed" in capsys.readouterr().out
    assert recorded == [2]


def test_a_manifest_without_output_fingerprints_is_refused(
    config_dir, roots, layout, recorded, no_plots, capsys
) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    manifest = _manifest(layout, rep)
    body = yaml.safe_load(manifest.read_text())
    del body["outputs"]
    manifest.write_text(yaml.safe_dump(body))
    assert _run(config_dir, roots, "tiny") == 1
    assert "rep1] refused: run.yaml differs in outputs: (absent) -> fingerprints (run.yaml predates them)" in (
        capsys.readouterr().out
    )
    assert recorded == [2]


def test_a_rep_from_before_layout_2_is_stale(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    manifest = _manifest(layout, rep)
    body = yaml.safe_load(manifest.read_text())
    del body["layout"]
    manifest.write_text(yaml.safe_dump(body))
    status = status_on_disk(rep, layout)
    assert (status.state, status.describe()) == (RepState.STALE, "layout: (absent) -> 2")
    assert _run(config_dir, roots, "tiny") == 1
    assert "rep1] refused: run.yaml differs in layout: (absent) -> 2; --force recomputes it" in capsys.readouterr().out
    assert recorded == [2]


def test_force_recomputes_complete_and_stale_reps(config_dir, roots, layout, recorded, no_plots) -> None:
    _finish(layout, _rep(config_dir, 1))
    assert _run(config_dir, roots, "tiny", "--force") == 0
    assert recorded == [1, 2]


def test_plots_wait_for_every_rep(config_dir, roots, recorded, no_plots) -> None:
    assert _run(config_dir, roots, "tiny", "--rep", "2") == 0
    assert recorded == [2]
    assert no_plots == []


def test_fail_fast_cancels_pending_reps(config_dir, roots, monkeypatch, no_plots, capsys) -> None:
    attempted: list[int] = []
    monkeypatch.setattr(run_mod, "_recompute", lambda rep, *a: attempted.append(rep.rep) or "analyze FAILED (exit 1)")
    assert _run(config_dir, roots, "tiny", "--fail-fast") == 1
    assert attempted == [1]
    assert capsys.readouterr().out.endswith(
        "[tiny] summary: 2 failed\n"
        "[tiny]   rep1 failed: analyze FAILED (exit 1)\n"
        "[tiny]   rep2 failed: cancelled by --fail-fast\n"
        "[tiny] plots skipped: reps 1-2 not complete\n"
        "[tiny] plots: absent\n"
    )


def test_summary_counts_skipped_and_computed_reps(config_dir, roots, layout, recorded, no_plots, capsys) -> None:
    _finish(layout, _rep(config_dir, 1))
    assert _run(config_dir, roots, "tiny") == 0
    assert "[tiny] summary: 1 skipped, 1 computed\n" in capsys.readouterr().out


def test_blas_is_always_pinned_and_kernels_only_for_concurrent_reps(monkeypatch) -> None:
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    one, many = launch_mod.child_env(1), launch_mod.child_env(2)
    assert one["OMP_NUM_THREADS"] == many["OMP_NUM_THREADS"] == "1"
    assert "NUMBA_NUM_THREADS" not in one
    assert many["NUMBA_NUM_THREADS"] == many["POLARS_MAX_THREADS"] == "1"


def test_pedigree_graph_splits_the_cores_across_concurrent_reps(monkeypatch) -> None:
    cores = os.process_cpu_count()
    monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
    assert launch_mod.child_env(1)["PEDIGREE_GRAPH_THREADS"] == str(cores)
    assert launch_mod.child_env(2)["PEDIGREE_GRAPH_THREADS"] == str(max(1, cores // 2))
    assert launch_mod.child_env(cores + 1)["PEDIGREE_GRAPH_THREADS"] == "1"
    monkeypatch.setenv("PEDIGREE_GRAPH_THREADS", "3")
    assert launch_mod.child_env(1)["PEDIGREE_GRAPH_THREADS"] == "3"
    assert launch_mod.child_env(2)["PEDIGREE_GRAPH_THREADS"] == "3"


def test_child_env_without_sched_getaffinity(monkeypatch) -> None:
    """macOS has no ``os.sched_getaffinity``; ``simace run`` must still build its env (#38)."""
    # Mirror os.py off Linux: no affinity call, process_cpu_count is cpu_count.
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 4)
    monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
    assert launch_mod.child_env(2)["PEDIGREE_GRAPH_THREADS"] == "2"
    monkeypatch.setattr(os, "process_cpu_count", lambda: None)  # indeterminable
    assert launch_mod.child_env(1)["PEDIGREE_GRAPH_THREADS"] == "1"


@pytest.mark.parametrize(
    ("text", "size"),
    [("512M", 512 * 2**20), ("8G", 8 * 2**30), ("8gb", 8 * 2**30), ("1.5G", 3 * 2**29), ("4096", 4096)],
)
def test_parse_size(text, size) -> None:
    assert launch_mod.parse_size(text) == size


@pytest.mark.parametrize("text", ["", "lots", "0", "-1G", "8X"])
def test_parse_size_rejects(text) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        launch_mod.parse_size(text)


@pytest.fixture(params=["cgroup", "polled"])
def cgroups(request, monkeypatch) -> Iterator[CgroupRoot | None]:
    """A delegated cgroup root, or None for the ``/proc`` polling fallback."""
    if request.param == "polled":
        yield None
        return
    monkeypatch.delenv("SIMACE_CGROUP_ROOT", raising=False)
    try:
        root = CgroupRoot.open()
    except CgroupUnavailable as exc:
        pytest.skip(f"no delegated cgroup: {exc}")
    with root:
        yield root


def test_launcher_kills_a_stage_over_the_memory_cap(tmp_path, cgroups) -> None:
    hog = [sys.executable, "-c", "import time; b = b'x' * (400 * 2**20); time.sleep(60)"]
    log = tmp_path / "hog.log"
    result = launch_mod.Launcher(dict(os.environ), max_rss=200 * 2**20, cgroups=cgroups).run(hog, log)
    assert (result.exit_code, result.over_memory) == (-9, True)
    assert result.wall_s < 30
    assert "went over --max-memory (200 MB)" in log.read_text()


def test_launcher_counts_and_kills_a_stage_s_worker_processes(tmp_path, cgroups) -> None:
    pids = tmp_path / "workers.txt"
    worker = "import time; time.sleep(0.5); b = b'x' * (120 * 2**20); time.sleep(60)"
    stage = [
        sys.executable,
        "-c",
        "import subprocess, sys, time; "
        f"ws = [subprocess.Popen([sys.executable, '-c', {worker!r}]) for _ in range(3)]; "
        f"open({str(pids)!r}, 'w').write(' '.join(str(w.pid) for w in ws)); time.sleep(60)",
    ]
    result = launch_mod.Launcher(dict(os.environ), max_rss=250 * 2**20, cgroups=cgroups).run(
        stage, tmp_path / "pool.log"
    )
    assert (result.exit_code, result.over_memory) == (-9, True)
    assert result.wall_s < 30
    for pid in map(int, pids.read_text().split()):
        deadline = time.monotonic() + 10
        while launch_mod._rss_bytes(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert launch_mod._rss_bytes(pid) == 0


def test_launcher_records_the_peak_of_a_stage_s_process_tree_in_a_cgroup(tmp_path, cgroups) -> None:
    worker = "import time; b = b'x' * (120 * 2**20); time.sleep(1)"
    stage = [
        sys.executable,
        "-c",
        f"import subprocess, sys; [w.wait() for w in [subprocess.Popen([sys.executable, '-c', {worker!r}]) for _ in range(3)]]",
    ]
    result = launch_mod.Launcher(dict(os.environ), cgroups=cgroups).run(stage, tmp_path / "pool.log")
    assert result.exit_code == 0
    if cgroups is None:
        assert result.tree_peak_mb is None
    else:
        assert result.tree_peak_mb > 3 * 120


def test_launcher_leaves_a_stage_under_the_cap_alone(tmp_path, cgroups) -> None:
    ok = [sys.executable, "-c", "pass"]
    result = launch_mod.Launcher(dict(os.environ), max_rss=2**30, cgroups=cgroups).run(ok, tmp_path / "ok.log")
    assert (result.exit_code, result.over_memory) == (0, False)


def test_max_memory_fails_the_rep_and_names_the_cap(config_dir, roots, layout, no_plots, capsys) -> None:
    assert _run(config_dir, roots, "tiny", "--rep", "1", "--max-memory", "20M") == 1
    assert "simulate FAILED (exit -9, killed for going over --max-memory)" in capsys.readouterr().out
    rep_dir = layout.rep_dir("t", "tiny", 1)
    assert (rep_dir / "timing.tsv").read_text().splitlines()[-1].endswith("\t-9")
    assert not (rep_dir / "run.yaml").exists()


@pytest.mark.slow
def test_recompute_runs_every_stage_and_records_it(config_dir, roots, layout, no_plots) -> None:
    rep = _rep(config_dir, 1)
    rep_dir = layout.rep_dir(rep.folder, rep.scenario, 1)
    rep_dir.mkdir(parents=True)
    (rep_dir / "cohort.parquet.tmp").write_text("left by a killed stage")
    for retired in RETIRED_OUTPUTS:
        (rep_dir / retired).write_text("written by an earlier results layout")

    assert _run(config_dir, roots, "tiny", "--rep", "1") == 0

    assert [name for name in RETIRED_OUTPUTS if (rep_dir / name).exists()] == []

    timing = (rep_dir / "timing.tsv").read_text().splitlines()
    assert [row.split("\t")[0] for row in timing[1:]] == [s.name for s in STAGES]
    assert all(row.endswith("\t0") for row in timing[1:])
    assert not list(rep_dir.glob("*.tmp"))
    for stage in STAGES:
        assert all(layout.rep(rep.folder, rep.scenario, 1, out).exists() for out in stage.outputs)
        assert layout.log(rep.folder, rep.scenario, 1, stage.name).exists()
    manifest = yaml.safe_load((rep_dir / "run.yaml").read_text())
    assert manifest["stages"] == [s.name for s in STAGES]
    assert manifest["layout"] == 2
    assert manifest["seed"] == 100
    params = yaml.safe_load((rep_dir / "params.yaml").read_text())
    assert (params["seed"], params["rep"], params["G_pheno"]) == (100, 1, 2)


def test_failing_stage_leaves_no_manifest(config_dir, roots, layout, no_plots, capsys) -> None:
    rep = _rep(config_dir, 1, "broken")
    rep_dir = layout.rep_dir(rep.folder, "broken", 1)
    rep_dir.mkdir(parents=True)
    old_outputs = ["pedigree.parquet", "cohort.parquet", "report.yaml", "run.yaml"]
    for name in old_outputs:
        (rep_dir / name).write_text("from the previous parameters")

    assert _run(config_dir, roots, "broken", "--force") == 1

    assert [name for name in old_outputs[1:] if (rep_dir / name).exists()] == []
    assert (rep_dir / "pedigree.parquet").read_bytes() != b"from the previous parameters"
    last = (rep_dir / "timing.tsv").read_text().splitlines()[-1].split("\t")
    assert (last[0], last[-1]) == ("cohort", "1")
    assert not (rep_dir / "run.yaml").exists()
    assert "cohort FAILED" in capsys.readouterr().out
    assert "G_pheno" in layout.log(rep.folder, "broken", 1, "cohort").read_text()


def _mtimes(layout: Layout, rep: ResolvedRep, *artifacts: RepArtifact) -> list[int]:
    return [layout.rep(rep.folder, rep.scenario, rep.rep, a).stat().st_mtime_ns for a in artifacts]


def test_until_cohort_leaves_a_partial_rep_and_skips_plots(config_dir, roots, layout, no_plots, capsys) -> None:
    rep = _rep(config_dir, 1)
    assert _run(config_dir, roots, "tiny", "--rep", "1", "--until", "cohort") == 0

    rep_dir = layout.rep_dir(rep.folder, rep.scenario, 1)
    assert (rep_dir / "pedigree.parquet").exists()
    assert (rep_dir / "cohort.parquet").exists()
    assert not (rep_dir / "report.yaml").exists()
    assert yaml.safe_load((rep_dir / "run.yaml").read_text())["stages"] == ["simulate", "cohort"]
    header, *timing = (rep_dir / "timing.tsv").read_text().splitlines()
    assert header.split("\t") == ["stage", "wall_s", "max_rss_mb", "tree_peak_mb", "exit_code"]
    assert [row.split("\t")[0] for row in timing] == ["simulate", "cohort"]
    assert all(len(row.split("\t")) == 5 for row in timing)
    status = status_on_disk(rep, layout)
    assert (status.state, status.reasons) == (RepState.PARTIAL, ("built through cohort",))
    assert status_on_disk(rep, layout, until=2).state is RepState.COMPLETE
    assert no_plots == []
    assert "[tiny] plots skipped: --until cohort" in capsys.readouterr().out


@pytest.mark.slow
def test_a_full_run_resumes_a_partial_rep_at_its_first_missing_stage(
    config_dir, roots, layout, no_plots, capsys
) -> None:
    rep = _rep(config_dir, 1)
    assert _run(config_dir, roots, "tiny", "--rep", "1", "--until", "cohort") == 0
    kept = _mtimes(layout, rep, RepArtifact.PEDIGREE, RepArtifact.COHORT)
    capsys.readouterr()

    assert _run(config_dir, roots, "tiny", "--rep", "1") == 0

    assert "[tiny/rep1] resume: built through cohort" in capsys.readouterr().out
    assert _mtimes(layout, rep, RepArtifact.PEDIGREE, RepArtifact.COHORT) == kept
    rep_dir = layout.rep_dir(rep.folder, rep.scenario, 1)
    timing = (rep_dir / "timing.tsv").read_text().splitlines()[1:]
    assert [row.split("\t")[0] for row in timing] == [s.name for s in STAGES]
    assert yaml.safe_load((rep_dir / "run.yaml").read_text())["stages"] == [s.name for s in STAGES]
    assert status_on_disk(rep, layout).state is RepState.COMPLETE


def test_until_skips_a_rep_built_further(config_dir, roots, layout, no_plots, capsys) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    before = _manifest(layout, rep).read_text()
    assert _run(config_dir, roots, "tiny", "--rep", "1", "--until", "cohort") == 0
    assert "[tiny/rep1] skip (run.yaml matches)" in capsys.readouterr().out
    assert _manifest(layout, rep).read_text() == before


@pytest.mark.slow
def test_an_analyze_only_key_leaves_a_partial_rep_resumable(config_dir, roots, layout, no_plots, capsys) -> None:
    rep = _rep(config_dir, 1)
    assert _run(config_dir, roots, "tiny", "--rep", "1", "--until", "cohort") == 0
    scenarios = yaml.safe_load((config_dir / "t.yaml").read_text())
    scenarios["tiny"]["analysis"] = {"max_degree": 1}
    (config_dir / "t.yaml").write_text(yaml.safe_dump(scenarios))
    capsys.readouterr()

    assert _run(config_dir, roots, "tiny", "--rep", "1") == 0

    assert "resume: built through cohort" in capsys.readouterr().out
    params = yaml.safe_load(layout.rep(rep.folder, rep.scenario, 1, RepArtifact.PARAMS).read_text())
    assert params["max_degree"] == 1
    assert status_on_disk(_rep(config_dir, 1), layout).state is RepState.COMPLETE


def test_a_partial_rep_is_stale_when_its_own_stage_keys_change(config_dir, layout) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    _write_manifest_through(layout, rep, 2)
    changed = ResolvedRep(rep.folder, rep.scenario, 1, {**rep.params, "death_rho": 3.0})
    assert status_on_disk(changed, layout).reasons == ("death_rho",)
    assert status_on_disk(changed, layout).state is RepState.STALE


def test_a_partial_rep_with_a_missing_output_is_recomputed(config_dir, layout) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    _write_manifest_through(layout, rep, 2)
    layout.rep(rep.folder, rep.scenario, 1, RepArtifact.COHORT).unlink()
    assert status_on_disk(rep, layout).state is RepState.INCOMPLETE


def test_dry_run_until_prints_only_the_stages_asked_for(config_dir, roots, layout, capsys) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    _write_manifest_through(layout, rep, 1)
    assert _run(config_dir, roots, "tiny", "--until", "cohort", "--dry-run") == 0
    lines = capsys.readouterr().out.splitlines()
    commands = [line.split(" -m simace ")[1].split()[0] for line in lines if " -m simace " in line]
    assert commands == ["cohort", "simulate", "cohort"]


def test_ls_reports_partial_reps(config_dir, layout, tmp_path, capsys) -> None:
    rep = _rep(config_dir, 1)
    _finish(layout, rep)
    _write_manifest_through(layout, rep, 2)
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    lines = dict(line.split("  ", 1) for line in capsys.readouterr().out.splitlines())
    assert lines["t/tiny"] == "2 reps: 1 partial (rep1: built through cohort), 1 absent (rep2); plots absent"


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
    assert shown["timing"] == {"from_complete_reps": 0}


def test_show_reports_timing_over_complete_reps(config_dir, layout, tmp_path, capsys) -> None:
    """Rep 1 predates tree_peak_mb; rep 2 records it for simulate only."""
    tables = [
        "stage\twall_s\tmax_rss_mb\texit_code\nsimulate\t10.0\t500\t0\nanalyze\t1.0\t250\t0\n",
        "stage\twall_s\tmax_rss_mb\ttree_peak_mb\texit_code\nsimulate\t30.0\t900\t1200.4\t0\nanalyze\t1.0\t450\t\t0\n",
    ]
    for r, table in enumerate(tables, start=1):
        rep = _rep(config_dir, r)
        timing = layout.rep("t", "tiny", r, RepArtifact.TIMING)
        timing.parent.mkdir(parents=True)
        timing.write_text(table)
        _finish(layout, rep)
    show_cli(["tiny", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    timing = yaml.safe_load(capsys.readouterr().out)["timing"]
    assert timing == {
        "from_complete_reps": 2,
        "simulate": {
            "wall_s_median": 20.0,
            "max_rss_mb": 900,
            "max_rss_rep": 2,
            "tree_peak_mb": 1200,
            "tree_peak_rep": 2,
        },
        "analyze": {"wall_s_median": 1.0, "max_rss_mb": 450, "max_rss_rep": 2},
    }


def test_show_can_inspect_gene_drop_scenario(config_dir, capsys) -> None:
    show_cli(["dropped", "--config-dir", str(config_dir)])
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["params"]["use_gene_drop"] is True
    assert shown["reps"]["rep1"]["seed"] == 100


def test_ls_reports_each_rep_state(config_dir, layout, tmp_path, capsys) -> None:
    _finish(layout, _rep(config_dir, 1))
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    lines = dict(line.split("  ", 1) for line in capsys.readouterr().out.splitlines())
    assert lines["t/tiny"] == "2 reps: 1 complete, 1 absent (rep2); plots absent"
    assert lines["t/dropped"] == "gene drop (not run by simace run)"


def test_ls_groups_reps_by_state_and_reason(config_dir, layout, tmp_path, capsys) -> None:
    (config_dir / "t.yaml").write_text(yaml.safe_dump({"many": {**TINY, "replicates": 7}}))
    reps = [_rep(config_dir, r, "many") for r in range(1, 8)]
    for rep in reps[:5]:
        _finish(layout, rep)
    for rep in reps[1:3]:
        _stale(layout, rep)
    layout.rep("t", "many", 5, RepArtifact.COHORT).unlink()
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    line = capsys.readouterr().out.strip()
    assert line == (
        "t/many  7 reps: 2 complete, 2 stale (reps 2-3: N: 999 -> 300), "
        "1 incomplete (rep5: cohort.parquet missing), 2 absent (reps 6-7); plots absent"
    )


@pytest.mark.parametrize(
    ("reps", "text"), [([2], "rep2"), ([1, 2, 3], "reps 1-3"), ([1, 2, 3, 5, 8, 9], "reps 1-3, 5, 8-9")]
)
def test_ls_rep_ranges(reps, text) -> None:
    assert rep_ranges(reps) == text


def test_ls_flags_reps_built_by_another_version(config_dir, layout, tmp_path, capsys) -> None:
    for r in (1, 2):
        _finish(layout, _rep(config_dir, r))
    old = _manifest(layout, _rep(config_dir, 2))
    old.write_text(old.read_text().replace(f"simace_version: {simace.__version__}", "simace_version: 2020.1.0"))
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    lines = dict(line.split("  ", 1) for line in capsys.readouterr().out.splitlines())
    assert lines["t/tiny"] == "2 reps: 2 complete (rep2: built by simace 2020.1.0); plots absent"


def test_ls_flags_reps_built_at_another_commit(config_dir, layout, tmp_path, monkeypatch, capsys) -> None:
    import simace.cli.inspect as inspect_mod
    import simace.cli.manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "source_ref", lambda: "v2026.9-3-gabc123")
    monkeypatch.setattr(inspect_mod, "source_ref", lambda: "v2026.9-3-gabc123")
    for r in (1, 2):
        _finish(layout, _rep(config_dir, r))
    assert yaml.safe_load(_manifest(layout, _rep(config_dir, 1)).read_text())["source"] == "v2026.9-3-gabc123"
    monkeypatch.setattr(inspect_mod, "source_ref", lambda: "v2026.9-5-gdef456-dirty")
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    lines = dict(line.split("  ", 1) for line in capsys.readouterr().out.splitlines())
    assert lines["t/tiny"] == "2 reps: 2 complete (reps 1-2: built at v2026.9-3-gabc123); plots absent"


def test_source_ref_describes_this_checkout() -> None:
    from simace.cli.manifest import source_ref

    ref = source_ref()
    assert ref is not None
    assert (
        subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"], capture_output=True, text=True, check=True
        ).stdout.strip()
        == ref
    )


@pytest.mark.slow
def test_end_to_end_run_builds_plots_and_atlas(config_dir, roots, layout) -> None:
    assert _run(config_dir, roots, "tiny_wf") == 0
    plots = layout.scenario_plots("t", "tiny_wf")
    assert (plots / "atlas.html").exists()
    assert not (plots / "atlas.pdf").exists()
    assert [row.split("\t")[0] for row in (plots / "timing.tsv").read_text().splitlines()[1:]] == ["plot", "atlas"]
    assert _run(config_dir, roots, "tiny_wf", "--format", "pdf") == 0
    assert (plots / "atlas.pdf").exists()
    rows = [row.split("\t")[0] for row in (plots / "timing.tsv").read_text().splitlines()[1:]]
    assert rows == ["plot", "atlas", "atlas-pdf"]
    assert layout.scenario_log("t", "tiny_wf", "atlas-pdf").exists()
    reps = [_rep(config_dir, 1, "tiny_wf")]
    assert scenario_plots_status(reps, layout).state is PlotsState.CURRENT
    assert set(yaml.safe_load(layout.scenario_plots_manifest("t", "tiny_wf").read_text())["outputs"]) == {
        "atlas.html",
        "atlas.pdf",
    }
    (plots / "atlas.pdf").unlink()
    assert scenario_plots_status(reps, layout).reasons == ("atlas.pdf missing",)


def _plotted(layout: Layout, reps: list[ResolvedRep], *outputs: str) -> None:
    """Leave the scenario's plots on disk as a finished plot pass does."""
    folder, scenario = reps[0].folder, reps[0].scenario
    plots = layout.scenario_plots(folder, scenario)
    plots.mkdir(parents=True, exist_ok=True)
    for name in outputs:
        (plots / name).write_text("")
    write_plots_manifest(
        layout.scenario_plots_manifest(folder, scenario),
        {f"rep{rep.rep}": _manifest(layout, rep) for rep in reps},
        [plots / name for name in outputs],
    )


def test_a_failed_plot_pass_leaves_no_plots_manifest(config_dir, roots, layout, recorded, monkeypatch, capsys) -> None:
    reps = [_rep(config_dir, r) for r in (1, 2)]
    for rep in reps:
        _finish(layout, rep)
    _plotted(layout, reps, "atlas.html")
    failing = launch_mod.StageResult(wall_s=0.1, max_rss_mb=1.0, tree_peak_mb=None, exit_code=1)
    monkeypatch.setattr(launch_mod.Launcher, "run", lambda self, command, log_path: failing)
    assert _run(config_dir, roots, "tiny") == 1
    assert capsys.readouterr().out.endswith("[tiny] plots: absent\n")
    assert not layout.scenario_plots_manifest("t", "tiny").exists()


def test_ls_and_show_survive_a_scenario_with_no_replicates(config_dir, tmp_path, capsys) -> None:
    (config_dir / "t.yaml").write_text(yaml.safe_dump({"parked": {**TINY, "replicates": 0}}))
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    assert capsys.readouterr().out.strip() == "t/parked  0 reps: ; plots absent"
    show_cli(["parked", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    assert yaml.safe_load(capsys.readouterr().out)["plots"]["state"] == "absent"


def test_plots_are_current_until_a_rep_or_atlas_changes(config_dir, layout) -> None:
    reps = [_rep(config_dir, r) for r in (1, 2)]
    for rep in reps:
        _finish(layout, rep)
    _plotted(layout, reps, "atlas.html")
    assert scenario_plots_status(reps, layout).state is PlotsState.CURRENT

    time.sleep(0.01)
    _finish(layout, reps[1])
    assert scenario_plots_status(reps, layout).describe() == "stale (rep2 recomputed since)"

    _plotted(layout, reps, "atlas.html")
    layout.rep("t", "tiny", 1, RepArtifact.COHORT).write_text("by hand")
    assert scenario_plots_status(reps, layout).describe() == "stale (rep1 not complete)"

    _finish(layout, reps[0])
    _plotted(layout, reps[:1], "atlas.html")
    assert scenario_plots_status(reps, layout).describe() == "stale (rep2 not plotted)"

    _plotted(layout, reps, "atlas.html")
    layout.scenario_plots("t", "tiny").joinpath("atlas.html").unlink()
    assert scenario_plots_status(reps, layout).describe() == "stale (atlas.html missing)"


def test_run_summary_and_ls_report_the_plot_state(
    config_dir, roots, layout, recorded, no_plots, tmp_path, capsys
) -> None:
    reps = [_rep(config_dir, r) for r in (1, 2)]
    for rep in reps:
        _finish(layout, rep)
    _plotted(layout, reps, "atlas.html")
    assert _run(config_dir, roots, "tiny", "--no-plots") == 0
    assert capsys.readouterr().out.endswith("[tiny] summary: 2 skipped\n[tiny] plots: current\n")

    time.sleep(0.01)
    assert _run(config_dir, roots, "tiny", "--rep", "2", "--force") == 0
    assert capsys.readouterr().out.endswith("[tiny] plots: stale (rep2 recomputed since)\n")
    ls_cli(["t", "--config-dir", str(config_dir), "--results", str(tmp_path / "results")])
    lines = dict(line.split("  ", 1) for line in capsys.readouterr().out.splitlines())
    assert lines["t/tiny"] == "2 reps: 2 complete; plots stale (rep2 recomputed since)"


def test_resolving_config_does_not_import_the_stage_stack():
    probe = (
        "import sys; from pathlib import Path; import simace.cli.run as r; r.resolve_all(Path(sys.argv[1])); "
        "print(' '.join(m for m in ('numba', 'polars', 'scipy', 'pedigree_graph', 'simace.phenotype') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe, str(REPO_CONFIG)], capture_output=True, text=True, check=True)
    assert out.stdout.split() == []
