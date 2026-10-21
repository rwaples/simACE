"""``simace validate``: on-demand checks over a scenario's complete reps (ADR 0027, 0028)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from simace.analysis.validate.on_demand import SCHEMA
from simace.analysis.validate.verdict import SCHEMA as VERDICT_SCHEMA
from simace.analysis.validate.verdict import Entry, Insufficient, MeanZ
from simace.cli.layout import Layout, RepArtifact
from simace.cli.run import cli as run_cli
from simace.cli.stages import ResolvedRep
from simace.cli.status import load_scenario
from simace.cli.validate import cli as validate_cli
from simace.core.cohort import read_cohort, read_pedigree
from simace.phenotype.runner import run_phenotype

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
TINY = {"N": 300, "G_ped": 3, "G_sim": 3, "G_pheno": 2, "replicates": 2, "seed": 100}


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> tuple[Path, list[str], Layout]:
    """A tiny scenario with both reps computed once for the module."""
    root = tmp_path_factory.mktemp("project")
    cfg = root / "config"
    cfg.mkdir()
    shutil.copy(REPO_CONFIG / "_default.yaml", cfg / "_default.yaml")
    (root / "pyproject.toml").write_text('[project]\nname = "simace"\n')
    (cfg / "t.yaml").write_text(yaml.safe_dump({"tiny": TINY}))
    roots = ["--config-dir", str(cfg), "--results", str(root / "results"), "--logs", str(root / "logs")]
    run_cli(["tiny", "--no-plots", *roots])
    return cfg, roots, Layout(root=root / "results", logs=root / "logs")


def _rep(cfg: Path, rep: int) -> ResolvedRep:
    params = load_scenario(cfg, "tiny")
    return ResolvedRep(params["folder"], "tiny", rep, params)


def test_writes_validation_yaml_for_each_complete_rep(project, capsys) -> None:
    cfg, roots, layout = project
    validate_cli(["tiny", *roots, "--no-plots"])
    out = capsys.readouterr().out
    assert "[tiny] validated 2 rep(s), 0 already current, 0 errors" in out
    assert "onset_tau " in out
    assert "cell(s) scored" in out
    assert "passed" not in out
    assert "FAILED" not in out
    for rep in (1, 2):
        data = yaml.safe_load(_rep(cfg, rep).path(layout, RepArtifact.VALIDATION).read_text())
        assert data["schema"] == SCHEMA
        assert "summary" not in data
        section = data["checks"]["onset_tau"]
        assert set(section) == {"scope", "cells", "summary"}
        assert {"trait1_FS", "trait2_FS", "within_person"} <= set(section["cells"])
        assert {"n_scored", "max_abs_z", "z_bound"} <= set(section["summary"])
        assert data["seed"] == _rep(cfg, rep).seed


def test_regenerated_onsets_are_the_cohort_stage_onsets(project) -> None:
    """The validate primitive's run_phenotype call reproduces the raw onsets the rep's cohort holds."""
    cfg, _, layout = project
    rep = _rep(cfg, 1)
    pedigree = read_pedigree(rep.path(layout, RepArtifact.PEDIGREE))
    phenotype = {key: rep.params[key] for key in ("G_pheno", "standardize")}
    for t in (1, 2):
        for key in ("phenotype_model", "beta", "beta_sex", "phenotype_params"):
            phenotype[f"{key}{t}"] = rep.params[f"{key}{t}"]
    raw = run_phenotype(pedigree, seed=rep.seed, **phenotype)
    cohort = read_cohort(rep.path(layout, RepArtifact.COHORT)).filter(pl.col("affected1").is_not_null())
    joined = cohort.select("id", "t1", "t2").join(raw, on="id", suffix="_regen")
    assert len(joined) == len(cohort)
    for t in ("t1", "t2"):
        np.testing.assert_array_equal(joined[t].to_numpy(), joined[f"{t}_regen"].to_numpy().astype(np.float32))


def test_incomplete_reps_are_skipped_with_their_state(project, capsys) -> None:
    cfg, roots, layout = project
    manifest = _rep(cfg, 2).path(layout, RepArtifact.RUN_MANIFEST)
    saved = manifest.read_bytes()
    manifest.unlink()
    try:
        validate_cli(["tiny", "--rep", "2", *roots, "--no-plots"])
    finally:
        manifest.write_bytes(saved)
    out = capsys.readouterr().out
    assert "[tiny/rep2] skipped (" in out
    assert "[tiny/rep2] validated" not in out
    assert "[tiny] validated 0 rep(s)" in out


def test_dry_run_prints_the_per_rep_command_and_writes_nothing(project, capsys) -> None:
    cfg, roots, layout = project
    target = _rep(cfg, 1).path(layout, RepArtifact.VALIDATION)
    target.unlink(missing_ok=True)
    validate_cli(["tiny", "--rep", "1", "--dry-run", *roots, "--no-plots"])
    out = capsys.readouterr().out
    assert "validate --pedigree" in out
    assert "--checks cross_trait_rC household_draw mendelian_residual onset_tau total_variance" in out
    assert "--report" in out
    assert "--phenotype-params1" in out
    assert not target.exists()


def test_a_recompute_removes_the_reps_validation_yaml_and_both_verdicts(project) -> None:
    cfg, roots, layout = project
    validate_cli(["t", *roots, "--no-plots"])
    targets = [
        _rep(cfg, 1).path(layout, RepArtifact.VALIDATION),
        layout.scenario_verdict("t", "tiny"),
        layout.folder_verdict("t"),
    ]
    assert all(target.exists() for target in targets)
    run_cli(["tiny", "--rep", "1", "--force", "--no-plots", *roots])
    assert not any(target.exists() for target in targets)


def test_a_per_rep_run_without_checks_is_rejected(project, tmp_path, capsys) -> None:
    """The recorded-pedigree debug run is gone (ADR 0028); ``simace analyze`` runs those checks."""
    cfg, _, layout = project
    rep = _rep(cfg, 1)
    with pytest.raises(SystemExit) as exc:
        validate_cli(
            [
                "--pedigree",
                str(rep.path(layout, RepArtifact.PEDIGREE)),
                "--params",
                str(rep.path(layout, RepArtifact.PARAMS)),
                "--output",
                str(tmp_path / "validation.yaml"),
            ]
        )
    assert exc.value.code == 2
    assert "`simace analyze` runs those checks" in capsys.readouterr().err


def test_current_reps_are_skipped_unless_forced(project, capsys) -> None:
    cfg, roots, layout = project
    validate_cli(["tiny", *roots, "--no-plots"])
    target = _rep(cfg, 1).path(layout, RepArtifact.VALIDATION)
    written = target.stat().st_mtime_ns
    capsys.readouterr()
    validate_cli(["tiny", *roots, "--no-plots"])
    assert "[tiny] validated 0 rep(s), 2 already current, 0 errors" in capsys.readouterr().out
    assert target.stat().st_mtime_ns == written
    validate_cli(["tiny", "--rep", "1", "--force", *roots, "--no-plots"])
    assert "[tiny] validated 1 rep(s), 0 already current, 0 errors" in capsys.readouterr().out
    assert target.stat().st_mtime_ns != written


def test_two_reps_give_an_insufficient_verdict_for_the_scenario_and_folder(project, capsys) -> None:
    _, roots, layout = project
    validate_cli(["t", *roots, "--no-plots"])
    out = capsys.readouterr().out
    assert "[tiny] verdict insufficient: no test decided over 2 rep(s)" in out
    assert "[t] folder verdict insufficient: no test decided over 2 rep(s)" in out
    scenario = yaml.safe_load(layout.scenario_verdict("t", "tiny").read_text())
    assert scenario["schema"] == VERDICT_SCHEMA
    assert (scenario["family"], scenario["scenario"], scenario["seed"], scenario["reps"]) == (
        "scenario",
        "tiny",
        100,
        "1-2",
    )
    assert scenario["tests"]["onset_tau"] == {"insufficient": "2 rep(s); the verdict needs 20"}
    folder = yaml.safe_load(layout.folder_verdict("t").read_text())
    assert folder["reps_by_scenario"] == {"tiny": {"seed": 100, "reps": "1-2"}}
    assert folder["pooled"]["onset_tau"] == {"insufficient": "tiny below 20 rep(s)"}


def test_reps_from_two_builds_get_no_verdict_until_revalidated(project, capsys) -> None:
    cfg, roots, layout = project
    validate_cli(["tiny", *roots, "--no-plots"])
    assert layout.scenario_verdict("t", "tiny").exists()
    path = _rep(cfg, 2).path(layout, RepArtifact.VALIDATION)
    data = yaml.safe_load(path.read_text())
    data["simace_version"] = "0.0.1"
    path.write_text(yaml.safe_dump(data))
    capsys.readouterr()
    validate_cli(["tiny", "--rep", "1", *roots, "--no-plots"])
    out = capsys.readouterr().out
    assert "[tiny] no verdict: reps validated by more than one build: 0.0.1 at " in out
    assert " (rep2); " in out
    assert not layout.scenario_verdict("t", "tiny").exists()
    validate_cli(["tiny", *roots, "--no-plots"])
    out = capsys.readouterr().out
    assert "[tiny] validated 1 rep(s), 1 already current, 0 errors" in out
    assert "[tiny] verdict insufficient" in out


def test_overlapping_seed_ranges_refuse_only_the_pooled_tests(project, capsys, monkeypatch) -> None:
    import simace.cli.validate as validate_module

    _, roots, layout = project
    monkeypatch.setattr(validate_module, "overlapping_seed_ranges", lambda ranges: [("tiny", "other")])
    validate_cli(["t", *roots, "--no-plots"])
    out = capsys.readouterr().out
    assert "[t] folder verdict insufficient" in out
    assert "; pooling refused, rep seed ranges overlap: tiny and other" in out
    folder = yaml.safe_load(layout.folder_verdict("t").read_text())
    assert folder["pooled"]["onset_tau"] == {"insufficient": "rep seed ranges overlap: tiny and other"}
    assert "onset_tau" in folder["scenarios"]["tiny"]


def test_a_folder_target_with_no_scenario_to_validate_has_no_verdict(project) -> None:
    import simace.cli.validate as validate_module

    _, _, layout = project
    assert validate_module._folder_verdict("gene_drop_only", [], {}, layout) == (
        None,
        "no verdict: no scenario to validate",
    )


def test_a_failed_rep_leaves_no_verdict_beside_its_old_file(project, capsys, monkeypatch) -> None:
    import simace.cli.validate as validate_module

    _, roots, layout = project
    validate_cli(["tiny", *roots, "--no-plots"])
    assert layout.scenario_verdict("t", "tiny").exists()
    monkeypatch.setattr(validate_module, "_validate_rep", lambda *args: "ERROR (exit 1); log: x")
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        validate_cli(["tiny", "--rep", "1", "--force", *roots, "--no-plots"])
    assert exc.value.code == 1
    assert "[tiny] no verdict: rep1 failed validation" in capsys.readouterr().out
    assert not layout.scenario_verdict("t", "tiny").exists()


def test_a_rep_from_another_checkout_state_is_validated_again(project, capsys) -> None:
    """The build is the version and the ``git describe`` source without ``-dirty``: a commit without a reinstall counts."""
    from simace.cli.manifest import source_ref

    cfg, roots, layout = project
    validate_cli(["tiny", *roots, "--no-plots"])
    path = _rep(cfg, 1).path(layout, RepArtifact.VALIDATION)
    data = yaml.safe_load(path.read_text())
    assert data["source"] == source_ref()
    source = data["source"]
    data["source"] = source.removesuffix("-dirty") if source.endswith("-dirty") else source + "-dirty"
    path.write_text(yaml.safe_dump(data))
    capsys.readouterr()
    validate_cli(["tiny", *roots, "--no-plots"])
    assert "[tiny] validated 0 rep(s), 2 already current, 0 errors" in capsys.readouterr().out, "-dirty is ignored"
    data["source"] = "v0.0.0-1-gdeadbee"
    path.write_text(yaml.safe_dump(data))
    validate_cli(["tiny", *roots, "--no-plots"])
    assert "[tiny] validated 1 rep(s), 1 already current, 0 errors" in capsys.readouterr().out


def test_a_verdict_needs_every_registered_test_whatever_checks_ran(project, capsys, monkeypatch) -> None:
    import simace.cli.validate as validate_module

    _, roots, layout = project
    validate_cli(["tiny", *roots, "--no-plots"])
    assert layout.scenario_verdict("t", "tiny").exists()
    registry = dict(validate_module.ON_DEMAND_CHECKS)
    registry["other"] = registry["onset_tau"]
    monkeypatch.setattr(validate_module, "ON_DEMAND_CHECKS", registry)
    capsys.readouterr()
    validate_cli(["tiny", "--checks", "onset_tau", *roots, "--no-plots"])
    assert f"[tiny] no verdict: reps 1-2 not validated with {', '.join(registry)}" in capsys.readouterr().out
    assert not layout.scenario_verdict("t", "tiny").exists()


def test_validating_a_scenario_alone_keeps_the_folder_verdict_only_while_it_matches(project, capsys) -> None:
    cfg, roots, layout = project
    validate_cli(["t", *roots, "--no-plots"])
    assert layout.folder_verdict("t").exists()
    validate_cli(["tiny", "--rep", "2", "--force", *roots, "--no-plots"])
    assert layout.folder_verdict("t").exists(), "same build and reps: the folder verdict still describes them"
    manifest = _rep(cfg, 2).path(layout, RepArtifact.RUN_MANIFEST)
    saved = manifest.read_bytes()
    manifest.unlink()
    capsys.readouterr()
    try:
        validate_cli(["tiny", *roots, "--no-plots"])
    finally:
        manifest.write_bytes(saved)
    out = capsys.readouterr().out
    assert "[tiny] verdict insufficient: no test decided over 1 rep(s)" in out
    assert "[t] folder verdict removed; rerun `simace validate t`" in out
    assert not layout.folder_verdict("t").exists()


def test_the_plots_and_atlas_are_rebuilt_after_the_verdict(project, capsys) -> None:
    """The onset-τ page reads the verdict, so validate rebuilds the plots under the lock it holds (ADR 0028 §6)."""
    _, roots, layout = project
    plots = layout.scenario_plots("t", "tiny")
    shutil.rmtree(plots, ignore_errors=True)
    validate_cli(["tiny", *roots])
    out = capsys.readouterr().out
    assert out.index("[tiny] verdict insufficient") < out.index("[tiny] plot started")
    assert "[tiny] atlas finished" in out
    assert (plots / "onset_tau.png").exists()
    assert (plots / "atlas.html").exists()
    assert layout.scenario_plots_manifest("t", "tiny").exists()


def test_no_plots_skips_the_rebuild(project, capsys) -> None:
    _, roots, layout = project
    plots = layout.scenario_plots("t", "tiny")
    shutil.rmtree(plots, ignore_errors=True)
    validate_cli(["tiny", "--no-plots", *roots])
    assert "plot started" not in capsys.readouterr().out
    assert not plots.exists()


def test_plots_wait_for_every_rep_to_be_complete(project, capsys) -> None:
    cfg, roots, layout = project
    manifest = _rep(cfg, 2).path(layout, RepArtifact.RUN_MANIFEST)
    saved = manifest.read_bytes()
    manifest.unlink()
    try:
        validate_cli(["tiny", *roots])
    finally:
        manifest.write_bytes(saved)
    out = capsys.readouterr().out
    assert "[tiny] plots skipped: rep2 not complete" in out
    assert "plot started" not in out


def test_a_dry_run_prints_the_plot_commands_with_the_verdict(project, capsys) -> None:
    _, roots, layout = project
    validate_cli(["tiny", "--dry-run", *roots])
    out = capsys.readouterr().out
    assert f"--verdict {layout.scenario_verdict('t', 'tiny')}" in out
    assert " atlas " in out
    validate_cli(["tiny", "--dry-run", "--no-plots", *roots])
    assert "--verdict" not in capsys.readouterr().out


def test_plots_that_already_show_the_verdict_are_not_rebuilt(project, capsys) -> None:
    _, roots, layout = project
    validate_cli(["tiny", *roots])
    atlas = layout.scenario_plots("t", "tiny") / "atlas.html"
    built = atlas.stat().st_mtime_ns
    capsys.readouterr()
    validate_cli(["tiny", *roots])
    out = capsys.readouterr().out
    assert "[tiny] plots current" in out
    assert "plot started" not in out
    assert atlas.stat().st_mtime_ns == built


def test_a_verdict_change_after_a_gather_says_to_gather_again(project, capsys) -> None:
    """``verdicts.tsv`` and the validation atlas are gather's snapshot; validate says when it has made them old."""
    _, roots, layout = project
    validate_cli(["t", *roots, "--no-plots"])
    layout.folder_verdicts("t").write_text("gathered\n")
    capsys.readouterr()
    try:
        validate_cli(["t", *roots, "--no-plots"])
        assert "rerun `simace gather t`" not in capsys.readouterr().out, "no verdict changed"
        validate_cli(["t", "--rep", "1", "--force", *roots, "--no-plots"])
        assert "rerun `simace gather t`" not in capsys.readouterr().out, "the same verdict again is no change"
        layout.scenario_verdict("t", "tiny").write_text("by hand\n")
        validate_cli(["t", *roots, "--no-plots"])
        out = capsys.readouterr().out
        assert "[t] verdicts changed since verdicts.tsv was gathered; rerun `simace gather t`" in out
    finally:
        layout.folder_verdicts("t").unlink()


def _family(verdict: str):
    """A family builder, in place of the real rules, whose one test decides ``verdict``."""
    statistic = {"pass": MeanZ(np.zeros(30)), "fail": MeanZ(np.full(30, 5.0)), "insufficient": Insufficient("few")}
    return lambda *args: [Entry(("onset_tau", "mean_z"), statistic[verdict])]


@pytest.fixture
def force_verdicts(project, monkeypatch):
    """Make the scenario and folder families decide given verdicts; remove the verdict files they write after."""
    import simace.cli.validate as validate_module

    _, _, layout = project

    def force(scenario: str, folder: str = "pass") -> None:
        monkeypatch.setattr(validate_module, "scenario_family", _family(scenario))
        monkeypatch.setattr(validate_module, "folder_family", _family(folder))

    yield force
    layout.scenario_verdict("t", "tiny").unlink(missing_ok=True)
    layout.folder_verdict("t").unlink(missing_ok=True)


def _exit_code(argv: list[str]) -> int | str | None:
    try:
        validate_cli(argv)
    except SystemExit as exc:
        return exc.code
    return 0


@pytest.mark.parametrize(
    ("target", "scenario", "folder", "code"),
    [
        pytest.param("t", "fail", "pass", 0, id="scenario-fail-inside-a-passing-folder"),
        pytest.param("t", "pass", "fail", 3, id="folder-fail"),
        pytest.param("tiny", "fail", "pass", 3, id="scenario-target-fail"),
        pytest.param("t", "insufficient", "insufficient", 0, id="folder-insufficient"),
        pytest.param("tiny", "insufficient", "insufficient", 0, id="scenario-insufficient"),
    ],
)
def test_a_targets_own_fail_verdict_exits_3(project, capsys, force_verdicts, target, scenario, folder, code) -> None:
    """Only the target's own verdict sets the exit code (ADR 0028 §6): a scenario in a folder is printed only."""
    _, roots, _ = project
    force_verdicts(scenario, folder)
    assert _exit_code([target, *roots, "--no-plots"]) == code
    captured = capsys.readouterr()
    out = captured.out
    assert ("verdict fail for" in captured.err) == (code == 3)
    assert f"[tiny] verdict {scenario}: " in out
    if target == "t":
        assert f"[t] folder verdict {folder}: " in out


@pytest.mark.slow
def test_a_rep_error_exits_1_over_another_targets_fail(project, capsys, monkeypatch, force_verdicts) -> None:
    import simace.cli.validate as validate_module

    cfg, roots, _ = project
    (cfg / "u.yaml").write_text(yaml.safe_dump({"tiny_u": {**TINY, "replicates": 1, "seed": 200}}))
    try:
        run_cli(["tiny_u", "--no-plots", *roots])
        validate_rep = validate_module._validate_rep
        monkeypatch.setattr(
            validate_module,
            "_validate_rep",
            lambda rep, *args: "ERROR (exit 1); log: x" if rep.scenario == "tiny_u" else validate_rep(rep, *args),
        )
        force_verdicts("fail")
        assert _exit_code(["tiny", "tiny_u", *roots, "--no-plots"]) == 1
    finally:
        (cfg / "u.yaml").unlink()
    captured = capsys.readouterr()
    out = captured.out
    assert "simace validate: verdict fail for tiny" in captured.err
    assert "[tiny] verdict fail: " in out
    assert "[tiny_u] no verdict: rep1 failed validation" in out
