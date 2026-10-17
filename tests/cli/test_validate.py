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
    validate_cli(["tiny", *roots])
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
        validate_cli(["tiny", "--rep", "2", *roots])
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
    validate_cli(["tiny", "--rep", "1", "--dry-run", *roots])
    out = capsys.readouterr().out
    assert "validate --pedigree" in out
    assert "--checks onset_tau" in out
    assert "--phenotype-params1" in out
    assert not target.exists()


def test_a_recompute_removes_the_reps_validation_yaml_and_both_verdicts(project) -> None:
    cfg, roots, layout = project
    validate_cli(["t", *roots])
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
    validate_cli(["tiny", *roots])
    target = _rep(cfg, 1).path(layout, RepArtifact.VALIDATION)
    written = target.stat().st_mtime_ns
    capsys.readouterr()
    validate_cli(["tiny", *roots])
    assert "[tiny] validated 0 rep(s), 2 already current, 0 errors" in capsys.readouterr().out
    assert target.stat().st_mtime_ns == written
    validate_cli(["tiny", "--rep", "1", "--force", *roots])
    assert "[tiny] validated 1 rep(s), 0 already current, 0 errors" in capsys.readouterr().out
    assert target.stat().st_mtime_ns != written


def test_two_reps_give_an_insufficient_verdict_for_the_scenario_and_folder(project, capsys) -> None:
    _, roots, layout = project
    validate_cli(["t", *roots])
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
    validate_cli(["tiny", *roots])
    assert layout.scenario_verdict("t", "tiny").exists()
    path = _rep(cfg, 2).path(layout, RepArtifact.VALIDATION)
    data = yaml.safe_load(path.read_text())
    data["simace_version"] = "0.0.1"
    path.write_text(yaml.safe_dump(data))
    capsys.readouterr()
    validate_cli(["tiny", "--rep", "1", *roots])
    out = capsys.readouterr().out
    assert "[tiny] no verdict: reps validated by more than one build: 0.0.1 at " in out
    assert " (rep2); " in out
    assert not layout.scenario_verdict("t", "tiny").exists()
    validate_cli(["tiny", *roots])
    out = capsys.readouterr().out
    assert "[tiny] validated 1 rep(s), 1 already current, 0 errors" in out
    assert "[tiny] verdict insufficient" in out


def test_overlapping_seed_ranges_refuse_only_the_pooled_tests(project, capsys, monkeypatch) -> None:
    import simace.cli.validate as validate_module

    _, roots, layout = project
    monkeypatch.setattr(validate_module, "overlapping_seed_ranges", lambda ranges: [("tiny", "other")])
    validate_cli(["t", *roots])
    out = capsys.readouterr().out
    assert "[t] folder verdict insufficient" in out
    assert "; pooling refused, rep seed ranges overlap: tiny and other" in out
    folder = yaml.safe_load(layout.folder_verdict("t").read_text())
    assert folder["pooled"]["onset_tau"] == {"insufficient": "rep seed ranges overlap: tiny and other"}
    assert "onset_tau" in folder["scenarios"]["tiny"]


def test_a_folder_target_with_no_scenario_to_validate_has_no_verdict(project) -> None:
    import simace.cli.validate as validate_module

    _, _, layout = project
    assert validate_module._folder_verdict("gene_drop_only", [], {}, layout) == "no verdict: no scenario to validate"


def test_a_failed_rep_leaves_no_verdict_beside_its_old_file(project, capsys, monkeypatch) -> None:
    import simace.cli.validate as validate_module

    _, roots, layout = project
    validate_cli(["tiny", *roots])
    assert layout.scenario_verdict("t", "tiny").exists()
    monkeypatch.setattr(validate_module, "_validate_rep", lambda *args: "ERROR (exit 1); log: x")
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        validate_cli(["tiny", "--rep", "1", "--force", *roots])
    assert exc.value.code == 1
    assert "[tiny] no verdict: rep1 failed validation" in capsys.readouterr().out
    assert not layout.scenario_verdict("t", "tiny").exists()


def test_a_rep_from_another_checkout_state_is_validated_again(project, capsys) -> None:
    """The build is the version and the ``git describe`` source without ``-dirty``: a commit without a reinstall counts."""
    from simace.cli.manifest import source_ref

    cfg, roots, layout = project
    validate_cli(["tiny", *roots])
    path = _rep(cfg, 1).path(layout, RepArtifact.VALIDATION)
    data = yaml.safe_load(path.read_text())
    assert data["source"] == source_ref()
    source = data["source"]
    data["source"] = source.removesuffix("-dirty") if source.endswith("-dirty") else source + "-dirty"
    path.write_text(yaml.safe_dump(data))
    capsys.readouterr()
    validate_cli(["tiny", *roots])
    assert "[tiny] validated 0 rep(s), 2 already current, 0 errors" in capsys.readouterr().out, "-dirty is ignored"
    data["source"] = "v0.0.0-1-gdeadbee"
    path.write_text(yaml.safe_dump(data))
    validate_cli(["tiny", *roots])
    assert "[tiny] validated 1 rep(s), 1 already current, 0 errors" in capsys.readouterr().out


def test_a_verdict_needs_every_registered_test_whatever_checks_ran(project, capsys, monkeypatch) -> None:
    import simace.cli.validate as validate_module

    _, roots, layout = project
    validate_cli(["tiny", *roots])
    assert layout.scenario_verdict("t", "tiny").exists()
    registry = dict(validate_module.ON_DEMAND_CHECKS)
    registry["other"] = registry["onset_tau"]
    monkeypatch.setattr(validate_module, "ON_DEMAND_CHECKS", registry)
    capsys.readouterr()
    validate_cli(["tiny", "--checks", "onset_tau", *roots])
    assert "[tiny] no verdict: reps 1-2 not validated with onset_tau, other" in capsys.readouterr().out
    assert not layout.scenario_verdict("t", "tiny").exists()


def test_validating_a_scenario_alone_keeps_the_folder_verdict_only_while_it_matches(project, capsys) -> None:
    cfg, roots, layout = project
    validate_cli(["t", *roots])
    assert layout.folder_verdict("t").exists()
    validate_cli(["tiny", "--rep", "2", "--force", *roots])
    assert layout.folder_verdict("t").exists(), "same build and reps: the folder verdict still describes them"
    manifest = _rep(cfg, 2).path(layout, RepArtifact.RUN_MANIFEST)
    saved = manifest.read_bytes()
    manifest.unlink()
    capsys.readouterr()
    try:
        validate_cli(["tiny", *roots])
    finally:
        manifest.write_bytes(saved)
    out = capsys.readouterr().out
    assert "[tiny] verdict insufficient: no test decided over 1 rep(s)" in out
    assert "[t] folder verdict removed; rerun `simace validate t`" in out
    assert not layout.folder_verdict("t").exists()
