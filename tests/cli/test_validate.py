"""``simace validate``: on-demand checks over a scenario's complete reps (ADR 0027, 0028)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from simace.analysis.validate.on_demand import SCHEMA
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
    assert "[tiny] validated 2 rep(s), 0 errors" in out
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
    assert "validated" not in out


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


def test_a_recompute_removes_the_reps_validation_yaml(project) -> None:
    cfg, roots, layout = project
    validate_cli(["tiny", "--rep", "1", *roots])
    target = _rep(cfg, 1).path(layout, RepArtifact.VALIDATION)
    assert target.exists()
    run_cli(["tiny", "--rep", "1", "--force", "--no-plots", *roots])
    assert not target.exists()


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
