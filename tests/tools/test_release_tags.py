"""The family SemVer tag rule in ``tools/release.py`` (ADR 0023)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import release
from release import main, next_versions, parse_family_tag, stated_version


@pytest.mark.parametrize(("tag", "expected"), [("v0.1.0", (0, 1, 0)), ("v0.10.3", (0, 10, 3)), ("v1.0.0", (1, 0, 0))])
def test_accepts_semver_tags(tag: str, expected: tuple[int, int, int]) -> None:
    assert parse_family_tag(tag) == expected


@pytest.mark.parametrize(
    "tag", ["v0.1", "v0.01.0", "v00.1.0", "v2026.10.0", "v2026.09.2", "2026.10", "v0.1.0-rc1", "v0.1.0\n"]
)
def test_refuses_non_family_tags(tag: str) -> None:
    assert parse_family_tag(tag) is None


def test_next_after_calver_only_is_first_semver() -> None:
    assert next_versions(["v2026.04.1", "v2026.09.2", "v2026.06"]) == ("v0.1.0", "v0.1.0")


def test_next_ignores_calver_in_a_mixed_history() -> None:
    assert next_versions(["v2026.09.2", "v2026.05.3", "v0.1.0", "v0.2.3"]) == ("v0.2.4", "v0.3.0")


def test_next_orders_numerically() -> None:
    assert next_versions(["v0.10.0", "v0.9.0"]) == ("v0.10.1", "v0.11.0")


@pytest.mark.parametrize(
    ("tag", "reason"),
    [("v2026.10", "vMAJOR.MINOR.PATCH"), ("v0.01.0", "leading zero"), ("v2026.10.0", "CalVer-era")],
)
def test_cli_refuses_with_reason(tag: str, reason: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([tag, "--dry-run"])
    assert exc.value.code == 2
    assert reason in capsys.readouterr().err


def _independent_checkout(root: Path, version: str, heading: str) -> None:
    repo = root / "external" / "pg-phenotype"
    (repo / "r" / "src" / "rust").mkdir(parents=True)
    (repo / "Cargo.toml").write_text(f'[workspace]\n\n[workspace.package]\nversion = "{version}"\n')
    (repo / "r" / "src" / "rust" / "Cargo.toml").write_text(f'[package]\nname = "x"\nversion = "{version}"\n')
    (repo / "r" / "DESCRIPTION").write_text(f"Package: x\nVersion: {version}\n")
    (repo / "CHANGELOG.md").write_text(f"# Changelog\n\n{heading}\n\nNotes.\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git[:3], "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)


def test_stated_version_reads_toml_and_dcf(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text('[workspace.package]\nversion = "1.2.3"\n')
    (tmp_path / "DESCRIPTION").write_text("Package: x\nVersion:   1.2.3\nTitle: y\n")
    assert stated_version(tmp_path / "a.toml", ("workspace", "package", "version")) == "1.2.3"
    assert stated_version(tmp_path / "a.toml", ("package", "version")) is None
    assert stated_version(tmp_path / "DESCRIPTION", ("Version",)) == "1.2.3"
    assert stated_version(tmp_path / "missing.toml", ("version",)) is None


def test_independent_repo_ready_when_versions_and_changelog_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _independent_checkout(tmp_path, "0.2.0", "## v0.2.0")
    monkeypatch.setattr(release, "_SIMACE_ROOT", tmp_path)
    assert release.check_independent("pg-phenotype", "v0.2.0") == []


def test_independent_repo_refuses_each_disagreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _independent_checkout(tmp_path, "0.1.0", "## Unreleased (0.2.0)")
    monkeypatch.setattr(release, "_SIMACE_ROOT", tmp_path)
    problems = dict(release.check_independent("pg-phenotype", "v0.2.0"))
    assert set(problems) == {
        "external/pg-phenotype/Cargo.toml",
        "external/pg-phenotype/r/src/rust/Cargo.toml",
        "external/pg-phenotype/r/DESCRIPTION",
        "external/pg-phenotype/CHANGELOG.md",
    }
