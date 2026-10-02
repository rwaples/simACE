"""``tools/verify_release.py`` reports a missing artifact as a failed check instead of crashing."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import verify_release


def test_missing_distribution_is_a_failed_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_release, "DISTRIBUTIONS", ("no-such-family-dist", "simace"))
    results = verify_release.check_installed("0.0.0-not-this")
    out = capsys.readouterr().out
    assert "FAIL no-such-family-dist distribution: <not installed>" in out
    assert "simace.__version__" in out
    assert results[0] is False
    assert len(results) > 1


def test_unimportable_package_is_a_failed_check() -> None:
    assert verify_release._package_version("no_such_family_package").startswith("<ModuleNotFoundError")


def test_missing_provenance_file_is_a_failed_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stamped = tmp_path / "params.yaml"
    stamped.write_text("simace_version: 0.1.0\n")
    results = verify_release.check_provenance("0.1.0", [tmp_path / "missing.meta", stamped])
    out = capsys.readouterr().out
    assert results == [False, True]
    assert f"FAIL {tmp_path / 'missing.meta'}: <No such file or directory>" in out


def test_main_summarises_after_a_missing_provenance_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert verify_release.main(["0.0.0-not-this", "--provenance", str(tmp_path / "missing.meta")]) == 1
    assert "checks report 0.0.0-not-this" in capsys.readouterr().out
