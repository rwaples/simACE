"""The family SemVer tag rule in ``tools/release.py`` (ADR 0023)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from release import main, next_versions, parse_family_tag


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
