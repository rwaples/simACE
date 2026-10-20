"""``tools/host_check.py``: clock clamps and load findings."""

from __future__ import annotations

import pytest

from tools import host_check


def sysfs(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    def read(path: object) -> str | None:
        return next((v for k, v in values.items() if str(path).endswith(k)), None)

    monkeypatch.setattr(host_check, "read", read)
    monkeypatch.setattr(host_check, "loaded_mhz", lambda _cpu: int(values.get("scaling_cur_freq", "0")) / 1000)


QUIET = {
    "scaling_cur_freq": "4100000",
    "platform_profile": "balanced",
    "no_turbo": "0",
    "scaling_max_freq": "4500000",
    "cpuinfo_max_freq": "4500000",
}


def test_unclamped_clock_has_no_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    sysfs(monkeypatch, QUIET)
    assert host_check.clock_problems() == []


@pytest.mark.parametrize(
    ("key", "value", "expect"),
    [
        ("scaling_cur_freq", "800000", "PROCHOT"),
        ("platform_profile", "quiet", "quiet"),
        ("no_turbo", "1", "no_turbo"),
        ("scaling_max_freq", "2600000", "2600 MHz < hardware max 4500 MHz"),
    ],
)
def test_each_clamp_is_reported(monkeypatch: pytest.MonkeyPatch, key: str, value: str, expect: str) -> None:
    sysfs(monkeypatch, {**QUIET, key: value})
    (finding,) = host_check.clock_problems()
    assert expect in finding


def test_load_above_threshold_lists_busiest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_check.os, "getloadavg", lambda: (9.0, 9.0, 9.0))
    monkeypatch.setattr(host_check, "busiest", lambda: ["99.0%     123 R --slave"])
    assert host_check.load_problems(1.5) == ["load: 1-min load 9.0 > 1.5; busiest:", "  99.0%     123 R --slave"]
    monkeypatch.setattr(host_check.os, "getloadavg", lambda: (1.0, 1.0, 1.0))
    assert host_check.load_problems(1.5) == []
