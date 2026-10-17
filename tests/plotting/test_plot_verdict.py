"""The validation atlas's verdict page shows the folder verdict `simace validate` decided (ADR 0028 §6)."""

from __future__ import annotations

import pytest

from simace.analysis.validate.verdict import decide, folder_family
from simace.plotting.plot_verdict import headline, plot_verdict
from tests.analysis.test_validate_verdict import TESTS, _reps

BUILD = {"simace_version": "0.1.0", "source": "v0.1.0-1-gabc1234", "family": "folder", "folder": "f"}


def _folder(**reps_by_scenario: list[dict]) -> dict:
    by_scenario = {name: {"onset_tau": reps} for name, reps in reps_by_scenario.items()}
    return {**BUILD, **decide(folder_family(by_scenario, TESTS))}


def _shifted(n: int) -> list[dict]:
    reps = _reps(n, seed=3)
    for section in reps:
        section["cells"]["trait2_MO"]["z"] += 2.0
    return reps


def test_the_headline_gives_the_family_and_the_failed_tests() -> None:
    lines = headline(_folder(a=_reps(20, seed=1), b=_shifted(20)))
    assert lines[0] == "Folder verdict: fail. 80 tests at α = 6.25e-04 each (familywise 5%)"  # noqa: RUF001
    assert "b/onset_tau/trait2_MO/mean_z" in lines[2]
    assert lines[2].startswith("Failed: ")


def test_the_headline_names_the_build_that_decided_it() -> None:
    """A page gathered before a later `simace validate` shows which build its verdict is from."""
    assert headline(_folder(a=_reps(5)))[1] == "Decided by simace 0.1.0 at v0.1.0-1-gabc1234"


def test_a_passing_headline_says_no_test_failed() -> None:
    assert headline(_folder(a=_reps(20, seed=1), b=_reps(20, seed=4)))[2] == "No test failed"


def test_insufficient_tests_are_named_with_their_reason() -> None:
    lines = headline(_folder(a=_reps(5)))
    assert lines[0] == "Folder verdict: insufficient. No test decided"
    assert lines[-1] == "Insufficient: a/onset_tau: 5 rep(s); the verdict needs 20; pooled/onset_tau: a below 20 rep(s)"


@pytest.mark.parametrize("case", ["none", "pass", "fail", "insufficient"])
def test_the_page_renders(tmp_path, case) -> None:
    verdict = {
        "none": None,
        "pass": _folder(a=_reps(20, seed=1), b=_reps(20, seed=4)),
        "fail": _folder(a=_reps(20, seed=1), b=_shifted(20)),
        "insufficient": _folder(a=_reps(5)),
    }[case]
    plot_verdict(verdict, tmp_path, "png")
    assert (tmp_path / "verdict.png").stat().st_size > 0


def test_a_series_without_cells_gets_no_slot() -> None:
    """Refused pooling leaves ``pooled`` insufficient: no dodge slot or legend marker without data behind it."""
    from simace.plotting.plot_verdict import _by_test

    by_scenario = {"a": {"onset_tau": _reps(20, seed=1)}, "b": {"onset_tau": _reps(5)}}
    verdict = {**BUILD, **decide(folder_family(by_scenario, TESTS, pooling_refused="seed ranges overlap"))}
    assert list(_by_test(verdict)["onset_tau"]) == ["a"]
