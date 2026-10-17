"""The onset-τ page states the scenario's verdict beside the per-rep z (ADR 0028 §6)."""

from __future__ import annotations

import pytest

from simace.analysis.validate.verdict import decide, scenario_family
from simace.plotting.plot_onset_tau import plot_onset_tau, verdict_note
from tests.analysis.test_validate_onset_tau import _cell, _summary
from tests.analysis.test_validate_verdict import TESTS, _reps


def _verdict(reps: list[dict]) -> dict:
    return {"family": "scenario", "reps": f"1-{len(reps)}", **decide(scenario_family({"onset_tau": reps}, TESTS))}


def test_a_passing_verdict_names_its_family_and_the_reps_over_their_bound() -> None:
    note = verdict_note(_verdict(_reps(100)))
    assert note.startswith("Scenario verdict: pass")
    assert "27 tests at α = 1.85e-03 each (familywise 5%) over reps 1-100" in note  # noqa: RUF001
    assert "of 100 replicates over their per-rep bound (" in note
    assert "allowed)" in note


def test_a_failing_verdict_lists_the_failed_tests() -> None:
    reps = _reps(100)
    for section in reps:
        section["cells"]["trait1_FS"]["z"] += 1.0
    note = verdict_note(_verdict(reps))
    assert note.startswith("Scenario verdict: fail")
    assert "failed: onset_tau/trait1_FS/mean_z" in note


def test_an_insufficient_verdict_gives_its_reason() -> None:
    assert verdict_note(_verdict(_reps(5))) == "Scenario verdict: insufficient (5 rep(s); the verdict needs 20)"


def test_no_verdict_says_how_to_get_one() -> None:
    assert verdict_note(None) == "Scenario verdict: none yet; `simace validate` decides it across replicates"


def test_no_note_calls_a_replicate_failed() -> None:
    """About 5% of correct reps exceed their bound; only the across-rep verdict passes or fails."""
    for verdict in (None, _verdict(_reps(100)), _verdict(_reps(5))):
        assert "failed" not in verdict_note(verdict)


@pytest.mark.parametrize("verdict", [None, "pass", "insufficient"])
def test_the_page_renders_with_each_verdict(tmp_path, verdict) -> None:
    from simace.analysis.validate.onset_tau import score_onset_tau

    decided = {None: None, "pass": _verdict(_reps(100)), "insufficient": _verdict(_reps(5))}[verdict]
    section = score_onset_tau(_summary(FS=_cell(observed=0.068 + 0.002 * 5.0), MO=_cell(), MZ=_cell(n_eff=40.0)))
    out = tmp_path / "onset_tau.png"
    plot_onset_tau([{"onset_tau": section}, {}], out, verdict=decided)
    assert out.stat().st_size > 0


def test_a_long_note_grows_the_figure_instead_of_squeezing_the_panel(tmp_path) -> None:
    from PIL import Image

    from simace.analysis.validate.onset_tau import score_onset_tau

    reps = _reps(100)
    for section in reps:
        for cell in section["cells"].values():
            cell["z"] += 1.0
    section = score_onset_tau(_summary(FS=_cell(), MO=_cell()))
    section["cells"] = {k: v for k, v in section["cells"].items() if k.startswith("trait1_")}
    short, long = tmp_path / "short.png", tmp_path / "long.png"
    plot_onset_tau([{"onset_tau": section}], short, verdict=_verdict(_reps(100)))
    plot_onset_tau([{"onset_tau": section}], long, verdict=_verdict(reps))
    assert Image.open(long).height > Image.open(short).height
