"""The generic calibrated-test page: per-rep z of every test with no page of its own (ADR 0028, 0029)."""

from __future__ import annotations

import numpy as np
import pytest

from simace.analysis.validate.calibrated import not_applicable_cell, scored_cell, summarize
from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS
from simace.analysis.validate.verdict import decide, scenario_family
from simace.plotting.plot_calibrated_tests import PAGE_TESTS, _cells_reported, plot_calibrated_tests
from simace.plotting.plot_onset_tau import verdict_note


def _stats(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        mendelian = {cell: scored_cell(float(rng.standard_normal()), 0.0, 1.0) for cell in ("trait1_FS", "trait1_PHS")}
        household = {"trait1_PHS": scored_cell(float(rng.standard_normal()), 0.0, 1.0)}
        household["trait2_PHS"] = not_applicable_cell("zero variance")
        out.append(
            {
                "mendelian_residual": {"cells": mendelian, "summary": summarize(mendelian)},
                "household_draw": {"cells": household, "summary": summarize(household)},
            }
        )
    return out


def _verdict(stats: list[dict]) -> dict:
    sections = {name: [s[name] for s in stats] for name in PAGE_TESTS}
    tests = {name: ON_DEMAND_CHECKS[name] for name in PAGE_TESTS}
    return {"family": "scenario", "reps": f"1-{len(stats)}", **decide(scenario_family(sections, tests))}


def test_the_page_covers_every_test_but_onset_tau() -> None:
    assert set(PAGE_TESTS) == set(ON_DEMAND_CHECKS) - {"onset_tau"}
    assert {"mendelian_residual", "household_draw"} <= set(PAGE_TESTS)


def test_cells_follow_the_tests_order_and_skip_unreported() -> None:
    assert _cells_reported([s["mendelian_residual"] for s in _stats(3)], "mendelian_residual") == [
        "trait1_FS",
        "trait1_PHS",
    ]


def test_the_note_counts_each_tests_reps_over_bound_by_name() -> None:
    note = verdict_note(_verdict(_stats(30)), PAGE_TESTS)
    assert note.startswith("Scenario verdict: pass")
    assert "mendelian_residual: " in note
    assert "household_draw: " in note


@pytest.mark.parametrize("verdict", [None, "pass"])
def test_the_page_renders(tmp_path, verdict) -> None:
    stats = _stats(30)
    out = tmp_path / "calibrated_tests.png"
    plot_calibrated_tests([*stats, {}], out, verdict=None if verdict is None else _verdict(stats))
    assert out.stat().st_size > 0


def test_unvalidated_reps_give_a_placeholder(tmp_path) -> None:
    out = tmp_path / "calibrated_tests.png"
    plot_calibrated_tests([{}, {}], out)
    assert out.stat().st_size > 0
