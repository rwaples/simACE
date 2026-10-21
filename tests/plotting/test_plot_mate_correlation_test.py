"""The mate_correlation test's own page: per-rep z by cell type and parents' generation (ADR 0031)."""

from __future__ import annotations

import numpy as np
import pytest

from simace.analysis.validate.calibrated import not_applicable_cell, scored_cell, summarize
from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS
from simace.analysis.validate.verdict import decide, scenario_family
from simace.plotting.plot_mate_correlation_test import plot_mate_correlation_test


def _checks(n: int, generations: int = 14) -> list[dict]:
    """Reps of a mixed scenario: trait 2 induced in generation 0, both traits assorting after, past the column cap."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        cells = {"g0_liability1": scored_cell(float(rng.standard_normal()), 0.0, 1.0)}
        cells["g0_liability2"] = not_applicable_cell("induced by assortment on trait 1")
        for g in range(1, generations):
            for label in ("liability1", "liability2", "cross_12", "cross_21"):
                cells[f"g{g}_{label}"] = scored_cell(float(rng.standard_normal()), 0.0, 1.0)
        out.append({"mate_correlation": {"cells": cells, "summary": summarize(cells)}})
    return out


def _verdict(checks: list[dict]) -> dict:
    sections = {"mate_correlation": [c["mate_correlation"] for c in checks]}
    return {"family": "scenario", "reps": f"1-{len(checks)}", **decide(scenario_family(sections, ON_DEMAND_CHECKS))}


@pytest.mark.parametrize("verdict", [None, "pass"])
def test_the_page_renders_every_reported_generation(tmp_path, verdict) -> None:
    checks = _checks(25)
    out = tmp_path / "mate_correlation_test.png"
    plot_mate_correlation_test([*checks, {}], out, verdict=None if verdict is None else _verdict(checks))
    assert out.stat().st_size > 0


@pytest.mark.parametrize(
    "checks",
    [
        [{}, {}],
        [{"mate_correlation": {"cells": {}, "summary": summarize({}), "not_applicable": "no assortative mating"}}],
    ],
)
def test_no_cells_give_a_placeholder(tmp_path, checks) -> None:
    out = tmp_path / "mate_correlation_test.png"
    plot_mate_correlation_test(checks, out)
    assert out.stat().st_size > 0
