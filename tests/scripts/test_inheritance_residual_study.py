"""``scripts/inheritance_residual_study.py``: the summary counts each draw × cell once and drops no cell."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from inheritance_residual_study import _COLUMNS, _MIN_SCENARIO_REPS, summarize


def _table(*specs: tuple[str, int, str, str, list[float | None]]) -> pl.DataFrame:
    """Rows from (scenario, trait, cell, draw-hash prefix, z per rep) specs; rep i carries draw ``{prefix}{i}``."""
    rows = []
    for scenario, trait, cell, prefix, zs in specs:
        for rep, z in enumerate(zs, 1):
            row = dict.fromkeys(_COLUMNS)
            row.update(folder="f", scenario=scenario, rep=rep, trait=trait, cell=cell, z=z)
            row.update(pedigree_hash=f"p{rep}", draw_hash=f"{prefix}{rep}")
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).select(_COLUMNS)


def _zs(n: int, offset: float = 0.0) -> list[float | None]:
    return [offset + 0.1 * (i % 7) - 0.3 for i in range(n)]


def _by_cell(summary: pl.DataFrame) -> dict[str, dict]:
    return {row["cell"]: row for row in summary.iter_rows(named=True)}


def test_the_same_draws_under_another_trait_count_once():
    n = _MIN_SCENARIO_REPS
    summary = _by_cell(summarize(_table(("s1", 1, "A_FS", "d", _zs(n)), ("s2", 2, "A_FS", "d", _zs(n)))))
    assert summary["A_FS"]["n"] == n


def test_other_draws_on_the_same_pedigrees_all_count():
    n = _MIN_SCENARIO_REPS
    summary = _by_cell(summarize(_table(("s1", 2, "A_FS", "d", _zs(n)), ("s2", 2, "A_FS", "e", _zs(n, 0.05)))))
    assert summary["A_FS"]["n"] == 2 * n


def test_a_cell_only_a_later_scenario_scores_survives():
    n = _MIN_SCENARIO_REPS
    summary = _by_cell(summarize(_table(("s1", 1, "A_FS", "d", _zs(n)), ("s2", 2, "C_PHS", "c", _zs(n)))))
    assert summary["C_PHS"]["n"] == n


def test_a_scored_row_wins_over_an_earlier_unscored_one():
    summary = _by_cell(summarize(_table(("s1", 1, "A_FS", "d", [None, None]), ("s2", 1, "A_FS", "d", [0.5, None]))))
    assert (summary["A_FS"]["n"], summary["A_FS"]["unscored"]) == (1, 1)


def test_a_cell_below_the_rep_floor_keeps_its_row_without_spread():
    n = _MIN_SCENARIO_REPS
    summary = _by_cell(summarize(_table(("s1", 1, "A_FS", "d", _zs(n)), ("s1", 1, "A_variance", "d", _zs(n - 1)))))
    assert summary["A_variance"]["n"] == n - 1
    assert summary["A_variance"]["sd_z_min"] is None
    assert summary["A_FS"]["sd_z_min"] is not None
