"""The expected-slope line on the PO heritability by sex and generation plot (ADR 0025)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pytest

from simace.plotting import plot_heritability

if TYPE_CHECKING:
    from pathlib import Path

_PARAMS = {"A1": 0.5, "C1": 0.2, "E1": 0.3, "A2": 0.5, "C2": 0.2, "E2": 0.3, "assort1": 0.0, "assort2": 0.0}


def _stats(truth_by_gen: dict[int, float] | None = None) -> list[dict[str, Any]]:
    fit = {"slope": 0.6}
    by_sex = {sex: {f"trait{t}": {"gen2": fit, "gen3": fit} for t in (1, 2)} for sex in ("female", "male")}
    stats: dict[str, Any] = {"parent_offspring_corr_by_sex": by_sex}
    if truth_by_gen is not None:
        entry = {"slope": 0.0, "by_generation": {g: {"slope": v} for g, v in truth_by_gen.items()}}
        stats["truth_midparent_slope"] = {"trait1": entry, "trait2": entry}
    return [stats]


def _expected_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stats, params) -> list[tuple[list, list]]:
    """``(xdata, ydata)`` of the dashed lines on trait 1's panel."""
    figures = []
    monkeypatch.setattr(plot_heritability, "finalize_plot", lambda *a, **k: figures.append(plt.gcf()))
    plot_heritability.plot_heritability_by_sex_generation(stats, tmp_path / "p.png", params=params)
    ax = figures[0].axes[0]
    lines = [(list(line.get_xdata()), list(line.get_ydata())) for line in ax.lines if line.get_linestyle() == "--"]
    plt.close(figures[0])
    return lines


def test_without_transmission_the_line_is_a_over_vp(monkeypatch, tmp_path):
    [(_, y)] = _expected_lines(monkeypatch, tmp_path, _stats(), _PARAMS)
    assert y == pytest.approx([0.5, 0.5])


def test_closed_form_transmission_moves_the_line(monkeypatch, tmp_path):
    params = {**_PARAMS, "Ctm_maternal1": 0.5, "Ctm_paternal1": 0.5}
    [(_, y)] = _expected_lines(monkeypatch, tmp_path, _stats(), params)
    assert y == pytest.approx([0.5 + 1.0 * 0.2, 0.5 + 1.0 * 0.2])


def test_outside_the_closed_form_each_generation_reads_its_truth(monkeypatch, tmp_path):
    params = {**_PARAMS, "Ctm_maternal1": 0.5, "Ctm_paternal1": 0.5, "assort1": 0.3}
    [(x, y)] = _expected_lines(monkeypatch, tmp_path, _stats({2: 0.66, 3: 0.71}), params)
    assert (x, y) == ([2, 3], pytest.approx([0.66, 0.71]))
