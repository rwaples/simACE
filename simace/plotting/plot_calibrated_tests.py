"""Per-rep z of every calibrated test with no page of its own (ADR 0028, 0029).

One panel per test, one column per cell the test can report. Each replicate's
scored z is a point; a red edge marks a z whose ``|z|`` exceeds the
replicate's per-rep Bonferroni bound, which about 5% of replicates of a
correct simulation show. The black bar is the cell's mean z; the band is
±1.96. Below each cell: the replicates that scored it over those where it
applies. Above the panels is the scenario's verdict from
``validation_summary.yaml``, which the page shows and never decides.
"""

from __future__ import annotations

__all__ = ["PAGE_TESTS", "plot_calibrated_tests"]

import textwrap
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS
from simace.analysis.validate.verdict import applicable_reps
from simace.plotting.plot_onset_tau import COLOR_BEYOND, verdict_note
from simace.plotting.plot_utils import finalize_plot, save_placeholder_plot

if TYPE_CHECKING:
    from pathlib import Path

#: The tests this page draws: every registered test without a page of its own.
PAGE_TESTS = tuple(name for name, test in ON_DEMAND_CHECKS.items() if not test.own_page)

_COLOR_POINT = "#4477AA"
#: Height of one line of the 9 pt verdict note, in inches.
_NOTE_LINE_IN = 0.17
_PANEL_HEIGHT_IN = 2.6
_MIN_COLUMNS = 4


def _cells_reported(sections: list[dict[str, Any]], name: str) -> list[str]:
    """The test's cells that some replicate reports, in the test's order."""
    seen = {cell for section in sections for cell in (section.get("cells") or {})}
    return [cell for cell in ON_DEMAND_CHECKS[name].cell_names if cell in seen]


def _draw_test(ax, sections: list[dict[str, Any]], name: str, cells: list[str], rng: np.random.Generator) -> None:
    ax.axhspan(-1.96, 1.96, color="0.92", zorder=0)
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)
    labels = []
    for i, cell in enumerate(cells):
        z, beyond = [], []
        for section in sections:
            row = (section.get("cells") or {}).get(cell)
            if row is None or row.get("status") != "scored":
                continue
            z.append(row["z"])
            bound = (section.get("summary") or {}).get("z_bound")
            beyond.append(bound is not None and abs(row["z"]) > bound)
        if z:
            x = i + rng.uniform(-0.18, 0.18, len(z))
            edges = [COLOR_BEYOND if b else _COLOR_POINT for b in beyond]
            ax.scatter(x, z, s=14, color=_COLOR_POINT, edgecolors=edges, linewidths=1.0, alpha=0.8, zorder=3)
            ax.hlines(float(np.mean(z)), i - 0.3, i + 0.3, color="black", linewidth=2, zorder=4)
        trait, _, code = cell.partition("_")
        labels.append(f"{trait}\n{code}\n{len(z)}/{applicable_reps(sections, cell)}")
    ax.set_xticks(range(len(cells)), labels, fontsize=8)
    # At least _MIN_COLUMNS columns wide, centred, so one cell keeps its width.
    half = max(len(cells), _MIN_COLUMNS) / 2
    ax.set_xlim((len(cells) - 1) / 2 - half, (len(cells) - 1) / 2 + half)
    ax.set_ylabel("z")
    ax.set_title(f"{name}: {ON_DEMAND_CHECKS[name].description}", fontsize=9)


def plot_calibrated_tests(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    verdict: dict[str, Any] | None = None,
) -> None:
    """Per-rep z of each test in :data:`PAGE_TESTS`, one panel per test, under the scenario verdict."""
    panels = []
    for name in PAGE_TESTS:
        sections = [stats.get(name) or {} for stats in all_stats]
        if cells := _cells_reported(sections, name):
            panels.append((name, sections, cells))
    if not panels:
        save_placeholder_plot(output_path, "No calibrated test reported: replicates not yet validated")
        return

    note = textwrap.wrap(verdict_note(verdict, [name for name, _, _ in panels]), width=110)
    height = _PANEL_HEIGHT_IN * len(panels) + 0.5 + _NOTE_LINE_IN * len(note)
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, height), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (name, sections, cells) in zip(axes[:, 0], panels, strict=True):
        _draw_test(ax, sections, name, cells, rng)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=_COLOR_POINT, label="replicate z"),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=_COLOR_POINT,
            markeredgecolor=COLOR_BEYOND,
            label="|z| > per-rep bound",
        ),
        plt.Line2D([], [], color="black", linewidth=2, label="mean z"),
        plt.Rectangle((0, 0), 1, 1, color="0.92", label="±1.96"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.text(0.5, 0.995, "\n".join(note), ha="center", va="top", fontsize=9)
    top = 1 - (_NOTE_LINE_IN * len(note) + 0.1) / height
    finalize_plot(output_path, tight_rect=[0, 0.35 / height, 1, top], scenario=scenario)
