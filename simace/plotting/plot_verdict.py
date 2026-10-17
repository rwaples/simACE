"""The validation atlas's verdict page: the folder's calibrated-test verdict, as `simace validate` decided it.

The page reads ``results/{folder}/validation_summary.yaml`` and decides
nothing (ADR 0028 §6). For each test with cells, one panel per across-rep
statistic: each scenario's, then the pooled, mean z and SD z of every cell, as
a marker on its acceptance interval, and each scenario's count of reps over
their per-rep bound against the count allowed. A marker outside its interval
is a failed test and carries a red ring. Coverage is listed only when it
fails.
"""

from __future__ import annotations

__all__ = ["headline", "plot_verdict"]

import textwrap
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from simace.analysis.validate.verdict import VERDICT_ALPHA, record_name, records
from simace.plotting.plot_utils import finalize_plot, save_placeholder_plot

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from matplotlib.axes import Axes

COLOR_FAIL = "#CC3311"
COLOR_POOLED = "#222222"
#: One hue per scenario in a fixed order (validated for colour-vision deficiency); red is kept for failures.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
#: Past the last hue, scenarios repeat the hues with the next marker.
SERIES_MARKERS = ("o", "s", "^", "v", "P")
POOLED = "pooled"
#: Failed tests the headline names before it counts the rest.
_NAMED = 8


def headline(verdict: dict[str, Any]) -> list[str]:
    """The page's title lines: the verdict and its family, its build, the failed tests, and insufficient reasons.

    The build shows which `simace validate` a page gathered before a later one is from.
    """
    if verdict["n_tests"]:
        family = f"{verdict['n_tests']} tests at α = {verdict['alpha_each']:.2e} each (familywise {VERDICT_ALPHA:.0%})"  # noqa: RUF001
    else:
        family = "No test decided"
    found = list(records(verdict))
    failed = [record_name(path) for path, record in found if record.get("passed") is False]
    lines = [
        f"Folder verdict: {verdict['verdict']}. {family}",
        f"Decided by simace {verdict['simace_version']} at {verdict['source']}",
    ]
    if failed:
        more = f" and {len(failed) - _NAMED} more" if len(failed) > _NAMED else ""
        lines.append(f"Failed: {', '.join(failed[:_NAMED])}{more}")
    elif verdict["n_tests"]:
        lines.append("No test failed")
    reasons = [f"{record_name(path)}: {record['insufficient']}" for path, record in found if "insufficient" in record]
    if reasons:
        lines.append(f"Insufficient: {'; '.join(reasons)}")
    return lines


def _by_test(verdict: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per test, the records of each series with cells: each scenario in file order, then the pooled.

    An insufficient series has no cells, so it gets no slot; the headline gives its reason.
    """
    by_test: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario, tests in (verdict.get("scenarios") or {}).items():
        for test, node in tests.items():
            by_test.setdefault(test, {})[scenario] = node
    for test, node in (verdict.get("pooled") or {}).items():
        by_test.setdefault(test, {})[POOLED] = node
    with_cells = {
        test: {name: node for name, node in series.items() if "cells" in node} for test, series in by_test.items()
    }
    return {test: series for test, series in with_cells.items() if series}


def _style(index: int, name: str) -> tuple[str, str]:
    if name == POOLED:
        return COLOR_POOLED, "D"
    return SERIES_COLORS[index % len(SERIES_COLORS)], SERIES_MARKERS[index // len(SERIES_COLORS) % len(SERIES_MARKERS)]


def _mark(ax: Axes, x: float, y: float, color: str, marker: str, passed: bool) -> None:
    edge, width = (COLOR_FAIL, 1.8) if not passed else ("white", 0.6)
    ax.scatter(x, y, marker=marker, s=34, color=color, edgecolors=edge, linewidths=width, zorder=3)


def _cell_panel(
    ax: Axes,
    series: dict[str, dict[str, Any]],
    statistic: str,
    value: str,
    interval: Callable[[dict[str, Any]], tuple[float, float]],
    reference: float,
) -> None:
    """One statistic of every cell: a marker per series on its acceptance interval, dodged within the cell."""
    cells = list(dict.fromkeys(cell for node in series.values() for cell in node["cells"]))
    width = 0.8 / len(series)
    for i, (name, node) in enumerate(series.items()):
        color, marker = _style(i, name)
        for j, cell in enumerate(cells):
            record = (node["cells"].get(cell) or {}).get(statistic)
            if not record or record.get(value) is None:
                continue
            x = j - 0.4 + width * (i + 0.5)
            ax.vlines(x, *interval(record), color=color, linewidth=2.5, alpha=0.35, zorder=2)
            _mark(ax, x, record[value], color, marker, record["passed"])
    ax.axhline(reference, color="0.6", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(cells)), cells, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(cells) - 0.4)


def _tail_panel(ax: Axes, series: dict[str, dict[str, Any]]) -> None:
    """Each scenario's reps over their per-rep bound, against the count allowed."""
    names = [name for name, node in series.items() if "reps_over_bound" in node]
    top = 1
    for i, name in enumerate(names):
        record = series[name]["reps_over_bound"]
        color, marker = _style(list(series).index(name), name)
        ax.scatter(i, record["allowed"], marker="_", s=400, color="0.4", linewidths=2, zorder=2)
        _mark(ax, i, record["over_bound"], color, marker, record["passed"])
        top = max(top, record["allowed"], record["over_bound"])
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, top * 1.2)
    ax.set_ylabel("Replicates")
    ax.set_title("Replicates over their per-rep bound", fontsize=9)


def _legend(ax: Axes, series: dict[str, dict[str, Any]]) -> None:
    handles = []
    for i, name in enumerate(series):
        color, marker = _style(i, name)
        handles.append(plt.Line2D([], [], marker=marker, linestyle="none", color=color, label=name))
    handles += [
        plt.Line2D([], [], color="0.5", alpha=0.5, linewidth=2.5, label="acceptance interval"),
        plt.Line2D([], [], marker="_", linestyle="none", markersize=14, color="0.4", label="replicates allowed"),
        plt.Line2D(
            [], [], marker="o", linestyle="none", color="white", markeredgecolor=COLOR_FAIL, label="test failed"
        ),
    ]
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", fontsize=8, frameon=False, ncol=2)


def plot_verdict(verdict: dict[str, Any] | None, out: Path, ext: str = "png") -> None:
    """Draw the folder's verdict to ``verdict.<ext>``; a placeholder when `simace validate` has decided none."""
    path = out / f"verdict.{ext}"
    if verdict is None:
        save_placeholder_plot(
            path,
            "No folder verdict yet: `simace validate <folder>` decides it across replicates.\n"
            "Scenario verdicts, if any, are in verdicts.tsv beside report_summary.tsv.",
            figsize=(8, 3),
        )
        return
    lines = [textwrap.fill(line, width=130) for line in headline(verdict)]
    by_test = _by_test(verdict)
    if not by_test:
        save_placeholder_plot(path, "\n".join(lines), figsize=(10, 3))
        return

    fig = plt.figure(figsize=(10, 7.4 * len(by_test)))
    grid = fig.add_gridspec(3 * len(by_test), 2, height_ratios=[1, 1, 0.8] * len(by_test), width_ratios=[1, 1])
    for t, (test, series) in enumerate(by_test.items()):
        mean_ax = fig.add_subplot(grid[3 * t, :])
        _cell_panel(mean_ax, series, "mean_z", "mean", lambda r: (-r["bound"], r["bound"]), 0.0)
        mean_ax.set_ylabel("Mean z")
        mean_ax.set_title(f"{test}: mean z across replicates, per cell", fontsize=9)
        sd_ax = fig.add_subplot(grid[3 * t + 1, :])
        _cell_panel(sd_ax, series, "sd_z", "sd", lambda r: (r["lo"], r["hi"]), 1.0)
        sd_ax.set_ylabel("SD of z")
        sd_ax.set_title(f"{test}: SD of z across replicates, per cell", fontsize=9)
        _tail_panel(fig.add_subplot(grid[3 * t + 2, 0]), series)
        _legend(fig.add_subplot(grid[3 * t + 2, 1]), series)
    fig.suptitle("\n".join(lines), fontsize=10)
    finalize_plot(path)
