"""Onset τ of the frailty model's raw onsets against its realized expectation (ADR 0027).

One panel per frailty trait and one within-person panel. Each relationship
type is a point at its cell's linear-predictor correlation ``r̂``: the observed
onset τ, filled, with its ``±z_bound·SE`` interval (the rep's per-rep
Bonferroni bound), and the realized onset-τ expectation it is scored against,
hollow. A red edge marks a cell whose ``|z|`` exceeds that bound; a correct
simulation has one in about 5% of reps, and the verdict is across reps. The curve is ``tau_oracle(s, s, r)``,
the onset τ of bivariate normal linear predictors with the cell's spread, which
needs no baseline hazard.
"""

from __future__ import annotations

__all__ = ["plot_onset_tau"]

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from simace.analysis.stats.onset_tau import CELL_CATEGORIES, tau_oracle
from simace.plotting.plot_style import PAIR_COLORS
from simace.plotting.plot_utils import finalize_plot, save_placeholder_plot

if TYPE_CHECKING:
    from pathlib import Path

COLOR_BEYOND = "#CC3311"
COLOR_UNSCORED = "#BBBBBB"
_PANELS = (("trait1", "Trait 1"), ("trait2", "Trait 2"), ("within_person", "Within-person (t1 vs t2)"))


def _rows(section: dict[str, Any], panel: str) -> list[tuple[str, dict[str, Any]]]:
    cells = section.get("cells") or {}
    if panel == "within_person":
        row = cells.get("within_person")
        return [("within", row)] if row else []
    return [(code, cells[f"{panel}_{code}"]) for code in CELL_CATEGORIES if f"{panel}_{code}" in cells]


def _draw_point(ax, row: dict[str, Any], color: str, z_bound: float | None) -> None:
    x, observed, expected = row.get("r_eta"), row.get("observed"), row.get("expected")
    if x is None or observed is None:
        return
    scored = row.get("status") == "scored"
    face = color if scored else COLOR_UNSCORED
    edge = COLOR_BEYOND if scored and abs(row["z"]) > z_bound else face
    if scored:
        ax.errorbar(x, observed, yerr=z_bound * row["se"], fmt="none", ecolor=face, elinewidth=1, zorder=3)
    ax.scatter(x, observed, s=36, color=face, edgecolors=edge, linewidths=1.5, zorder=5)
    if expected is not None:
        ax.scatter(x, expected, s=60, facecolors="none", edgecolors=face, linewidths=1.2, zorder=4)


def plot_onset_tau(all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str = "") -> None:
    """Observed onset τ and its realized expectation per relationship type, over the τ–r curve."""
    sections = [stats.get("onset_tau") or {} for stats in all_stats]
    panels = [(key, title) for key, title in _PANELS if any(_rows(section, key) for section in sections)]
    if not panels:
        save_placeholder_plot(output_path, "Onset τ not computed: no frailty trait, or reps predate the check")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
    for ax, (key, title) in zip(axes[0], panels, strict=True):
        spreads, xs = [], []
        for section in sections:
            z_bound = (section.get("summary") or {}).get("z_bound")
            for code, row in _rows(section, key):
                _draw_point(ax, row, PAIR_COLORS.get(code, "black"), z_bound)
                if row.get("r_eta") is not None:
                    xs.append(row["r_eta"])
                    spreads.append((row["s1"], row["s2"]))
        if spreads:
            s1, s2 = np.median(np.array(spreads), axis=0)
            lo, hi = min(0.0, *xs) - 0.05, max(0.6, max(xs) + 0.05)
            grid = np.linspace(lo, min(hi, 1.0), 41)
            ax.plot(grid, [tau_oracle(s1, s2, r) for r in grid], color="black", linewidth=1, zorder=2)
        ax.axhline(0, color="0.8", linewidth=0.8, zorder=1)
        ax.set_title(title)
        ax.set_xlabel(r"Linear-predictor correlation $\hat{r}$")
        ax.set_ylabel("Onset τ")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=PAIR_COLORS[code], label=code)
        for code in CELL_CATEGORIES
    ]
    handles += [
        plt.Line2D([], [], marker="o", linestyle="none", color="0.3", label="observed (±z_bound·SE)"),
        plt.Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="none", color="0.3", label="realized expectation"
        ),
        plt.Line2D([], [], marker="o", linestyle="none", color=COLOR_UNSCORED, label="not scored"),
        plt.Line2D(
            [], [], marker="o", linestyle="none", color="white", markeredgecolor=COLOR_BEYOND, label="|z| > z_bound"
        ),
        plt.Line2D([], [], color="black", linewidth=1, label="tau_oracle(s, s, r)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    finalize_plot(output_path, tight_rect=[0, 0.1, 1, 1], scenario=scenario)
