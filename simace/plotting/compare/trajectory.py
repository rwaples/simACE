"""Realized variance components per generation, one line per scenario (AM and time-varying E pages)."""

from __future__ import annotations

__all__ = ["compare_realized_variance_trajectory"]

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from simace.plotting.compare._draw import draw_envelope_line, rep_envelope, save, scenario_color
from simace.plotting.compare.estimates import load_per_generation
from simace.plotting.plot_style import apply_nature_style, enable_value_gridlines

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from simace.plotting.compare._inputs import ScenarioReps


def compare_realized_variance_trajectory(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    expected_A: float | list[float] | None = None,
    expected_C: float | list[float] | None = None,
    expected_E: float | list[float] | None = None,
) -> None:
    """Plot realized vA, vC, vE, and h² per generation, one line per scenario.

    Args:
        scenarios: the scenarios to overlay; each rep's ``report.yaml`` is read.
        output_path: image path to save (extension determines format).
        trait: 1 or 2; which trait's variance components to plot.
        expected_A: optional input vA reference. Scalar draws a horizontal
            dashed line; list draws a per-generation dashed curve.
        expected_C: optional input vC reference, scalar or per-generation list.
        expected_E: optional input vE reference, scalar or per-generation list.
            If all three ``expected_*`` are supplied, the h² reference is
            ``A / (A + C + E)`` (computed per-generation when any input is
            a list).
    """
    h2_expected = None
    if expected_A is not None and expected_C is not None and expected_E is not None:
        if any(isinstance(e, list) for e in (expected_A, expected_C, expected_E)):
            n = max(len(e) if isinstance(e, list) else 1 for e in (expected_A, expected_C, expected_E))
            a_list = expected_A if isinstance(expected_A, list) else [expected_A] * n
            c_list = expected_C if isinstance(expected_C, list) else [expected_C] * n
            e_list = expected_E if isinstance(expected_E, list) else [expected_E] * n
            h2_expected = [
                a / (a + c + e) if (a + c + e) > 0 else None for a, c, e in zip(a_list, c_list, e_list, strict=True)
            ]
        else:
            total = expected_A + expected_C + expected_E  # ty: ignore[unsupported-operator]
            h2_expected = expected_A / total if total > 0 else None

    apply_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    panel_defs = [
        (axes[0, 0], 0, f"Realized vA (trait {trait})", expected_A),
        (axes[0, 1], 1, f"Realized vC (trait {trait})", expected_C),
        (axes[1, 0], 2, f"Realized vE (trait {trait})", expected_E),
        (axes[1, 1], 3, f"Realized h² (trait {trait})", h2_expected),
    ]

    for scen_idx, scenario in enumerate(scenarios):
        per_gen = load_per_generation(scenario.reports, trait=trait)
        gens = sorted(per_gen.keys())
        color = scenario_color(scen_idx)
        for ax, col, _title, _expected in panel_defs:
            draw_envelope_line(ax, gens, [rep_envelope(per_gen[gen][:, col]) for gen in gens], color, scenario.label)

    for ax, _col, title, expected in panel_defs:
        ax.set_title(title)
        ax.set_ylabel("Variance" if _col < 3 else "h²")
        enable_value_gridlines(ax)
        if expected is None:
            continue
        if isinstance(expected, list):
            # Per-generation reference; align to gens 1..len (matches
            # report.yaml's 1-indexed generation_N keys).
            ref_gens = list(range(1, len(expected) + 1))
            ref_vals = [v for v in expected if v is not None]
            ref_x = [g for g, v in zip(ref_gens, expected, strict=True) if v is not None]
            if ref_vals:
                ax.plot(ref_x, ref_vals, linestyle="--", color="#888888", linewidth=1, alpha=0.8)
        else:
            ax.axhline(y=expected, linestyle="--", color="#888888", linewidth=1, alpha=0.8)

    for ax in axes[1, :]:
        ax.set_xlabel("Generation")

    # Single legend at the top of the figure
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(scenarios),
        frameon=False,
    )

    save(fig, output_path, rect=(0, 0, 1, 0.97))
