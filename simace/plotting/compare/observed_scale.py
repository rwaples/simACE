"""Phenotype-model figure: observed-scale (tetrachoric) vs liability-scale Falconer h²."""

from __future__ import annotations

__all__ = ["OBSERVED_LIABILITY_ESTIMATOR_DEFS", "compare_observed_vs_liability_h2"]

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from simace.analysis.stats.pedigree_estimates import load_observed_vs_liability_h2
from simace.plotting.compare._draw import bar_offsets, draw_grouped_bars, rep_envelope, save
from simace.plotting.compare.estimates import load_tetrachoric
from simace.plotting.plot_style import apply_nature_style, enable_value_gridlines

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from simace.plotting.compare._inputs import ScenarioReps

# Two estimators of the same underlying narrow-sense h²:
# - "Liability Falconer" uses the continuous (unobservable-in-real-data)
#   liability values directly, via 2 * (r_MZ - r_FS).  It's an oracle
#   reference: what a perfectly observable liability would give you.
# - "Tetrachoric Falconer" uses tetrachoric correlations on binary affected
#   status (observable), via 2 * (r_tetra_MZ - r_tetra_FS).  Under a clean
#   liability-threshold model these match; under frailty / age-of-onset
#   models they diverge, because the mapping from liability to "affected"
#   isn't a single threshold.
OBSERVED_LIABILITY_ESTIMATOR_DEFS: tuple[tuple[str, str], ...] = (
    ("Liability Falconer", "liability_falconer"),
    ("Tetrachoric Falconer", "tetrachoric_falconer"),
)


def compare_observed_vs_liability_h2(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    min_generation: int | None = None,
    input_h2: float | None = None,
) -> None:
    """Two-panel figure: observed-scale vs liability-scale h² by phenotype model.

    Top panel: liability-Falconer and tetrachoric-Falconer h² per scenario,
    with a dashed grey reference line at the simulation input h².  A short
    dark-grey solid tick is drawn at each scenario's realized h² only when
    it meaningfully differs from the input (otherwise the tick would just
    overplot the dashed input reference).  Bottom panel: per-rep signed
    bias of each estimator vs. that rep's realized h².

    Args:
        scenarios: one bar group each; each rep's recorded pedigree and
            ``report.yaml`` (for the tetrachoric correlations) are read.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: forwarded to
            :func:`~simace.analysis.stats.pedigree_estimates.load_pedigree_estimates`.
        input_h2: simulation input h²; drawn as a dashed reference line.
    """
    n_scen = len(scenarios)
    apply_nature_style()
    estimator_labels = [d[0] for d in OBSERVED_LIABILITY_ESTIMATOR_DEFS]
    estimator_keys = [d[1] for d in OBSERVED_LIABILITY_ESTIMATOR_DEFS]
    n_est = len(estimator_keys)

    per_scen = [
        load_observed_vs_liability_h2(
            scenario.pedigrees,
            [load_tetrachoric(p, trait=trait) for p in scenario.reports],
            trait=trait,
            min_generation=min_generation,
        )
        for scenario in scenarios
    ]

    fig, (ax_raw, ax_bias) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)
    x = np.arange(n_scen, dtype=float)  # groups are scenarios (x-axis)
    total_group_width = 0.7
    bar_width = total_group_width / n_est

    # Dedicated palette for the two estimators so scenarios are the x-axis.
    est_colors = ("#4477AA", "#EE6677")

    for est_idx, (est_label, est_key) in enumerate(zip(estimator_labels, estimator_keys, strict=True)):
        raw = [rep_envelope(ests.estimates[est_key]) for ests in per_scen]
        bias = [rep_envelope(ests.estimates[est_key] - ests.realized) for ests in per_scen]
        offsets = bar_offsets(x, total_group_width, n_est, est_idx)
        color = est_colors[est_idx]
        draw_grouped_bars(ax_raw, offsets, raw, bar_width, color, None)
        draw_grouped_bars(ax_bias, offsets, bias, bar_width, color, est_label)

    # Per-scenario realized-h² solid tick in the top panel.  Skip when the
    # scenario's realized h² is effectively equal to the input h² (common in
    # no-AM setups) so we don't double-draw the reference and visually
    # clutter the bars.
    for scen_idx, ests in enumerate(per_scen):
        mean_r = rep_envelope(ests.realized).mean
        if not np.isfinite(mean_r):
            continue
        if input_h2 is not None and abs(mean_r - input_h2) < 0.01:
            continue
        # Narrow tick (half the bar-group width) centered on the scenario.
        half = total_group_width / 4
        ax_raw.hlines(
            mean_r,
            x[scen_idx] - half,
            x[scen_idx] + half,
            colors="#222222",
            linestyles="solid",
            linewidth=1.8,
            zorder=6,
        )

    if input_h2 is not None:
        ax_raw.axhline(
            y=input_h2,
            linestyle="--",
            color="#888888",
            linewidth=1,
            alpha=0.8,
        )
        # Proxy handle so the bias panel's legend can document the dashed
        # reference line (which lives on the raw panel).
        ax_bias.plot(
            [],
            [],
            linestyle="--",
            color="#888888",
            linewidth=1,
            label=f"input / realized h² = {input_h2:.2f} (top panel)",
        )

    ax_bias.axhline(y=0, color="#444444", linewidth=1, alpha=0.9)
    # Show scenario labels between panels (bottom of raw panel), not at figure bottom.
    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels([scenario.label for scenario in scenarios])
    ax_raw.tick_params(axis="x", labelbottom=True)
    ax_bias.tick_params(axis="x", labelbottom=False)
    ax_raw.set_ylabel(f"h² estimate (trait {trait})")
    ax_bias.set_ylabel("Bias vs. realized h² (estimate - truth)")
    ax_raw.set_title("Observed-scale vs liability-scale h² by phenotype model")
    ax_bias.set_title("Bias vs realized h²")
    enable_value_gridlines(ax_raw)
    enable_value_gridlines(ax_bias)
    ax_bias.legend(loc="lower left", fontsize=9, frameon=False)

    save(fig, output_path)
