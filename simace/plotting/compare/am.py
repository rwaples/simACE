"""Assortative-mating figures: component distributions, correlations by class, sib pairs, naive h² estimators."""

from __future__ import annotations

__all__ = [
    "NAIVE_ESTIMATOR_DEFS",
    "compare_component_distributions",
    "compare_correlations_by_relclass",
    "compare_naive_estimators",
    "compare_sib_liability_scatter",
]

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from simace.analysis.stats.pedigree_estimates import (
    POOLED_RELATIONSHIP_CLASSES,
    load_naive_estimator_h2,
    load_pooled_liability_correlations,
    load_sib_pair_liabilities,
)
from simace.core.numerics import safe_corrcoef
from simace.core.parquet import load_parquet
from simace.core.relationships import reference_corr
from simace.plotting.compare._draw import (
    bar_offsets,
    draw_grouped_bars,
    envelope_arrays,
    padded_range,
    rep_envelope,
    save,
    scenario_color,
)
from simace.plotting.plot_style import apply_nature_style, enable_value_gridlines

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from simace.plotting.compare._inputs import ScenarioReps
    from simace.simulation.params import SimulationParams

# (display label, NaiveH2 field).  Each estimator maps one correlation (or
# regression slope) into an h² estimate that is unbiased under random mating
# but drifts under AM, by different amounts for different relationship
# classes (simace.analysis.stats.naive_h2).
NAIVE_ESTIMATOR_DEFS: tuple[tuple[str, str], ...] = (
    ("Twin Falconer", "falconer"),
    ("Sibs only", "sibs"),
    ("Midparent PO", "po"),
    ("Half-sibs", "hs"),
    ("Cousins", "cousins"),
)


def compare_component_distributions(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    min_generation: int | None = None,
    n_bins: int = 60,
) -> None:
    """Overlay histograms of A (top) and total liability (bottom) per scenario.

    The trajectory plot
    (:func:`~simace.plotting.compare.trajectory.compare_realized_variance_trajectory`)
    shows *v_A* as a time series of numbers.  This function shows what that
    numerical inflation looks like as a change in the actual distribution
    of per-individual A values and per-individual liability.  Under AM,
    both widen; the A distribution reveals the pure additive-genetic
    inflation and the liability distribution shows how that inflation
    propagates to the total phenotype.

    Args:
        scenarios: the scenarios to overlay; each rep's recorded pedigree is read.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: restrict to individuals in ``generation >=
            min_generation``.  Use a late-generation filter if you want to
            show the equilibrium distribution; ``None`` pools every
            generation.
        n_bins: histogram bin count (shared across all scenarios and both
            panels for visual comparability).
    """
    apply_nature_style()
    a_key = f"A{trait}"
    l_key = f"liability{trait}"

    per_scen_a: list[np.ndarray] = []
    per_scen_l: list[np.ndarray] = []
    for scenario in scenarios:
        a_parts: list[np.ndarray] = []
        l_parts: list[np.ndarray] = []
        for path in scenario.pedigrees:
            df = load_parquet(path, columns=["generation", a_key, l_key])
            if min_generation is not None:
                df = df.filter(pl.col("generation") >= min_generation)
            a_parts.append(df[a_key].to_numpy())
            l_parts.append(df[l_key].to_numpy())
        per_scen_a.append(np.concatenate(a_parts))
        per_scen_l.append(np.concatenate(l_parts))

    # Shared x-axis across both panels makes the relative widths (A is
    # tighter than liability, because liability = A + E) directly legible.
    lo, hi = padded_range(np.concatenate([np.concatenate(per_scen_a), np.concatenate(per_scen_l)]))
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, (ax_a, ax_l) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for scen_idx, (a_vals, l_vals, scenario) in enumerate(zip(per_scen_a, per_scen_l, scenarios, strict=True)):
        color = scenario_color(scen_idx)
        sd_a = float(np.std(a_vals, ddof=1))
        sd_l = float(np.std(l_vals, ddof=1))
        for ax, vals, sd in ((ax_a, a_vals, sd_a), (ax_l, l_vals, sd_l)):
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="stepfilled",
                color=color,
                alpha=0.25,
                linewidth=0,
            )
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linewidth=1.8,
                label=f"{scenario.label}  (sd = {sd:.3f})",
            )

    ax_a.set_title(f"Additive genetic component A{trait}")
    ax_l.set_title(f"Total liability (A + C + E), trait {trait}")
    ax_a.set_ylabel("Density")
    ax_l.set_ylabel("Density")
    ax_l.set_xlabel(f"Value (trait {trait})")
    ax_a.legend(loc="upper left", fontsize=9, frameon=False)
    ax_l.legend(loc="upper left", fontsize=9, frameon=False)
    enable_value_gridlines(ax_a)
    enable_value_gridlines(ax_l)

    save(fig, output_path)


def compare_correlations_by_relclass(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    reference: SimulationParams | None = None,
    min_generation: int | None = None,
) -> None:
    """Grouped bar chart: liability correlation by relationship class, per scenario.

    Relationship classes are pooled to {FS, PO, HS, 1C} and displayed on
    the x-axis in decreasing expected kinship.  Each scenario gets one bar
    per class, drawn with min/max error bars across replicates.  If
    ``reference`` is supplied, a dashed line per class shows that
    scenario's reference correlation
    (:func:`~simace.core.relationships.reference_corr`): under random mating
    the analytic ``(2 * kinship * A + C_shared * C) / V_P``, with C
    transmission where it applies; this is what any AM scenario should
    deviate from as AM strength grows. A class whose pooled members expect
    different values (HS when ``C > 0``), or that has no analytic value,
    gets no line.

    Args:
        scenarios: the scenarios to compare; each rep's recorded pedigree is read.
        output_path: image path to save.
        trait: 1 or 2.
        reference: parameters of the reference scenario, usually the first
            (:meth:`ScenarioReps.parameters`).  If ``None``, no reference
            lines are drawn.
        min_generation: restrict correlation computations to individuals in
            generations ``>= min_generation``.  Useful for letting AM reach
            its equilibrium.  ``None`` uses all generations.
    """
    apply_nature_style()
    class_names = [c[0] for c in POOLED_RELATIONSHIP_CLASSES]
    n_classes = len(class_names)
    n_scen = len(scenarios)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_classes, dtype=float)
    total_group_width = 0.8
    bar_width = total_group_width / n_scen

    # First pass: compute per-scenario envelopes so we can use the baseline
    # (first scenario) means to annotate non-baseline bars with absolute and
    # relative inflation.
    per_scen_envelopes = []
    for scenario in scenarios:
        per_class = load_pooled_liability_correlations(scenario.pedigrees, trait=trait, min_generation=min_generation)
        per_scen_envelopes.append([rep_envelope(per_class[cls]) for cls in class_names])
    per_scen_offsets = [bar_offsets(x, total_group_width, n_scen, scen_idx) for scen_idx in range(n_scen)]

    # Second pass: draw bars.
    for scen_idx, scenario in enumerate(scenarios):
        draw_grouped_bars(
            ax,
            per_scen_offsets[scen_idx],
            per_scen_envelopes[scen_idx],
            bar_width,
            scenario_color(scen_idx),
            scenario.label,
        )

    # Third pass: annotate each non-baseline bar with inflation vs scenario 0.
    baseline = envelope_arrays(per_scen_envelopes[0])[0]
    for scen_idx in range(1, n_scen):
        means, _, highs = envelope_arrays(per_scen_envelopes[scen_idx])
        offsets = per_scen_offsets[scen_idx]
        color = scenario_color(scen_idx)
        for i in range(n_classes):
            if not (np.isfinite(means[i]) and np.isfinite(baseline[i])) or baseline[i] <= 0:
                continue
            abs_delta = means[i] - baseline[i]
            rel_pct = abs_delta / baseline[i] * 100.0
            top = highs[i] if np.isfinite(highs[i]) else means[i]
            ax.text(
                offsets[i],
                top + 0.008,
                f"{abs_delta:+.2f}\n({rel_pct:+.0f}%)",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
            )

    if reference is not None:
        for i, (_, _, members) in enumerate(POOLED_RELATIONSHIP_CLASSES):
            # A pooled class has one expectation only when its members agree:
            # HS pools MHS (shares C) with PHS (does not), so it has none when C > 0.
            expected, *others = (
                reference_corr(member, reference.assortment, reference.c_transmission, reference.vc, trait)
                for member in members
            )
            if expected is None or any(other != expected for other in others):
                continue
            ax.hlines(
                expected,
                x[i] - total_group_width / 2,
                x[i] + total_group_width / 2,
                colors="#888888",
                linestyles="dashed",
                linewidth=1,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Liability correlation")
    ax.set_xlabel("Relationship class (pooled)")
    ax.set_title(f"Liability correlation by relationship class (trait {trait})")
    enable_value_gridlines(ax)
    ax.legend(loc="upper right", frameon=False)

    save(fig, output_path)


def _covariance_ellipse_xy(
    x: np.ndarray,
    y: np.ndarray,
    n_std: float,
    n_points: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` coordinates tracing the n-SD covariance ellipse.

    The ellipse is the level set of the bivariate-Gaussian fit whose Mahalanobis
    distance equals ``n_std``.  At ``n_std=2`` this is approximately the 95%
    contour.
    """
    cov = np.cov(x, y)
    mean_x, mean_y = float(x.mean()), float(y.mean())
    vals, vecs = np.linalg.eigh(cov)
    # descending eigenvalues
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    rx = n_std * float(np.sqrt(max(vals[0], 0.0)))
    ry = n_std * float(np.sqrt(max(vals[1], 0.0)))
    ex = rx * np.cos(theta)
    ey = ry * np.sin(theta)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_x = ex * cos_a - ey * sin_a + mean_x
    rot_y = ex * sin_a + ey * cos_a + mean_y
    return rot_x, rot_y


def compare_sib_liability_scatter(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    min_generation: int | None = None,
) -> None:
    """Overlay of sib-pair liability shape changes across scenarios.

    A hexbin cloud of ~half a million pairs looks similar whether the
    correlation is 0.25 or 0.35 — the signal is real but the visual isn't.
    This plot instead overlays, on a single axis, two crisp summary shapes
    per scenario: the 2-SD covariance ellipse (approximately the 95% contour
    under a Gaussian fit) and the best-fit linear regression line.  Together,
    ellipse eccentricity and line slope make the AM-induced elongation of the
    joint distribution immediately legible.

    Args:
        scenarios: the scenarios to overlay; each rep's recorded pedigree is read.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: same meaning as in
            :func:`~simace.analysis.stats.pedigree_estimates.load_sib_pair_liabilities`.
    """
    apply_nature_style()

    loaded = [
        load_sib_pair_liabilities(scenario.pedigrees, trait=trait, min_generation=min_generation)
        for scenario in scenarios
    ]

    combined = np.concatenate([np.concatenate([a, b]) for a, b in loaded if a.size])
    if combined.size == 0:
        raise ValueError("No sib pairs found in any scenario.")
    lo, hi = padded_range(combined)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    for scen_idx, ((liab_a, liab_b), scenario) in enumerate(zip(loaded, scenarios, strict=True)):
        if liab_a.size == 0:
            continue
        color = scenario_color(scen_idx)
        r = safe_corrcoef(liab_a, liab_b)
        # Regression through (mean_a, mean_b) with slope = r * (sd_b/sd_a).
        mean_a, mean_b = float(liab_a.mean()), float(liab_b.mean())
        sd_a = float(liab_a.std(ddof=1))
        sd_b = float(liab_b.std(ddof=1))
        slope = r * sd_b / sd_a if sd_a > 0 else 0.0
        line_x = np.array([lo, hi])
        line_y = mean_b + slope * (line_x - mean_a)

        # 2-SD covariance ellipse (approx 95% contour for bivariate normal).
        ex, ey = _covariance_ellipse_xy(liab_a, liab_b, n_std=2.0)

        full_label = f"{scenario.label}  (r = {r:.3f}, slope = {slope:.2f})"
        ax.plot(ex, ey, color=color, linewidth=2.0, zorder=5, label=full_label)
        ax.plot(line_x, line_y, color=color, linewidth=1.5, linestyle=":", zorder=6)

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888888", linewidth=1, alpha=0.8, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Sib A liability{trait}")
    ax.set_ylabel(f"Sib B liability{trait}")
    ax.set_title(f"Sib-pair liability distribution (trait {trait}): 2-SD ellipse + regression")
    enable_value_gridlines(ax)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    save(fig, output_path)


def compare_naive_estimators(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    input_h2: float | None = None,
    min_generation: int | None = None,
) -> None:
    """Two-panel figure showing naive h² estimates and their bias vs. realized.

    Top panel: five naive estimators' h² values per scenario, with a dashed
    grey reference line at ``input_h2`` (the value that was typed into the
    config).  Bottom panel: each estimator's per-rep signed bias relative to
    that scenario's *realized* h² (variance ratio in the same
    ``min_generation``-filtered subset used to compute the correlations),
    so the reader can separate "population moved" from "estimator is
    biased."

    Args:
        scenarios: the scenarios to compare; each rep's recorded pedigree is read.
        output_path: image path to save.
        trait: 1 or 2.
        input_h2: the simulation input h².  Drawn as a dashed reference line
            in the top panel; omitted if ``None``.
        min_generation: restrict all estimator computations (and the
            realized-h² reference) to ``generation >= min_generation``.
            ``None`` uses all generations.
    """
    apply_nature_style()
    estimator_labels = [d[0] for d in NAIVE_ESTIMATOR_DEFS]
    estimator_keys = [d[1] for d in NAIVE_ESTIMATOR_DEFS]
    n_est = len(estimator_keys)
    n_scen = len(scenarios)

    per_scen = [
        load_naive_estimator_h2(scenario.pedigrees, trait=trait, min_generation=min_generation)
        for scenario in scenarios
    ]

    fig, (ax_raw, ax_bias) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = np.arange(n_est, dtype=float)
    total_group_width = 0.8
    bar_width = total_group_width / n_scen

    for scen_idx, (scenario, ests) in enumerate(zip(scenarios, per_scen, strict=True)):
        color = scenario_color(scen_idx)
        offsets = bar_offsets(x, total_group_width, n_scen, scen_idx)
        realized = ests.realized
        raw = [rep_envelope(ests.estimates[key]) for key in estimator_keys]
        # Per-rep bias vs. per-rep realized, then aggregate.
        bias = [rep_envelope(ests.estimates[key] - realized) for key in estimator_keys]
        draw_grouped_bars(ax_raw, offsets, raw, bar_width, color, scenario.label)
        draw_grouped_bars(ax_bias, offsets, bias, bar_width, color, None)

        # Short horizontal tick at each scenario's realized h² in the top
        # panel, spanning just that scenario's bar slice of each group.
        realized_mean = rep_envelope(realized).mean
        if np.isfinite(realized_mean):
            for i in range(n_est):
                ax_raw.hlines(
                    realized_mean,
                    offsets[i] - bar_width / 2,
                    offsets[i] + bar_width / 2,
                    colors=color,
                    linestyles="solid",
                    linewidth=1.6,
                    zorder=6,
                )

    if input_h2 is not None:
        ax_raw.axhline(
            y=input_h2,
            linestyle="--",
            color="#888888",
            linewidth=1,
            alpha=0.8,
            label=f"input h² = {input_h2:.2f}",
        )

    ax_bias.axhline(y=0, color="#444444", linewidth=1, alpha=0.9)
    # Show estimator labels between the two panels (bottom of the top panel),
    # not at the very bottom of the figure.
    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels(estimator_labels)
    ax_raw.tick_params(axis="x", labelbottom=True)
    ax_bias.tick_params(axis="x", labelbottom=False)
    ax_raw.set_ylabel("Naive h² estimate")
    ax_bias.set_ylabel("Bias vs. realized h² (estimate - truth)")
    ax_raw.set_title("Naive h² estimators across AM levels")
    ax_bias.set_title("Bias relative to each scenario's realized h² (solid tick in top panel)")
    enable_value_gridlines(ax_raw)
    enable_value_gridlines(ax_bias)
    ax_raw.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    save(fig, output_path)
