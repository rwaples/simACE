"""Cross-scenario comparison plots for Examples docs pages.

The per-scenario atlas answers "what happened in this one run?".  The Examples
docs pages in ``docs/examples/`` ask a different question: "how does the
outcome shift when we dial a knob?".  That needs a plot that overlays two or
more scenarios on the same axes.  This module hosts those comparison plots;
each new Examples topic adds one function here.
"""

from __future__ import annotations

__all__ = [
    "NAIVE_ESTIMATOR_DEFS",
    "OBSERVED_LIABILITY_ESTIMATOR_DEFS",
    "SCENARIO_PALETTE",
    "compare_cohort_falconer",
    "compare_cohort_fs_correlations",
    "compare_component_distributions",
    "compare_components_by_generation",
    "compare_correlations_by_relclass",
    "compare_naive_estimators",
    "compare_observed_vs_liability_h2",
    "compare_prevalence_drift",
    "compare_realized_variance_trajectory",
    "compare_sib_liability_scatter",
    "load_per_generation",
    "scenario_parameters",
]

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from simace.analysis.stats.naive_h2 import falconer_h2
from simace.analysis.stats.pedigree_estimates import (
    POOLED_RELATIONSHIP_CLASSES,
    load_naive_estimator_h2,
    load_observed_vs_liability_h2,
    load_pedigree_estimates,
    load_pedigree_estimates_per_generation,
    load_pooled_liability_correlations,
    load_sib_pair_liabilities,
)
from simace.core.numerics import safe_corrcoef
from simace.core.parquet import load_parquet
from simace.core.relationships import RELATIONSHIP_TYPES, reference_corr
from simace.core.yaml_io import load_yaml
from simace.plotting.plot_style import (
    apply_nature_style,
    enable_value_gridlines,
)
from simace.plotting.plot_utils import finalize_plot
from simace.plotting.stats_report import plotting_report_view, report_per_generation
from simace.simulation.params import SimulationParams

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from matplotlib.axes import Axes

    from simace.analysis.stats.pedigree_estimates import CohortEstimates

logger = logging.getLogger(__name__)

# Ordered palette for scenario lines in comparison plots.  Colorblind-safe,
# picked to read cleanly when 2-5 scenarios are overlaid.
SCENARIO_PALETTE: tuple[str, ...] = (
    "#4477AA",  # blue
    "#EE6677",  # rose
    "#228833",  # green
    "#CCBB44",  # olive
    "#AA3377",  # purple
)


def load_per_generation(
    report_paths: list[Path],
    trait: int = 1,
) -> dict[int, np.ndarray]:
    """Read ``per_generation`` variance components from report.yaml files.

    Args:
        report_paths: one path per replicate (of a single scenario).
        trait: 1 or 2.

    Returns:
        Dict keyed by generation (1-indexed, matching the YAML keys) whose
        values are (n_reps, 4) arrays of columns ``[vA, vC, vE, h2]``.  ``h2``
        is ``vA / (vA + vC + vE)``.  Generations missing from one rep raise
        ``KeyError``; this is intentional since the comparison plot needs a
        consistent gen axis.
    """
    per_rep: list[dict[int, tuple[float, float, float, float]]] = []
    for path in report_paths:
        per_gen = report_per_generation(load_yaml(path))
        rep_dict: dict[int, tuple[float, float, float, float]] = {}
        for gen_key, gen_data in per_gen.items():
            gen = int(str(gen_key).removeprefix("generation_"))
            vA = float(gen_data[f"A{trait}_var"])
            vC = float(gen_data[f"C{trait}_var"])
            vE = float(gen_data[f"E{trait}_var"])
            total = vA + vC + vE
            h2 = vA / total if total > 0 else float("nan")
            rep_dict[gen] = (vA, vC, vE, h2)
        per_rep.append(rep_dict)

    gens = sorted(set().union(*(r.keys() for r in per_rep)))
    out: dict[int, np.ndarray] = {}
    for gen in gens:
        rows = [r[gen] for r in per_rep]
        out[gen] = np.asarray(rows, dtype=float)
    return out


def scenario_color(index: int) -> str:
    """The palette color of the *index*-th scenario, cycling past the palette's end."""
    return SCENARIO_PALETTE[index % len(SCENARIO_PALETTE)]


@dataclass(frozen=True)
class Envelope:
    """Mean and min/max across reps of one quantity; all NaN when no rep has a finite value.

    For a small number of reps (typical: 3) a parametric CI is less honest
    than min/max across reps, so min/max is the envelope.
    """

    mean: float
    low: float
    high: float


def rep_envelope(values: Iterable[float] | np.ndarray) -> Envelope:
    """The :class:`Envelope` of the finite *values* (one per rep)."""
    arr = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return Envelope(math.nan, math.nan, math.nan)
    return Envelope(float(finite.mean()), float(finite.min()), float(finite.max()))


def _envelope_arrays(envelopes: Sequence[Envelope]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(means, lows, highs)`` arrays of *envelopes*."""
    return (
        np.array([e.mean for e in envelopes], dtype=float),
        np.array([e.low for e in envelopes], dtype=float),
        np.array([e.high for e in envelopes], dtype=float),
    )


def draw_envelope_line(
    ax: Axes, x: Sequence[float], envelopes: Sequence[Envelope], color: str, label: str, marker: str = "o"
) -> None:
    """Plot the envelope means as a marked line over a translucent min/max band."""
    ax.plot(x, [e.mean for e in envelopes], color=color, marker=marker, label=label)
    ax.fill_between(x, [e.low for e in envelopes], [e.high for e in envelopes], color=color, alpha=0.15, linewidth=0)


def bar_offsets(x: np.ndarray, group_width: float, n_bars: int, index: int) -> np.ndarray:
    """Centers of the *index*-th of *n_bars* bars side by side in each group of width *group_width* around *x*."""
    bar_width = group_width / n_bars
    return x - group_width / 2 + bar_width / 2 + index * bar_width


def draw_grouped_bars(
    ax: Axes, offsets: np.ndarray, envelopes: Sequence[Envelope], bar_width: float, color: str, label: str | None
) -> None:
    """Bars at the envelope means with min/max error bars; groups with no finite mean are left empty."""
    means, lows, highs = _envelope_arrays(envelopes)
    mask = np.isfinite(means)
    err_low = np.where(mask, means - lows, 0.0)
    err_high = np.where(mask, highs - means, 0.0)
    ax.bar(
        offsets[mask],
        means[mask],
        width=bar_width * 0.95,
        color=color,
        label=label,
        yerr=[err_low[mask], err_high[mask]],
        capsize=3,
        linewidth=0,
    )


def padded_range(values: np.ndarray) -> tuple[float, float]:
    """The 0.1%-99.9% quantile range of *values*, widened by 5% on each side: a shared axis that ignores outliers."""
    lo = float(np.quantile(values, 0.001))
    hi = float(np.quantile(values, 0.999))
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _save(fig: plt.Figure, output_path: Path | str, rect: tuple[float, float, float, float] | None = None) -> None:
    """Write ``fig`` to ``output_path`` through :func:`finalize_plot`, creating its directory first."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(fig.number)  # finalize_plot lays out and saves the current figure
    finalize_plot(output_path, tight_rect=None if rect is None else list(rect))
    logger.info("Wrote %s", output_path)


def compare_realized_variance_trajectory(
    scenario_paths: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    expected_A: float | list[float] | None = None,
    expected_C: float | list[float] | None = None,
    expected_E: float | list[float] | None = None,
) -> None:
    """Plot realized vA, vC, vE, and h² per generation, one line per scenario.

    Args:
        scenario_paths: outer list = scenarios, inner list = replicate
            ``report.yaml`` paths for that scenario.
        labels: display label per scenario (same order as ``scenario_paths``).
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
    if len(scenario_paths) != len(labels):
        raise ValueError(f"scenario_paths ({len(scenario_paths)}) and labels ({len(labels)}) must match")

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

    for scen_idx, (rep_paths, label) in enumerate(zip(scenario_paths, labels, strict=True)):
        reps = [Path(p) for p in rep_paths]
        per_gen = load_per_generation(reps, trait=trait)
        gens = sorted(per_gen.keys())
        color = scenario_color(scen_idx)
        for ax, col, _title, _expected in panel_defs:
            draw_envelope_line(ax, gens, [rep_envelope(per_gen[gen][:, col]) for gen in gens], color, label)

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
        ncol=len(labels),
        frameon=False,
    )

    _save(fig, output_path, rect=(0, 0, 1, 0.97))


def compare_component_distributions(
    scenario_paths: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    min_generation: int | None = None,
    n_bins: int = 60,
) -> None:
    """Overlay histograms of A (top) and total liability (bottom) per scenario.

    The trajectory plot (see :func:`compare_realized_variance_trajectory`)
    shows *v_A* as a time series of numbers.  This function shows what that
    numerical inflation looks like as a change in the actual distribution
    of per-individual A values and per-individual liability.  Under AM,
    both widen; the A distribution reveals the pure additive-genetic
    inflation and the liability distribution shows how that inflation
    propagates to the total phenotype.

    Args:
        scenario_paths: outer list = scenarios, inner list = per-replicate
            recorded ``pedigree.parquet`` paths.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: restrict to individuals in ``generation >=
            min_generation``.  Use a late-generation filter if you want to
            show the equilibrium distribution; ``None`` pools every
            generation.
        n_bins: histogram bin count (shared across all scenarios and both
            panels for visual comparability).
    """
    if len(scenario_paths) != len(labels):
        raise ValueError(f"scenario_paths ({len(scenario_paths)}) and labels ({len(labels)}) must match")

    apply_nature_style()
    a_key = f"A{trait}"
    l_key = f"liability{trait}"

    per_scen_a: list[np.ndarray] = []
    per_scen_l: list[np.ndarray] = []
    for reps in scenario_paths:
        a_parts: list[np.ndarray] = []
        l_parts: list[np.ndarray] = []
        for path in reps:
            df = load_parquet(Path(path), columns=["generation", a_key, l_key])
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

    for scen_idx, (a_vals, l_vals, label) in enumerate(zip(per_scen_a, per_scen_l, labels, strict=True)):
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
                label=f"{label}  (sd = {sd:.3f})",
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

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Liability correlations by relationship class
# ---------------------------------------------------------------------------


def scenario_parameters(pedigree_path: Path) -> SimulationParams:
    """The parameters of the rep that wrote *pedigree_path*, parsed from the ``report.yaml`` beside it."""
    report = load_yaml(Path(pedigree_path).with_name("report.yaml"))
    return SimulationParams.from_mapping(report["inputs"]["parameters"])


def compare_correlations_by_relclass(
    scenario_paths: list[list[Path]],
    labels: list[str],
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
        scenario_paths: outer list = scenarios, inner list = replicate
            recorded ``pedigree.parquet`` paths for that scenario.
        labels: display label per scenario (same order as ``scenario_paths``).
        output_path: image path to save.
        trait: 1 or 2.
        reference: parameters of the reference scenario,
            usually the first (:func:`scenario_parameters`).  If ``None``, no
            reference lines are drawn.
        min_generation: restrict correlation computations to individuals in
            generations ``>= min_generation``.  Useful for letting AM reach
            its equilibrium.  ``None`` uses all generations.
    """
    if len(scenario_paths) != len(labels):
        raise ValueError(f"scenario_paths ({len(scenario_paths)}) and labels ({len(labels)}) must match")

    apply_nature_style()
    class_names = [c[0] for c in POOLED_RELATIONSHIP_CLASSES]
    n_classes = len(class_names)
    n_scen = len(labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_classes, dtype=float)
    total_group_width = 0.8
    bar_width = total_group_width / n_scen

    # First pass: compute per-scenario means/lows/highs so we can use the
    # baseline (first scenario) means to annotate non-baseline bars with
    # absolute and relative inflation.
    per_scen_envelopes: list[list[Envelope]] = []
    for reps in scenario_paths:
        per_class = load_pooled_liability_correlations(
            [Path(p) for p in reps], trait=trait, min_generation=min_generation
        )
        per_scen_envelopes.append([rep_envelope(per_class[cls]) for cls in class_names])
    per_scen_offsets = [bar_offsets(x, total_group_width, n_scen, scen_idx) for scen_idx in range(n_scen)]

    # Second pass: draw bars.
    for scen_idx, label in enumerate(labels):
        draw_grouped_bars(
            ax, per_scen_offsets[scen_idx], per_scen_envelopes[scen_idx], bar_width, scenario_color(scen_idx), label
        )

    # Third pass: annotate each non-baseline bar with inflation vs scenario 0.
    baseline = _envelope_arrays(per_scen_envelopes[0])[0]
    for scen_idx in range(1, n_scen):
        means, _, highs = _envelope_arrays(per_scen_envelopes[scen_idx])
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
                f"+{abs_delta:.2f}\n({rel_pct:+.0f}%)",
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

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Sib-pair liability scatter
# ---------------------------------------------------------------------------


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
    scenario_paths: list[list[Path]],
    labels: list[str],
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
        scenario_paths: outer list = scenarios, inner list = per-replicate
            recorded ``pedigree.parquet`` paths.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: same meaning as in :func:`load_sib_pair_liabilities`.
    """
    if len(scenario_paths) != len(labels):
        raise ValueError(f"scenario_paths ({len(scenario_paths)}) and labels ({len(labels)}) must match")

    apply_nature_style()

    loaded: list[tuple[np.ndarray, np.ndarray]] = []
    for reps in scenario_paths:
        liab_a, liab_b = load_sib_pair_liabilities([Path(p) for p in reps], trait=trait, min_generation=min_generation)
        loaded.append((liab_a, liab_b))

    combined = np.concatenate([np.concatenate([a, b]) for a, b in loaded if a.size])
    if combined.size == 0:
        raise ValueError("No sib pairs found in any scenario.")
    lo, hi = padded_range(combined)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    for scen_idx, ((liab_a, liab_b), label) in enumerate(zip(loaded, labels, strict=True)):
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

        full_label = f"{label}  (r = {r:.3f}, slope = {slope:.2f})"
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

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Naive h² estimators
# ---------------------------------------------------------------------------

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


def compare_naive_estimators(
    pedigree_paths_per_scenario: list[list[Path]],
    labels: list[str],
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
        pedigree_paths_per_scenario: outer list = scenarios, inner list =
            per-rep recorded ``pedigree.parquet`` paths.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        input_h2: the simulation input h².  Drawn as a dashed reference line
            in the top panel; omitted if ``None``.
        min_generation: restrict all estimator computations (and the
            realized-h² reference) to ``generation >= min_generation``.
            ``None`` uses all generations.
    """
    n_scen = len(labels)
    if len(pedigree_paths_per_scenario) != n_scen:
        raise ValueError("pedigree_paths and labels must have the same length")

    apply_nature_style()
    estimator_labels = [d[0] for d in NAIVE_ESTIMATOR_DEFS]
    estimator_keys = [d[1] for d in NAIVE_ESTIMATOR_DEFS]
    n_est = len(estimator_keys)

    per_scen = [
        load_naive_estimator_h2(
            [Path(p) for p in ped_paths],
            trait=trait,
            min_generation=min_generation,
        )
        for ped_paths in pedigree_paths_per_scenario
    ]

    fig, (ax_raw, ax_bias) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = np.arange(n_est, dtype=float)
    total_group_width = 0.8
    bar_width = total_group_width / n_scen

    for scen_idx, (label, ests) in enumerate(zip(labels, per_scen, strict=True)):
        color = scenario_color(scen_idx)
        offsets = bar_offsets(x, total_group_width, n_scen, scen_idx)
        realized = ests.realized
        raw = [rep_envelope(ests.estimates[key]) for key in estimator_keys]
        # Per-rep bias vs. per-rep realized, then aggregate.
        bias = [rep_envelope(ests.estimates[key] - realized) for key in estimator_keys]
        draw_grouped_bars(ax_raw, offsets, raw, bar_width, color, label)
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

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Time-varying E and h² drift (cohort-level correlations + Falconer + prevalence)
# ---------------------------------------------------------------------------


def compare_components_by_generation(
    pedigree_paths_per_scenario: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    show_generations: tuple[int, ...] = (1, 5, 9),
    n_bins: int = 50,
) -> None:
    """Per-generation distributions of A and total liability, one column per scenario.

    Two rows × *n_scenarios* columns. Top row shows the additive genetic
    component A; bottom row shows total liability (A + C + E). Within each
    panel, distributions for *show_generations* are overlaid (lighter →
    darker shade for earlier → later gens) so the reader can see the
    cohort evolution of the per-individual distribution directly. The A
    panels should look essentially identical across both gens and
    scenarios (A is fixed by config); the liability panels widen,
    narrow, or stay put depending on the per-gen E schedule.

    Args:
        pedigree_paths_per_scenario: outer list = scenarios, inner list =
            per-rep recorded ``pedigree.parquet`` paths.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        show_generations: which generations to overlay within each panel.
            Default (1, 5, 9) bookends + middle for a 10-gen pedigree.
        n_bins: histogram bin count, shared across panels for comparability.
    """
    n_scen = len(labels)
    if len(pedigree_paths_per_scenario) != n_scen:
        raise ValueError("pedigree_paths and labels must have the same length")

    apply_nature_style()
    a_key = f"A{trait}"
    l_key = f"liability{trait}"

    # Pool reps within each (scenario, generation) cell.
    per_scen_a: list[dict[int, np.ndarray]] = []
    per_scen_l: list[dict[int, np.ndarray]] = []
    for paths in pedigree_paths_per_scenario:
        a_by_gen: dict[int, list[np.ndarray]] = {g: [] for g in show_generations}
        l_by_gen: dict[int, list[np.ndarray]] = {g: [] for g in show_generations}
        for path in paths:
            df = load_parquet(Path(path), columns=["generation", a_key, l_key])
            gen_arr = df["generation"].to_numpy()
            a_arr = df[a_key].to_numpy()
            l_arr = df[l_key].to_numpy()
            for g in show_generations:
                gen_mask = gen_arr == g
                if not gen_mask.any():
                    continue
                a_by_gen[g].append(a_arr[gen_mask])
                l_by_gen[g].append(l_arr[gen_mask])
        per_scen_a.append({g: np.concatenate(v) if v else np.array([]) for g, v in a_by_gen.items()})
        per_scen_l.append({g: np.concatenate(v) if v else np.array([]) for g, v in l_by_gen.items()})

    # Shared x-range across both rows + every panel for cross-comparability.
    lo, hi = padded_range(np.concatenate([arr for d in per_scen_a + per_scen_l for arr in d.values() if arr.size]))
    bins = np.linspace(lo, hi, n_bins + 1)

    # Lighter → darker shade for earlier → later gens, using a perceptually
    # uniform colormap so the gen ordering is unambiguous in the legend.
    cmap = plt.get_cmap("viridis")
    n_g = len(show_generations)
    gen_colors = [cmap(0.15 + 0.7 * (i / max(1, n_g - 1))) for i in range(n_g)]

    fig, axes = plt.subplots(2, n_scen, figsize=(3.2 * n_scen, 5.5), sharex=True, sharey="row")
    if n_scen == 1:
        axes = axes.reshape(2, 1)

    for col, (label, a_dict, l_dict) in enumerate(zip(labels, per_scen_a, per_scen_l, strict=True)):
        for row_idx, (ax, vals_dict) in enumerate(((axes[0, col], a_dict), (axes[1, col], l_dict))):
            for g_idx, g in enumerate(show_generations):
                vals = vals_dict.get(g, np.array([]))
                if vals.size == 0:
                    continue
                color = gen_colors[g_idx]
                sd = float(np.std(vals, ddof=1))
                ax.hist(
                    vals,
                    bins=bins,
                    density=True,
                    histtype="step",
                    color=color,
                    linewidth=1.6,
                    label=f"gen {g}  (sd = {sd:.3f})",
                )
            if row_idx == 0:
                ax.set_title(label, fontsize=10)
            enable_value_gridlines(ax)

        axes[0, col].legend(loc="upper left", fontsize=7, frameon=False)
        axes[1, col].legend(loc="upper left", fontsize=7, frameon=False)

    axes[0, 0].set_ylabel(f"Density — A{trait}")
    axes[1, 0].set_ylabel(f"Density — liability {trait}")
    for ax in axes[1, :]:
        ax.set_xlabel(f"Value (trait {trait})")

    fig.suptitle("Per-individual A and total liability by generation")
    _save(fig, output_path, rect=(0, 0, 1, 0.96))


def compare_cohort_fs_correlations(
    pedigree_paths_per_scenario: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    reference: SimulationParams | None = None,
    min_generation: int = 1,
) -> None:
    """Per-generation FS liability correlation, one line per scenario.

    Within each generation cohort, computes Pearson ``r`` on liability between
    full-sib pairs (both members in the same generation; twins already
    excluded by the extractor).  Plots ``r_FS(g)`` as a line per scenario
    with min/max envelope across replicates.

    Args:
        pedigree_paths_per_scenario: outer list = scenarios, inner list =
            per-rep recorded ``pedigree.parquet`` paths.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        reference: parameters of the reference scenario,
            usually the first (:func:`scenario_parameters`), for the FS
            reference line (:func:`~simace.core.relationships.reference_corr`).
            Omitted if ``None`` or when that scenario has no analytic FS value.
        min_generation: smallest generation to include (founders have no FS
            pairs by construction; default 1).
    """
    n_scen = len(labels)
    if len(pedigree_paths_per_scenario) != n_scen:
        raise ValueError("pedigree_paths and labels must have the same length")

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for scen_idx, (paths, label) in enumerate(zip(pedigree_paths_per_scenario, labels, strict=True)):
        per_rep = [load_pedigree_estimates_per_generation(Path(p), trait=trait) for p in paths]
        # Union of generations seen across reps, restricted to >= min_generation.
        gens = sorted({g for d in per_rep for g in d if g >= min_generation})
        draw_envelope_line(
            ax, gens, _per_gen_envelope(per_rep, gens, lambda cell: cell.r_fs), scenario_color(scen_idx), label
        )

    ref = (
        None
        if reference is None
        else reference_corr("FS", reference.assortment, reference.c_transmission, reference.vc, trait)
    )
    if ref is not None:
        ax.axhline(
            y=ref,
            linestyle="--",
            color="#888888",
            linewidth=1,
            alpha=0.8,
            label=f"expected FS r ({labels[0]}) = {ref:.3f}",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Within-cohort r(FS, liability {trait})")
    ax.set_title("Per-cohort full-sib liability correlation")
    enable_value_gridlines(ax)
    ax.legend(loc="best", fontsize=9, frameon=False)

    _save(fig, output_path)


def _per_gen_envelope(
    per_rep_pergen: list[dict[int, CohortEstimates]],
    gens: list[int],
    get_val: Callable[[CohortEstimates], float],
) -> list[Envelope]:
    """One :class:`Envelope` per generation of a per-rep cohort value; reps without the generation are skipped."""
    return [rep_envelope(get_val(d[g]) for d in per_rep_pergen if g in d) for g in gens]


def compare_cohort_falconer(
    pedigree_paths_per_scenario: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    min_generation: int = 1,
) -> None:
    """Per-generation realized h² vs per-gen Falconer vs pooled Falconer.

    Side-by-side panels (one per scenario, shared y-axis):
      - per-gen realized h² (variance ratio in the cohort)
      - per-gen Falconer = 2(r_MZ - r_FS) computed within the cohort
      - horizontal dashed line: pooled-across-gens Falconer (computed on
        the full pedigree).

    Args:
        pedigree_paths_per_scenario: outer list = scenarios, inner list =
            per-rep recorded ``pedigree.parquet`` paths.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: smallest generation to include in per-gen lines
            (founders have no FS pairs).  Pooled estimate always uses every
            individual.
    """
    n_scen = len(labels)
    if len(pedigree_paths_per_scenario) != n_scen:
        raise ValueError("pedigree_paths and labels must have the same length")

    apply_nature_style()
    fig, axes = plt.subplots(1, n_scen, figsize=(3.4 * n_scen, 4.5), sharey=True)
    if n_scen == 1:
        axes = np.array([axes])

    for paths, label, ax in zip(pedigree_paths_per_scenario, labels, axes, strict=True):
        per_rep_pergen = [load_pedigree_estimates_per_generation(Path(p), trait=trait) for p in paths]
        per_rep_pooled = [load_pedigree_estimates(Path(p), trait=trait, min_generation=None) for p in paths]
        gens = sorted({g for d in per_rep_pergen for g in d if g >= min_generation})

        truth = _per_gen_envelope(per_rep_pergen, gens, lambda d: d.realized_h2)
        falconer = _per_gen_envelope(per_rep_pergen, gens, lambda d: falconer_h2(d.r_mz, d.r_fs))
        draw_envelope_line(ax, gens, truth, scenario_color(0), "Realized h²")
        draw_envelope_line(ax, gens, falconer, scenario_color(1), "Per-cohort Falconer", marker="s")

        pooled_mean = rep_envelope(falconer_h2(est.r["MZ"], est.r["FS"]) for est in per_rep_pooled).mean
        if np.isfinite(pooled_mean):
            ax.axhline(
                y=pooled_mean,
                linestyle="--",
                color="#444444",
                linewidth=1.2,
                alpha=0.9,
                label=f"Pooled Falconer = {pooled_mean:.3f}",
            )

        ax.set_title(label)
        ax.set_xlabel("Generation")
        enable_value_gridlines(ax)
        ax.legend(loc="best", fontsize=8, frameon=False)

    axes[0].set_ylabel(f"h² (trait {trait})")
    fig.suptitle("Per-cohort vs pooled-across-gens naive Falconer")
    _save(fig, output_path, rect=(0, 0, 1, 0.96))


def _load_per_gen_prevalence(
    report_paths: list[Path],
    trait: int = 1,
) -> dict[int, list[float]]:
    """Read ``prevalence.by_generation.{N}.trait{trait}`` across replicates.

    Returns dict ``g -> [per-rep prevalence values]``.  Generations are the
    union of those present across reps; missing reps contribute nothing
    to a given generation's list.
    """
    per_gen: dict[int, list[float]] = {}
    for path in report_paths:
        ps = plotting_report_view(load_yaml(path))
        by_gen = ps.sample.prevalence.get("by_generation") or {}
        for g_key, entry in by_gen.items():
            g = int(g_key)
            val = entry.get(f"trait{trait}")
            if val is None:
                continue
            per_gen.setdefault(g, []).append(float(val))
    return per_gen


def compare_prevalence_drift(
    std_paths_per_trajectory: list[list[Path]],
    nostd_paths_per_trajectory: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    target_prevalence: float | None = None,
    std_label: str = "standardize=global",
    nostd_label: str = "standardize=none",
    pergen_paths_per_trajectory: list[list[Path]] | None = None,
    pergen_label: str = "standardize=per_generation",
) -> None:
    """Per-generation observed prevalence comparing two or three ``standardize`` modes.

    One panel per E trajectory (e.g. e_flat / e_rise_mild / e_rise_steep /
    e_fall_steep). Two lines per panel by default; if
    ``pergen_paths_per_trajectory`` is supplied, a third line is drawn for
    the ``per_generation`` mode (which preserves K per generation
    regardless of how :math:`v_E` drifts across cohorts).

    Args:
        std_paths_per_trajectory: outer list = trajectory, inner list =
            per-replicate ``report.yaml`` paths for the first variant.
        nostd_paths_per_trajectory: same shape, for the second variant.
        labels: display label per trajectory.
        output_path: image path to save.
        trait: 1 or 2.
        target_prevalence: input K to draw as dashed reference (typically
            0.1).  Omitted if ``None``.
        std_label: legend label for the ``std_paths_per_trajectory`` series.
        nostd_label: legend label for the ``nostd_paths_per_trajectory`` series.
        pergen_paths_per_trajectory: optional third series; when present,
            plotted as the per-generation comparison line.
        pergen_label: legend label for the third series.
    """
    n_traj = len(labels)
    if len(std_paths_per_trajectory) != n_traj or len(nostd_paths_per_trajectory) != n_traj:
        raise ValueError(
            "std_paths_per_trajectory, nostd_paths_per_trajectory, and labels must all have the same length"
        )
    if pergen_paths_per_trajectory is not None and len(pergen_paths_per_trajectory) != n_traj:
        raise ValueError("pergen_paths_per_trajectory must have the same length as labels")

    apply_nature_style()
    fig, axes = plt.subplots(1, n_traj, figsize=(3.4 * n_traj, 4.5), sharey=True)
    if n_traj == 1:
        axes = np.array([axes])

    series: list[tuple[str, str, list[list[Path]]]] = [
        (SCENARIO_PALETTE[0], std_label, std_paths_per_trajectory),
        (SCENARIO_PALETTE[1], nostd_label, nostd_paths_per_trajectory),
    ]
    if pergen_paths_per_trajectory is not None:
        series.append((SCENARIO_PALETTE[2], pergen_label, pergen_paths_per_trajectory))

    for traj_idx, (ax, traj_label) in enumerate(zip(axes, labels, strict=True)):
        for color, line_label, paths_per_traj in series:
            paths = paths_per_traj[traj_idx]
            per_gen = _load_per_gen_prevalence([Path(p) for p in paths], trait=trait)
            gens = sorted(per_gen.keys())
            draw_envelope_line(ax, gens, [rep_envelope(per_gen[g]) for g in gens], color, line_label)

        if target_prevalence is not None:
            ax.axhline(
                y=target_prevalence,
                linestyle="--",
                color="#888888",
                linewidth=1,
                alpha=0.8,
            )

        ax.set_title(traj_label)
        ax.set_xlabel("Generation")
        enable_value_gridlines(ax)
        ax.legend(loc="best", fontsize=8, frameon=False)

    axes[0].set_ylabel(f"Observed prevalence (trait {trait})")
    title_labels = [s[1] for s in series]
    fig.suptitle("Per-generation prevalence: " + " vs ".join(title_labels))
    _save(fig, output_path, rect=(0, 0, 1, 0.96))


# ---------------------------------------------------------------------------
# Observed-scale vs liability-scale h² (phenotype-model comparison)
# ---------------------------------------------------------------------------

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


def _load_tetrachoric(stats_report_path: Path, trait: int) -> dict[str, float]:
    """Tetrachoric r on affected status for every relationship code; NaN where the report has none."""
    ps = plotting_report_view(load_yaml(stats_report_path))
    tet = ps.sample.tetrachoric.get(f"trait{trait}", {}) or {}
    out: dict[str, float] = {}
    for code in RELATIONSHIP_TYPES:
        r = (tet.get(code) or {}).get("r")
        out[code] = float(r) if r is not None else float("nan")
    return out


def compare_observed_vs_liability_h2(
    pedigree_paths_per_scenario: list[list[Path]],
    report_paths_per_scenario: list[list[Path]],
    labels: list[str],
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
        pedigree_paths_per_scenario: outer list = scenarios, inner list =
            per-rep recorded ``pedigree.parquet`` paths.
        report_paths_per_scenario: same shape, per-replicate
            ``report.yaml`` paths. Rep order must match.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: forwarded to
            :func:`~simace.analysis.stats.pedigree_estimates.load_pedigree_estimates`.
        input_h2: simulation input h²; drawn as a dashed reference line.
    """
    n_scen = len(labels)
    if len(pedigree_paths_per_scenario) != n_scen or len(report_paths_per_scenario) != n_scen:
        raise ValueError("pedigree_paths, report_paths, and labels must have the same length")

    apply_nature_style()
    estimator_labels = [d[0] for d in OBSERVED_LIABILITY_ESTIMATOR_DEFS]
    estimator_keys = [d[1] for d in OBSERVED_LIABILITY_ESTIMATOR_DEFS]
    n_est = len(estimator_keys)

    per_scen = [
        load_observed_vs_liability_h2(
            [Path(p) for p in ped_paths],
            [_load_tetrachoric(Path(p), trait=trait) for p in ps_paths],
            trait=trait,
            min_generation=min_generation,
        )
        for ped_paths, ps_paths in zip(pedigree_paths_per_scenario, report_paths_per_scenario, strict=True)
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
    ax_raw.set_xticklabels(labels)
    ax_raw.tick_params(axis="x", labelbottom=True)
    ax_bias.tick_params(axis="x", labelbottom=False)
    ax_raw.set_ylabel(f"h² estimate (trait {trait})")
    ax_bias.set_ylabel("Bias vs. realized h² (estimate - truth)")
    ax_raw.set_title("Observed-scale vs liability-scale h² by phenotype model")
    ax_bias.set_title("Bias vs realized h²")
    enable_value_gridlines(ax_raw)
    enable_value_gridlines(ax_bias)
    ax_bias.legend(loc="lower left", fontsize=9, frameon=False)

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Legacy CLI entry (variance trajectory only)
# ---------------------------------------------------------------------------


def main(
    scenario_paths: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    expected_A: float | None = None,
    expected_C: float | None = None,
    expected_E: float | None = None,
) -> None:
    """Library entry point for the example-comparison scripts."""
    compare_realized_variance_trajectory(
        scenario_paths=scenario_paths,
        labels=labels,
        output_path=output_path,
        trait=trait,
        expected_A=expected_A,
        expected_C=expected_C,
        expected_E=expected_E,
    )
