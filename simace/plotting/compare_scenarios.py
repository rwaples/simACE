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
    "POOLED_RELATIONSHIP_CLASSES",
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
    "load_naive_estimator_h2",
    "load_observed_vs_liability_h2",
    "load_pedigree_estimates",
    "load_pedigree_estimates_per_generation",
    "load_per_generation",
    "load_pooled_liability_correlations",
    "load_sib_pair_liabilities",
]

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats as sci_stats

from simace.core.pedigree_graph import extract_relationship_pairs
from simace.core.utils import PAIR_TYPES
from simace.plotting.plot_style import (
    apply_nature_style,
    enable_value_gridlines,
)

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
    validation_paths: list[Path],
    trait: int = 1,
) -> dict[int, np.ndarray]:
    """Read ``per_generation`` variance components from validation.yaml files.

    Args:
        validation_paths: one path per replicate (of a single scenario).
        trait: 1 or 2.

    Returns:
        Dict keyed by generation (1-indexed, matching the YAML keys) whose
        values are (n_reps, 4) arrays of columns ``[vA, vC, vE, h2]``.  ``h2``
        is ``vA / (vA + vC + vE)``.  Generations missing from one rep raise
        ``KeyError``; this is intentional since the comparison plot needs a
        consistent gen axis.
    """
    per_rep: list[dict[int, tuple[float, float, float, float]]] = []
    for path in validation_paths:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        per_gen = data.get("per_generation", {})
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


def _mean_envelope(arr: np.ndarray, column: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, low, high) across reps for one column of ``load_per_generation`` output.

    For a small number of reps (typical: 3) a parametric CI is less honest
    than min/max across reps, so we use min/max as the envelope.
    """
    vals = arr[:, column]
    return vals.mean(), vals.min(), vals.max()


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
            ``validation.yaml`` paths for that scenario.
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
            n = max(
                len(e) if isinstance(e, list) else 1
                for e in (expected_A, expected_C, expected_E)
            )
            a_list = expected_A if isinstance(expected_A, list) else [expected_A] * n
            c_list = expected_C if isinstance(expected_C, list) else [expected_C] * n
            e_list = expected_E if isinstance(expected_E, list) else [expected_E] * n
            h2_expected = [
                a / (a + c + e) if (a + c + e) > 0 else None
                for a, c, e in zip(a_list, c_list, e_list, strict=True)
            ]
        else:
            total = expected_A + expected_C + expected_E
            h2_expected = expected_A / total if total > 0 else None

    apply_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    panel_defs = [
        (axes[0, 0], 0, f"Realized vA (trait {trait})", expected_A),
        (axes[0, 1], 1, f"Realized vC (trait {trait})", expected_C),
        (axes[1, 0], 2, f"Realized vE (trait {trait})", expected_E),
        (axes[1, 1], 3, f"Realized h² (trait {trait})", h2_expected),
    ]

    for scen_idx, (reps, label) in enumerate(zip(scenario_paths, labels, strict=True)):
        reps = [Path(p) for p in reps]
        per_gen = load_per_generation(reps, trait=trait)
        gens = sorted(per_gen.keys())
        color = SCENARIO_PALETTE[scen_idx % len(SCENARIO_PALETTE)]
        for ax, col, _title, _expected in panel_defs:
            means, lows, highs = [], [], []
            for gen in gens:
                mean, lo, hi = _mean_envelope(per_gen[gen], col)
                means.append(mean)
                lows.append(lo)
                highs.append(hi)
            ax.plot(gens, means, color=color, marker="o", label=label)
            ax.fill_between(gens, lows, highs, color=color, alpha=0.15, linewidth=0)

    for ax, _col, title, expected in panel_defs:
        ax.set_title(title)
        ax.set_ylabel("Variance" if _col < 3 else "h²")
        enable_value_gridlines(ax)
        if expected is None:
            continue
        if isinstance(expected, list):
            # Per-generation reference; align to gens 1..len (matches
            # validation.yaml's 1-indexed generation_N keys).
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

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


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
            ``pedigree.parquet`` paths.
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
            df = pd.read_parquet(Path(path), columns=["generation", a_key, l_key])
            if min_generation is not None:
                df = df[df["generation"] >= min_generation]
            a_parts.append(df[a_key].to_numpy())
            l_parts.append(df[l_key].to_numpy())
        per_scen_a.append(np.concatenate(a_parts))
        per_scen_l.append(np.concatenate(l_parts))

    # Shared x-axis across both panels makes the relative widths (A is
    # tighter than liability, because liability = A + E) directly legible.
    combined = np.concatenate([np.concatenate(per_scen_a), np.concatenate(per_scen_l)])
    lo = float(np.quantile(combined, 0.001))
    hi = float(np.quantile(combined, 0.999))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, (ax_a, ax_l) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for scen_idx, (a_vals, l_vals, label) in enumerate(zip(per_scen_a, per_scen_l, labels, strict=True)):
        color = SCENARIO_PALETTE[scen_idx % len(SCENARIO_PALETTE)]
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

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Liability correlations by relationship class
# ---------------------------------------------------------------------------

# Display-ordered relationship classes, pooled from the raw classes in
# phenotype_stats.yaml.  Expected liability correlation under random mating is
# ``k * A + c * C`` where k is the kinship coefficient; the middle column
# below is k so callers can draw reference bars.  MZ twins are deliberately
# omitted — their liability correlation is pinned at ``A + C`` regardless of
# AM, so including them washes out the visual story that this plot tells.
POOLED_RELATIONSHIP_CLASSES: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("FS", 0.5, ("FS",)),
    ("PO", 0.5, ("MO", "FO")),
    ("HS", 0.25, ("MHS", "PHS")),
    ("1C", 0.125, ("1C",)),
)


def load_pedigree_estimates(
    pedigree_path: Path,
    trait: int = 1,
    min_generation: int | None = None,
) -> dict[str, float]:
    """Per-rep liability correlations, midparent-PO slope, and realized h².

    Everything is computed on a single replicate's ``pedigree.parquet`` —
    not on pre-aggregated YAML — so that a generation filter can be applied
    cleanly.  When ``min_generation`` is set, all liability correlations
    and the realized h² are computed on the subset of individuals in
    ``generation >= min_generation``.  Pair extraction uses the *full*
    pedigree so that cousins of filtered-subset individuals can still be
    found via their grandparents in earlier generations.  The
    midparent-offspring regression uses offspring in the filtered subset
    but reads parent liabilities from the full pedigree (parents may be in
    earlier generations).

    Args:
        pedigree_path: one ``pedigree.parquet`` path (one replicate).
        trait: 1 or 2.
        min_generation: keep individuals with ``generation >= min_generation``
            (inclusive).  ``None`` keeps every generation.

    Returns:
        Flat dict with keys for each of the seven raw relationship classes
        (``MZ``, ``FS``, ``MO``, ``FO``, ``MHS``, ``PHS``, ``1C``) holding
        liability correlations, plus ``po_slope`` (midparent-offspring
        regression slope), ``realized_h2`` (in the filtered subset), and
        the raw variance components ``vA``, ``vC``, ``vE`` used to compute
        it.  Values are ``NaN`` where there aren't enough pairs.
    """
    cols = [
        "id",
        "sex",
        "mother",
        "father",
        "twin",
        "generation",
        f"A{trait}",
        f"C{trait}",
        f"E{trait}",
        f"liability{trait}",
    ]
    df_full = pd.read_parquet(pedigree_path, columns=cols)
    if min_generation is not None:
        df = df_full[df_full["generation"] >= min_generation].reset_index(drop=True)
    else:
        df = df_full.reset_index(drop=True)

    pairs = extract_relationship_pairs(df, full_pedigree=df_full, max_degree=2)
    liab = df[f"liability{trait}"].to_numpy()

    corrs: dict[str, float] = {}
    for ptype in PAIR_TYPES:
        idx1, idx2 = pairs.get(ptype, (np.array([]), np.array([])))
        if len(idx1) < 10:
            corrs[ptype] = float("nan")
        else:
            corrs[ptype] = float(np.corrcoef(liab[idx1], liab[idx2])[0, 1])

    # Midparent-offspring regression: offspring from filtered subset, parent
    # liabilities looked up in the full pedigree (may be pre-filter).
    id_to_liab = pd.Series(df_full[f"liability{trait}"].to_numpy(), index=df_full["id"].to_numpy())
    sub = df[(df["mother"] >= 0) & (df["father"] >= 0)]
    if len(sub) >= 10:
        mother_liab = id_to_liab.reindex(sub["mother"].to_numpy()).to_numpy()
        father_liab = id_to_liab.reindex(sub["father"].to_numpy()).to_numpy()
        offspring = sub[f"liability{trait}"].to_numpy()
        midparent = (mother_liab + father_liab) / 2.0
        mask = np.isfinite(midparent) & np.isfinite(offspring)
        if mask.sum() >= 10 and np.var(midparent[mask]) > 0:
            reg = sci_stats.linregress(midparent[mask], offspring[mask])
            po_slope = float(reg.slope)
        else:
            po_slope = float("nan")
    else:
        po_slope = float("nan")

    vA = float(df[f"A{trait}"].var(ddof=1))
    vC = float(df[f"C{trait}"].var(ddof=1))
    vE = float(df[f"E{trait}"].var(ddof=1))
    total = vA + vC + vE
    realized_h2 = vA / total if total > 0 else float("nan")

    return {**corrs, "po_slope": po_slope, "realized_h2": realized_h2, "vA": vA, "vC": vC, "vE": vE}


def load_pooled_liability_correlations(
    pedigree_paths: list[Path],
    trait: int = 1,
    min_generation: int | None = None,
) -> dict[str, list[float]]:
    """Read per-rep liability correlations pooled to FS/PO/HS/1C.

    Thin aggregator over :func:`load_pedigree_estimates`: pools MO+FO → PO
    and MHS+PHS → HS (equal-weight average) so the chart has four x-axis
    categories instead of seven.

    Args:
        pedigree_paths: one ``pedigree.parquet`` path per replicate of a
            single scenario.
        trait: 1 or 2.
        min_generation: passed through to :func:`load_pedigree_estimates`.

    Returns:
        Dict keyed by pooled class name whose values are lists of per-rep
        correlations.  ``NaN`` entries are skipped rather than propagated.
    """
    out: dict[str, list[float]] = {cls: [] for cls, _, _ in POOLED_RELATIONSHIP_CLASSES}
    for path in pedigree_paths:
        est = load_pedigree_estimates(path, trait=trait, min_generation=min_generation)
        for pooled_name, _kinship, members in POOLED_RELATIONSHIP_CLASSES:
            vals = [est.get(m, float("nan")) for m in members]
            finite = [float(v) for v in vals if np.isfinite(v)]
            if finite:
                out[pooled_name].append(float(np.mean(finite)))
    return out


def compare_correlations_by_relclass(
    scenario_paths: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    expected_A: float | None = None,
    expected_C: float | None = None,
    min_generation: int | None = None,
) -> None:
    """Grouped bar chart: liability correlation by relationship class, per scenario.

    Relationship classes are pooled to {FS, PO, HS, 1C} and displayed on
    the x-axis in decreasing expected kinship.  Each scenario gets one bar
    per class, drawn with min/max error bars across replicates.  If
    ``expected_A`` is supplied, a dashed reference line per class shows the
    random-mating expectation ``k * A + c * C`` (where ``k`` is the kinship
    coefficient for that class); this is what any AM scenario should deviate
    from as AM strength grows.

    Args:
        scenario_paths: outer list = scenarios, inner list = replicate
            ``pedigree.parquet`` paths for that scenario.
        labels: display label per scenario (same order as ``scenario_paths``).
        output_path: image path to save.
        trait: 1 or 2.
        expected_A: input ``A`` for the trait; used for random-mating reference
            markers.  If ``None``, no reference markers are drawn.
        expected_C: input ``C`` for the trait.  Defaults to 0 if ``None`` and
            ``expected_A`` is set.
        min_generation: restrict correlation computations to individuals in
            generations ``>= min_generation``.  Useful for letting AM reach
            its equilibrium.  ``None`` uses all generations.
    """
    if len(scenario_paths) != len(labels):
        raise ValueError(f"scenario_paths ({len(scenario_paths)}) and labels ({len(labels)}) must match")

    apply_nature_style()
    class_names = [c[0] for c in POOLED_RELATIONSHIP_CLASSES]
    kinship = {c[0]: c[1] for c in POOLED_RELATIONSHIP_CLASSES}
    n_classes = len(class_names)
    n_scen = len(labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_classes, dtype=float)
    total_group_width = 0.8
    bar_width = total_group_width / n_scen

    # First pass: compute per-scenario means/lows/highs so we can use the
    # baseline (first scenario) means to annotate non-baseline bars with
    # absolute and relative inflation.
    per_scen_means = np.full((n_scen, n_classes), np.nan)
    per_scen_lows = np.full((n_scen, n_classes), np.nan)
    per_scen_highs = np.full((n_scen, n_classes), np.nan)
    per_scen_offsets = np.full((n_scen, n_classes), np.nan)
    for scen_idx, reps in enumerate(scenario_paths):
        per_class = load_pooled_liability_correlations(
            [Path(p) for p in reps], trait=trait, min_generation=min_generation
        )
        for i, cls in enumerate(class_names):
            vals = per_class[cls]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            per_scen_means[scen_idx, i] = arr.mean()
            per_scen_lows[scen_idx, i] = arr.min()
            per_scen_highs[scen_idx, i] = arr.max()
        per_scen_offsets[scen_idx] = x - total_group_width / 2 + bar_width / 2 + scen_idx * bar_width

    # Second pass: draw bars.
    for scen_idx, label in enumerate(labels):
        means = per_scen_means[scen_idx]
        lows = per_scen_lows[scen_idx]
        highs = per_scen_highs[scen_idx]
        offsets = per_scen_offsets[scen_idx]
        color = SCENARIO_PALETTE[scen_idx % len(SCENARIO_PALETTE)]
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

    # Third pass: annotate each non-baseline bar with inflation vs scenario 0.
    baseline = per_scen_means[0]
    for scen_idx in range(1, n_scen):
        means = per_scen_means[scen_idx]
        highs = per_scen_highs[scen_idx]
        offsets = per_scen_offsets[scen_idx]
        color = SCENARIO_PALETTE[scen_idx % len(SCENARIO_PALETTE)]
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

    if expected_A is not None:
        c = 0.0 if expected_C is None else expected_C
        for i, cls in enumerate(class_names):
            expected = kinship[cls] * expected_A + c
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Sib-pair liability scatter
# ---------------------------------------------------------------------------


def load_sib_pair_liabilities(
    pedigree_paths: list[Path],
    trait: int = 1,
    min_generation: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of paired sibling liabilities across all reps.

    Picks two full (DZ) sibs per family — the first two by ``id`` within each
    ``(mother, father)`` group.  Founders (mother == -1) and MZ twins
    (``twin != -1``) are excluded, so the returned pairs have an expected
    kinship of exactly 0.5 under random mating.

    Args:
        pedigree_paths: one ``pedigree.parquet`` path per replicate.
        trait: 1 or 2.
        min_generation: if set, restrict to individuals in generations
            ``>= min_generation``; useful for waiting out AM's burn-in period.

    Returns:
        ``(liab_a, liab_b)`` 1-D arrays of equal length, one element per family.
    """
    liab_col = f"liability{trait}"
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for path in pedigree_paths:
        df = pd.read_parquet(
            path,
            columns=["id", "mother", "father", "twin", "generation", liab_col],
        )
        df = df[(df["mother"] >= 0) & (df["twin"] == -1)]
        if min_generation is not None:
            df = df[df["generation"] >= min_generation]
        df = df.sort_values(["mother", "father", "id"], kind="stable")
        rank = df.groupby(["mother", "father"], sort=False).cumcount()
        first = df.loc[rank == 0, ["mother", "father", liab_col]].rename(columns={liab_col: "a"})
        second = df.loc[rank == 1, ["mother", "father", liab_col]].rename(columns={liab_col: "b"})
        pairs = first.merge(second, on=["mother", "father"], how="inner")
        a_list.append(pairs["a"].to_numpy(dtype=float, copy=False))
        b_list.append(pairs["b"].to_numpy(dtype=float, copy=False))
    return np.concatenate(a_list), np.concatenate(b_list)


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
            ``pedigree.parquet`` paths.
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
    lo = float(np.quantile(combined, 0.001))
    hi = float(np.quantile(combined, 0.999))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    for scen_idx, ((liab_a, liab_b), label) in enumerate(zip(loaded, labels, strict=True)):
        if liab_a.size == 0:
            continue
        color = SCENARIO_PALETTE[scen_idx % len(SCENARIO_PALETTE)]
        r = float(np.corrcoef(liab_a, liab_b)[0, 1])
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

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Naive h² estimators
# ---------------------------------------------------------------------------

# (display label, internal key).  Each estimator maps one correlation (or
# regression slope) into an h² estimate that is unbiased under random mating
# but drifts under AM, by different amounts for different relationship
# classes.
NAIVE_ESTIMATOR_DEFS: tuple[tuple[str, str], ...] = (
    ("Twin Falconer", "falconer"),
    ("Sibs only", "sibs"),
    ("Midparent PO", "po"),
    ("Half-sibs", "hs"),
    ("Cousins", "1c"),
)


def load_naive_estimator_h2(
    pedigree_paths: list[Path],
    trait: int = 1,
    min_generation: int | None = None,
) -> dict[str, np.ndarray]:
    """Per-rep h² from five naive estimators + per-rep realized h².

    All quantities are computed directly from ``pedigree.parquet`` so the
    generation filter applies consistently to every estimator *and* to the
    realized-h² reference.

    Args:
        pedigree_paths: one ``pedigree.parquet`` path per replicate.
        trait: 1 or 2.
        min_generation: passed through to :func:`load_pedigree_estimates`.

    Returns:
        Dict keyed by ``{'falconer', 'sibs', 'po', 'hs', '1c', 'realized'}``,
        each an array of per-rep floats (NaN where there weren't enough
        pairs of a given type).
    """
    out: dict[str, list[float]] = {k: [] for k in ("falconer", "sibs", "po", "hs", "1c", "realized")}
    for path in pedigree_paths:
        est = load_pedigree_estimates(path, trait=trait, min_generation=min_generation)
        r_mz = est.get("MZ", float("nan"))
        r_fs = est.get("FS", float("nan"))
        r_mhs = est.get("MHS", float("nan"))
        r_phs = est.get("PHS", float("nan"))
        r_1c = est.get("1C", float("nan"))

        falconer = 2.0 * (r_mz - r_fs) if np.isfinite(r_mz) and np.isfinite(r_fs) else float("nan")
        out["falconer"].append(falconer)
        out["sibs"].append(2.0 * r_fs if np.isfinite(r_fs) else float("nan"))
        out["po"].append(est.get("po_slope", float("nan")))
        hs_vals = [r for r in (r_mhs, r_phs) if np.isfinite(r)]
        out["hs"].append(4.0 * float(np.mean(hs_vals)) if hs_vals else float("nan"))
        out["1c"].append(8.0 * r_1c if np.isfinite(r_1c) else float("nan"))
        out["realized"].append(est.get("realized_h2", float("nan")))

    return {k: np.asarray(vals, dtype=float) for k, vals in out.items()}


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
            per-rep ``pedigree.parquet`` paths.
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

    per_scen: list[dict[str, np.ndarray]] = [
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

    def _stats(arr: np.ndarray) -> tuple[float, float, float]:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return float("nan"), float("nan"), float("nan")
        return float(valid.mean()), float(valid.min()), float(valid.max())

    for scen_idx, (label, ests) in enumerate(zip(labels, per_scen, strict=True)):
        color = SCENARIO_PALETTE[scen_idx % len(SCENARIO_PALETTE)]
        offsets = x - total_group_width / 2 + bar_width / 2 + scen_idx * bar_width

        raw_means = np.full(n_est, np.nan)
        raw_lows = np.full(n_est, np.nan)
        raw_highs = np.full(n_est, np.nan)
        bias_means = np.full(n_est, np.nan)
        bias_lows = np.full(n_est, np.nan)
        bias_highs = np.full(n_est, np.nan)
        realized = ests["realized"]
        for i, key in enumerate(estimator_keys):
            vals = ests[key]
            raw_means[i], raw_lows[i], raw_highs[i] = _stats(vals)
            # Per-rep bias vs. per-rep realized, then aggregate.
            if vals.size == realized.size:
                bias = vals - realized
                bias_means[i], bias_lows[i], bias_highs[i] = _stats(bias)

        for ax, means, lows, highs in (
            (ax_raw, raw_means, raw_lows, raw_highs),
            (ax_bias, bias_means, bias_lows, bias_highs),
        ):
            mask = np.isfinite(means)
            err_low = np.where(mask, means - lows, 0.0)
            err_high = np.where(mask, highs - means, 0.0)
            ax.bar(
                offsets[mask],
                means[mask],
                width=bar_width * 0.95,
                color=color,
                label=label if ax is ax_raw else None,
                yerr=[err_low[mask], err_high[mask]],
                capsize=3,
                linewidth=0,
            )

        # Short horizontal tick at each scenario's realized h² in the top
        # panel, spanning just that scenario's bar slice of each group.
        realized_mean, _, _ = _stats(realized)
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

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


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
            per-rep ``pedigree.parquet`` paths.
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
            df = pd.read_parquet(Path(path), columns=["generation", a_key, l_key])
            for g in show_generations:
                sub = df[df["generation"] == g]
                if len(sub) == 0:
                    continue
                a_by_gen[g].append(sub[a_key].to_numpy())
                l_by_gen[g].append(sub[l_key].to_numpy())
        per_scen_a.append({g: np.concatenate(v) if v else np.array([]) for g, v in a_by_gen.items()})
        per_scen_l.append({g: np.concatenate(v) if v else np.array([]) for g, v in l_by_gen.items()})

    # Shared x-range across both rows + every panel for cross-comparability.
    all_vals = np.concatenate(
        [arr for d in per_scen_a + per_scen_l for arr in d.values() if arr.size]
    )
    lo = float(np.quantile(all_vals, 0.001))
    hi = float(np.quantile(all_vals, 0.999))
    pad = 0.05 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, n_bins + 1)

    # Lighter → darker shade for earlier → later gens, using a perceptually
    # uniform colormap so the gen ordering is unambiguous in the legend.
    cmap = plt.get_cmap("viridis")
    n_g = len(show_generations)
    gen_colors = [cmap(0.15 + 0.7 * (i / max(1, n_g - 1))) for i in range(n_g)]

    fig, axes = plt.subplots(2, n_scen, figsize=(3.2 * n_scen, 5.5), sharex=True, sharey="row")
    if n_scen == 1:
        axes = axes.reshape(2, 1)

    for col, (label, a_dict, l_dict) in enumerate(
        zip(labels, per_scen_a, per_scen_l, strict=True)
    ):
        for row_idx, (ax, vals_dict) in enumerate(
            ((axes[0, col], a_dict), (axes[1, col], l_dict))
        ):
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
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def load_pedigree_estimates_per_generation(
    pedigree_path: Path,
    trait: int = 1,
    gens: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Per-generation cohort liability correlations and realized h².

    For each generation ``g``, the dataframe is filtered to ``generation == g``
    (a single cohort) and MZ/FS pair correlations are computed on that cohort
    only.  ``extract_relationship_pairs`` is called on the filtered cohort
    so the returned FS pairs already exclude twins
    (``simace/core/pedigree_graph.py:_sibling_pairs`` filters ``twin != -1``).
    MZ pairs are read from the cohort directly.

    Args:
        pedigree_path: one ``pedigree.parquet`` path (one replicate).
        trait: 1 or 2.
        gens: which generations to compute for; defaults to every generation
            present in the pedigree with ``>= 1`` non-founder individual.

    Returns:
        Dict ``g -> {MZ, FS, n_MZ, n_FS, realized_h2, vA, vC, vE}``.  Values
        are ``NaN`` (or ``0`` for counts) where there aren't enough pairs.
    """
    cols = [
        "id",
        "sex",
        "mother",
        "father",
        "twin",
        "generation",
        f"A{trait}",
        f"C{trait}",
        f"E{trait}",
        f"liability{trait}",
    ]
    df_full = pd.read_parquet(pedigree_path, columns=cols)
    if gens is None:
        gens = sorted(int(g) for g in df_full["generation"].unique())

    out: dict[int, dict[str, float]] = {}
    for g in gens:
        df = df_full[df_full["generation"] == g].reset_index(drop=True)
        if len(df) == 0:
            out[g] = {
                "MZ": float("nan"), "FS": float("nan"),
                "n_MZ": 0, "n_FS": 0,
                "realized_h2": float("nan"),
                "vA": float("nan"), "vC": float("nan"), "vE": float("nan"),
            }
            continue

        pairs = extract_relationship_pairs(df, full_pedigree=df_full, max_degree=1)
        liab = df[f"liability{trait}"].to_numpy()
        cohort_corrs: dict[str, tuple[float, int]] = {}
        for code in ("MZ", "FS"):
            idx1, idx2 = pairs.get(code, (np.array([]), np.array([])))
            n_pairs = len(idx1)
            if n_pairs < 10:
                cohort_corrs[code] = (float("nan"), n_pairs)
            else:
                cohort_corrs[code] = (float(np.corrcoef(liab[idx1], liab[idx2])[0, 1]), n_pairs)
        r_mz, n_mz = cohort_corrs["MZ"]
        r_fs, n_fs = cohort_corrs["FS"]

        vA = float(df[f"A{trait}"].var(ddof=1))
        vC = float(df[f"C{trait}"].var(ddof=1))
        vE = float(df[f"E{trait}"].var(ddof=1))
        total = vA + vC + vE
        realized_h2 = vA / total if total > 0 else float("nan")

        out[g] = {
            "MZ": r_mz, "FS": r_fs,
            "n_MZ": n_mz, "n_FS": n_fs,
            "realized_h2": realized_h2,
            "vA": vA, "vC": vC, "vE": vE,
        }
    return out


def compare_cohort_fs_correlations(
    pedigree_paths_per_scenario: list[list[Path]],
    labels: list[str],
    output_path: Path,
    trait: int = 1,
    expected_A: float | None = None,
    expected_C: float | None = None,
    min_generation: int = 1,
) -> None:
    """Per-generation FS liability correlation, one line per scenario.

    Within each generation cohort, computes Pearson ``r`` on liability between
    full-sib pairs (both members in the same generation; twins already
    excluded by the extractor).  Plots ``r_FS(g)`` as a line per scenario
    with min/max envelope across replicates.

    Args:
        pedigree_paths_per_scenario: outer list = scenarios, inner list =
            per-rep ``pedigree.parquet`` paths.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        expected_A: input ``A`` for the random-mating reference line at
            ``0.5 * A`` (set ``C`` to non-None to include the C contribution
            ``0.5*A + C``).  Omitted if ``None``.
        expected_C: input ``C``; combined with ``expected_A`` for the
            reference line.
        min_generation: smallest generation to include (founders have no FS
            pairs by construction; default 1).
    """
    n_scen = len(labels)
    if len(pedigree_paths_per_scenario) != n_scen:
        raise ValueError("pedigree_paths and labels must have the same length")

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for scen_idx, (paths, label) in enumerate(zip(pedigree_paths_per_scenario, labels, strict=True)):
        per_rep = [
            load_pedigree_estimates_per_generation(Path(p), trait=trait)
            for p in paths
        ]
        # Union of generations seen across reps, restricted to >= min_generation.
        gens = sorted({g for d in per_rep for g in d if g >= min_generation})
        means, lows, highs = [], [], []
        for g in gens:
            vals = np.array(
                [d[g]["FS"] for d in per_rep if g in d and np.isfinite(d[g]["FS"])],
                dtype=float,
            )
            if vals.size == 0:
                means.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
            else:
                means.append(float(vals.mean()))
                lows.append(float(vals.min()))
                highs.append(float(vals.max()))
        color = SCENARIO_PALETTE[scen_idx % len(SCENARIO_PALETTE)]
        ax.plot(gens, means, color=color, marker="o", label=label)
        ax.fill_between(gens, lows, highs, color=color, alpha=0.15, linewidth=0)

    if expected_A is not None:
        ref = 0.5 * expected_A + (expected_C or 0.0)
        ax.axhline(
            y=ref,
            linestyle="--",
            color="#888888",
            linewidth=1,
            alpha=0.8,
            label=f"random-mating expectation = 0.5·A{(' + C' if expected_C else '')} = {ref:.3f}",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Within-cohort r(FS, liability {trait})")
    ax.set_title("Per-cohort full-sib liability correlation")
    enable_value_gridlines(ax)
    ax.legend(loc="best", fontsize=9, frameon=False)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def _per_gen_envelope(
    per_rep_pergen: list[dict[int, dict[str, float]]],
    gens: list[int],
    get_val,
) -> tuple[list[float], list[float], list[float]]:
    """Aggregate a per-rep, per-gen scalar to (mean, min, max) per generation.

    Skips reps that don't have an entry for a given generation, and skips
    NaN values within a generation.  Returns NaN-padded lists where no rep
    contributed a finite value.
    """
    means, lows, highs = [], [], []
    for g in gens:
        vals = np.array(
            [get_val(d[g]) for d in per_rep_pergen if g in d],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            means.append(float("nan"))
            lows.append(float("nan"))
            highs.append(float("nan"))
        else:
            means.append(float(vals.mean()))
            lows.append(float(vals.min()))
            highs.append(float(vals.max()))
    return means, lows, highs


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
            per-rep ``pedigree.parquet`` paths.
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
        per_rep_pergen = [
            load_pedigree_estimates_per_generation(Path(p), trait=trait)
            for p in paths
        ]
        per_rep_pooled = [
            load_pedigree_estimates(Path(p), trait=trait, min_generation=None)
            for p in paths
        ]
        gens = sorted({g for d in per_rep_pergen for g in d if g >= min_generation})

        truth_m, truth_lo, truth_hi = _per_gen_envelope(
            per_rep_pergen, gens, lambda d: d["realized_h2"]
        )
        falc_m, falc_lo, falc_hi = _per_gen_envelope(
            per_rep_pergen,
            gens,
            lambda d: 2.0 * (d["MZ"] - d["FS"])
            if np.isfinite(d["MZ"]) and np.isfinite(d["FS"])
            else float("nan"),
        )

        truth_color = SCENARIO_PALETTE[0]
        falc_color = SCENARIO_PALETTE[1]

        ax.plot(gens, truth_m, color=truth_color, marker="o", label="Realized h²")
        ax.fill_between(gens, truth_lo, truth_hi, color=truth_color, alpha=0.15, linewidth=0)
        ax.plot(gens, falc_m, color=falc_color, marker="s", label="Per-cohort Falconer")
        ax.fill_between(gens, falc_lo, falc_hi, color=falc_color, alpha=0.15, linewidth=0)

        pooled_falconer_vals = np.array(
            [
                2.0 * (est.get("MZ", float("nan")) - est.get("FS", float("nan")))
                for est in per_rep_pooled
            ],
            dtype=float,
        )
        pooled_falconer_vals = pooled_falconer_vals[np.isfinite(pooled_falconer_vals)]
        if pooled_falconer_vals.size:
            pooled_mean = float(pooled_falconer_vals.mean())
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
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def _load_per_gen_prevalence(
    phenotype_stats_paths: list[Path],
    trait: int = 1,
) -> dict[int, list[float]]:
    """Read ``prevalence.by_generation.{N}.trait{trait}`` across replicates.

    Returns dict ``g -> [per-rep prevalence values]``.  Generations are the
    union of those present across reps; missing reps contribute nothing
    to a given generation's list.
    """
    per_gen: dict[int, list[float]] = {}
    for path in phenotype_stats_paths:
        with open(path) as fh:
            ps = yaml.safe_load(fh) or {}
        by_gen = (ps.get("prevalence") or {}).get("by_generation") or {}
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
) -> None:
    """Per-generation observed prevalence under standardize=true vs false.

    One panel per E trajectory (e.g. e_flat / e_rise_mild / e_rise_steep /
    e_fall_steep), with two lines per panel: ``standardize=true`` (held at
    the input K by per-gen re-standardization) and ``standardize=false``
    (drifts as v_E shifts the upper-tail mass past the fixed threshold).

    Args:
        std_paths_per_trajectory: outer list = trajectory, inner list =
            per-rep ``phenotype_stats.yaml`` paths for the
            ``standardize=true`` variant.
        nostd_paths_per_trajectory: same shape, for ``standardize=false``.
        labels: display label per trajectory.
        output_path: image path to save.
        trait: 1 or 2.
        target_prevalence: input K to draw as dashed reference (typically
            0.1).  Omitted if ``None``.
    """
    n_traj = len(labels)
    if len(std_paths_per_trajectory) != n_traj or len(nostd_paths_per_trajectory) != n_traj:
        raise ValueError(
            "std_paths_per_trajectory, nostd_paths_per_trajectory, and labels "
            "must all have the same length"
        )

    apply_nature_style()
    fig, axes = plt.subplots(1, n_traj, figsize=(3.4 * n_traj, 4.5), sharey=True)
    if n_traj == 1:
        axes = np.array([axes])

    std_color = SCENARIO_PALETTE[0]
    nostd_color = SCENARIO_PALETTE[1]

    for ax, label, std_paths, nostd_paths in zip(
        axes, labels, std_paths_per_trajectory, nostd_paths_per_trajectory, strict=True
    ):
        for color, line_label, paths in (
            (std_color, "standardize=true", std_paths),
            (nostd_color, "standardize=false", nostd_paths),
        ):
            per_gen = _load_per_gen_prevalence([Path(p) for p in paths], trait=trait)
            gens = sorted(per_gen.keys())
            means = [float(np.mean(per_gen[g])) if per_gen[g] else float("nan") for g in gens]
            lows = [float(np.min(per_gen[g])) if per_gen[g] else float("nan") for g in gens]
            highs = [float(np.max(per_gen[g])) if per_gen[g] else float("nan") for g in gens]
            ax.plot(gens, means, color=color, marker="o", label=line_label)
            ax.fill_between(gens, lows, highs, color=color, alpha=0.15, linewidth=0)

        if target_prevalence is not None:
            ax.axhline(
                y=target_prevalence,
                linestyle="--",
                color="#888888",
                linewidth=1,
                alpha=0.8,
            )

        ax.set_title(label)
        ax.set_xlabel("Generation")
        enable_value_gridlines(ax)
        ax.legend(loc="best", fontsize=8, frameon=False)

    axes[0].set_ylabel(f"Observed prevalence (trait {trait})")
    fig.suptitle("Per-generation prevalence: standardize=true vs false")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


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


def _load_tetrachoric(phenotype_stats_path: Path, trait: int) -> dict[str, float]:
    """Return a flat ``{MZ, FS, MO, FO, MHS, PHS, 1C}`` tetrachoric r dict."""
    with open(phenotype_stats_path) as fh:
        ps = yaml.safe_load(fh) or {}
    tet = (ps.get("tetrachoric") or {}).get(f"trait{trait}", {}) or {}
    out: dict[str, float] = {}
    for key, entry in tet.items():
        r = (entry or {}).get("r")
        out[key] = float(r) if r is not None else float("nan")
    return out


def load_observed_vs_liability_h2(
    pedigree_paths: list[Path],
    phenotype_stats_paths: list[Path],
    trait: int = 1,
    min_generation: int | None = None,
) -> dict[str, np.ndarray]:
    """Per-rep liability-Falconer + tetrachoric-Falconer h² + realized h².

    The two input lists must be in the same replicate order.  Liability
    correlations and realized h² come from :func:`load_pedigree_estimates`
    (pedigree.parquet); tetrachoric correlations come from
    ``phenotype_stats.yaml.tetrachoric.trait{trait}``.

    Args:
        pedigree_paths: one ``pedigree.parquet`` path per rep.
        phenotype_stats_paths: one ``phenotype_stats.yaml`` path per rep.
        trait: 1 or 2.
        min_generation: forwarded to :func:`load_pedigree_estimates` for the
            liability correlations and realized h².  Tetrachoric values in
            ``phenotype_stats.yaml`` are pre-aggregated over phenotyped
            generations and not re-filtered here.

    Returns:
        Dict keyed ``{'liability_falconer', 'tetrachoric_falconer',
        'realized'}``; each value is a per-rep ``np.ndarray``.
    """
    out: dict[str, list[float]] = {k: [] for k in ("liability_falconer", "tetrachoric_falconer", "realized")}
    for ped_path, ps_path in zip(pedigree_paths, phenotype_stats_paths, strict=True):
        est = load_pedigree_estimates(ped_path, trait=trait, min_generation=min_generation)
        r_mz_liab = est.get("MZ", float("nan"))
        r_fs_liab = est.get("FS", float("nan"))
        tet = _load_tetrachoric(ps_path, trait=trait)
        r_mz_tet = tet.get("MZ", float("nan"))
        r_fs_tet = tet.get("FS", float("nan"))

        out["liability_falconer"].append(
            2.0 * (r_mz_liab - r_fs_liab) if np.isfinite(r_mz_liab) and np.isfinite(r_fs_liab) else float("nan")
        )
        out["tetrachoric_falconer"].append(
            2.0 * (r_mz_tet - r_fs_tet) if np.isfinite(r_mz_tet) and np.isfinite(r_fs_tet) else float("nan")
        )
        out["realized"].append(est.get("realized_h2", float("nan")))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def compare_observed_vs_liability_h2(
    pedigree_paths_per_scenario: list[list[Path]],
    phenotype_stats_paths_per_scenario: list[list[Path]],
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
            per-rep ``pedigree.parquet`` paths.
        phenotype_stats_paths_per_scenario: same shape, per-rep
            ``phenotype_stats.yaml`` paths.  Rep order must match.
        labels: display label per scenario.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: forwarded to :func:`load_pedigree_estimates`.
        input_h2: simulation input h²; drawn as a dashed reference line.
    """
    n_scen = len(labels)
    if len(pedigree_paths_per_scenario) != n_scen or len(phenotype_stats_paths_per_scenario) != n_scen:
        raise ValueError("pedigree_paths, phenotype_stats_paths, and labels must have the same length")

    apply_nature_style()
    estimator_labels = [d[0] for d in OBSERVED_LIABILITY_ESTIMATOR_DEFS]
    estimator_keys = [d[1] for d in OBSERVED_LIABILITY_ESTIMATOR_DEFS]
    n_est = len(estimator_keys)

    per_scen = [
        load_observed_vs_liability_h2(
            [Path(p) for p in ped_paths],
            [Path(p) for p in ps_paths],
            trait=trait,
            min_generation=min_generation,
        )
        for ped_paths, ps_paths in zip(pedigree_paths_per_scenario, phenotype_stats_paths_per_scenario, strict=True)
    ]

    fig, (ax_raw, ax_bias) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)
    x = np.arange(n_scen, dtype=float)  # groups are scenarios (x-axis)
    total_group_width = 0.7
    bar_width = total_group_width / n_est

    def _stats(arr: np.ndarray) -> tuple[float, float, float]:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return float("nan"), float("nan"), float("nan")
        return float(valid.mean()), float(valid.min()), float(valid.max())

    # Dedicated palette for the two estimators so scenarios are the x-axis.
    est_colors = ("#4477AA", "#EE6677")

    for est_idx, (est_label, est_key) in enumerate(zip(estimator_labels, estimator_keys, strict=True)):
        raw_means = np.full(n_scen, np.nan)
        raw_lows = np.full(n_scen, np.nan)
        raw_highs = np.full(n_scen, np.nan)
        bias_means = np.full(n_scen, np.nan)
        bias_lows = np.full(n_scen, np.nan)
        bias_highs = np.full(n_scen, np.nan)
        for scen_idx, ests in enumerate(per_scen):
            vals = ests[est_key]
            realized = ests["realized"]
            raw_means[scen_idx], raw_lows[scen_idx], raw_highs[scen_idx] = _stats(vals)
            if vals.size == realized.size:
                bias_means[scen_idx], bias_lows[scen_idx], bias_highs[scen_idx] = _stats(vals - realized)

        offsets = x - total_group_width / 2 + bar_width / 2 + est_idx * bar_width
        color = est_colors[est_idx]
        for ax, means, lows, highs in (
            (ax_raw, raw_means, raw_lows, raw_highs),
            (ax_bias, bias_means, bias_lows, bias_highs),
        ):
            mask = np.isfinite(means)
            err_low = np.where(mask, means - lows, 0.0)
            err_high = np.where(mask, highs - means, 0.0)
            ax.bar(
                offsets[mask],
                means[mask],
                width=bar_width * 0.95,
                color=color,
                label=est_label if ax is ax_bias else None,
                yerr=[err_low[mask], err_high[mask]],
                capsize=3,
                linewidth=0,
            )

    # Per-scenario realized-h² solid tick in the top panel.  Skip when the
    # scenario's realized h² is effectively equal to the input h² (common in
    # no-AM setups) so we don't double-draw the reference and visually
    # clutter the bars.
    for scen_idx, ests in enumerate(per_scen):
        realized = ests["realized"]
        valid = realized[np.isfinite(realized)]
        if valid.size == 0:
            continue
        mean_r = float(valid.mean())
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

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


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
    """Library entry point used by Snakemake script wrappers."""
    compare_realized_variance_trajectory(
        scenario_paths=scenario_paths,
        labels=labels,
        output_path=output_path,
        trait=trait,
        expected_A=expected_A,
        expected_C=expected_C,
        expected_E=expected_E,
    )


def cli() -> None:
    """Standalone CLI for ad-hoc rendering outside Snakemake."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        metavar="LABEL=PATH1,PATH2,...",
        help="Repeat per scenario. LABEL is the legend label, PATHS is a "
        "comma-separated list of validation.yaml files (one per replicate).",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trait", type=int, default=1, choices=[1, 2])
    parser.add_argument("--expected-A", type=float, default=None)
    parser.add_argument("--expected-C", type=float, default=None)
    parser.add_argument("--expected-E", type=float, default=None)
    args = parser.parse_args()

    labels: list[str] = []
    scenario_paths: list[list[Path]] = []
    for spec in args.scenario:
        label, _, paths_spec = spec.partition("=")
        if not paths_spec:
            parser.error(f"--scenario '{spec}' must be LABEL=PATH1,PATH2,...")
        labels.append(label)
        scenario_paths.append([Path(p) for p in paths_spec.split(",")])

    main(
        scenario_paths=scenario_paths,
        labels=labels,
        output_path=args.output,
        trait=args.trait,
        expected_A=args.expected_A,
        expected_C=args.expected_C,
        expected_E=args.expected_E,
    )


if __name__ == "__main__":
    cli()
