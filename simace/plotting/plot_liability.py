"""Liability-related phenotype plots.

Contains: plot_liability_joint, plot_liability_joint_affected,
plot_liability_violin, plot_liability_violin_by_generation, plot_joint_affection,
plot_censoring_confusion, plot_censoring_cascade, plot_mate_correlation.
"""

from __future__ import annotations

__all__ = [
    "plot_censoring_cascade",
    "plot_censoring_confusion",
    "plot_joint_affection",
    "plot_liability_components_by_generation",
    "plot_liability_joint",
    "plot_liability_joint_affected",
    "plot_liability_joint_affected_t2",
    "plot_liability_violin",
    "plot_liability_violin_by_generation",
    "plot_liability_violin_by_sex_generation",
    "plot_mate_correlation",
]

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

from simace.plotting.plot_style import COLOR_AFFECTED, COLOR_FEMALE, COLOR_MALE, COLOR_UNAFFECTED  # noqa: F401
from simace.plotting.plot_utils import (
    HEATMAP_CMAP,
    annotate_heatmap,
    draw_split_violin,
    finalize_plot,
    param_as_float,
    save_placeholder_plot,
)

logger = logging.getLogger(__name__)


def _plot_joint_grid(
    df_samples: pd.DataFrame,
    output_path: str | Path,
    scenario: str = "",
    color_by_affected: bool = False,
    affected_trait: int = 1,
    subsample_note: str = "",
) -> None:
    """Internal: 2x2 grid of jointplots for cross-trait correlations."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    panels = [
        ("liability1", "liability2", "Liability"),
        ("A1", "A2", "A (Additive genetic)"),
        ("C1", "C2", "C (Common environment)"),
        ("E1", "E2", "E (Unique environment)"),
    ]
    panels = [(x, y, t) for x, y, t in panels if x in df_samples.columns and y in df_samples.columns]

    fig = plt.figure(figsize=(13, 12))
    outer = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    if color_by_affected:
        aff_col = f"affected{affected_trait}"
        aff_label = f"Affected (T{affected_trait})"
        affected = df_samples[aff_col].values.astype(bool)

    for idx, (xcol, ycol, title) in enumerate(panels):
        inner = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer[idx],
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)

        x, y = df_samples[xcol].values, df_samples[ycol].values

        if color_by_affected:
            bins_x = np.linspace(x.min(), x.max(), 51)
            bins_y = np.linspace(y.min(), y.max(), 51)
            for mask, color, alpha, label in [
                (~affected, COLOR_UNAFFECTED, 0.2, "Unaffected"),
                (affected, COLOR_AFFECTED, 0.5, aff_label),
            ]:
                ax_joint.plot(
                    x[mask],
                    y[mask],
                    "o",
                    ms=2,
                    mew=0,
                    color=color,
                    alpha=alpha,
                    rasterized=True,
                    label=label,
                )
            ax_marg_x.hist(x[~affected], bins=bins_x.tolist(), edgecolor="none", alpha=0.5, color=COLOR_UNAFFECTED)
            ax_marg_x.hist(x[affected], bins=bins_x.tolist(), edgecolor="none", alpha=0.7, color=COLOR_AFFECTED)
            ax_marg_y.hist(
                y[~affected],
                bins=bins_y.tolist(),
                orientation="horizontal",
                edgecolor="none",
                alpha=0.5,
                color=COLOR_UNAFFECTED,
            )
            ax_marg_y.hist(
                y[affected],
                bins=bins_y.tolist(),
                orientation="horizontal",
                edgecolor="none",
                alpha=0.7,
                color=COLOR_AFFECTED,
            )
        else:
            ax_joint.plot(x, y, "o", ms=2, mew=0, alpha=0.3, rasterized=True)
            ax_marg_x.hist(x, bins=50, edgecolor="none", alpha=0.7)
            ax_marg_y.hist(y, bins=50, orientation="horizontal", edgecolor="none", alpha=0.7)

        from simace.core.numerics import fast_pearsonr

        r, _p = fast_pearsonr(x, y)
        ann = f"r = {r:.4f}"
        ax_joint.text(
            0.05,
            0.95,
            ann,
            transform=ax_joint.transAxes,
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax_joint.set_box_aspect(1)
        ax_joint.set_xlabel(f"{title} (Trait 1)")
        ax_joint.set_ylabel(f"{title} (Trait 2)")

        ax_marg_x.set_title(f"{title}: Trait 1 vs Trait 2", fontsize=11)
        ax_marg_x.tick_params(labelbottom=False, labelleft=False)
        ax_marg_x.set_ylabel("")
        ax_marg_y.tick_params(labelleft=False, labelbottom=False)
        ax_marg_y.set_xlabel("")

        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

    if color_by_affected:
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_UNAFFECTED, markersize=6, label="Unaffected"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_AFFECTED, markersize=6, label=aff_label),
        ]
        fig.legend(handles=legend_handles, loc="lower left", fontsize=10, bbox_to_anchor=(0.05, 0.02))

    finalize_plot(output_path, subsample_note=subsample_note, scenario=scenario)


def plot_liability_joint(
    df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "", subsample_note: str = ""
) -> None:
    """2x2 grid of jointplots: Liability, A, C, E (trait 1 vs trait 2)."""
    _plot_joint_grid(df_samples, output_path, scenario, color_by_affected=False, subsample_note=subsample_note)


def plot_liability_joint_affected(
    df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "", subsample_note: str = ""
) -> None:
    """2x2 grid of jointplots colored by affected status (trait 1)."""
    _plot_joint_grid(
        df_samples, output_path, scenario, color_by_affected=True, affected_trait=1, subsample_note=subsample_note
    )


def plot_liability_joint_affected_t2(
    df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "", subsample_note: str = ""
) -> None:
    """2x2 grid of jointplots colored by affected status (trait 2)."""
    _plot_joint_grid(
        df_samples, output_path, scenario, color_by_affected=True, affected_trait=2, subsample_note=subsample_note
    )


def plot_liability_violin(
    df_samples: pd.DataFrame,
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    subsample_note: str = "",
) -> None:
    """Split violin plot of liability by trait, split on affected status."""
    # Use pre-computed prevalence averaged across reps
    prev1 = np.mean([s["prevalence"]["trait1"] for s in all_stats])
    prev2 = np.mean([s["prevalence"]["trait2"] for s in all_stats])

    _fig, ax = plt.subplots(figsize=(8, 6))
    for i, trait_num in enumerate([1, 2]):
        liab = df_samples[f"liability{trait_num}"].values
        aff = df_samples[f"affected{trait_num}"].values.astype(bool)
        draw_split_violin(ax, liab[~aff], liab[aff], pos=i)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Trait 1\n{prev1:.1%}", f"Trait 2\n{prev2:.1%}"])
    ax.set_ylabel("Liability")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=COLOR_UNAFFECTED, edgecolor="black", linewidth=0.8, label="0"),
            Patch(facecolor=COLOR_AFFECTED, edgecolor="black", linewidth=0.8, label="1"),
        ],
        title="Affected",
    )
    ax.set_title("Liability by Affected Status")

    # Annotate mean liability for each trait x affected/unaffected group
    for i, trait_num in enumerate([1, 2]):
        liab = df_samples[f"liability{trait_num}"].values
        aff = df_samples[f"affected{trait_num}"].values.astype(bool)
        if aff.any():
            mean_aff = liab[aff].mean()
            ax.plot(i + 0.05, mean_aff, "D", color="black", markersize=5, zorder=5)
            ax.text(
                i + 0.12,
                mean_aff,
                f"\u03bc={mean_aff:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        if (~aff).any():
            mean_unaff = liab[~aff].mean()
            ax.plot(i - 0.05, mean_unaff, "D", color="black", markersize=5, zorder=5)
            ax.text(
                i - 0.12,
                mean_unaff,
                f"\u03bc={mean_unaff:.2f}",
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
    finalize_plot(output_path, subsample_note=subsample_note, scenario=scenario)


def plot_liability_violin_by_generation(
    df_samples: pd.DataFrame,
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    subsample_note: str = "",
) -> None:
    """Split violin of liability by affected status, one column per generation."""
    if "generation" not in df_samples.columns:
        save_placeholder_plot(output_path, "No generation data")
        return

    gens = sorted(df_samples["generation"].unique())
    n_gens = len(gens)

    _fig, axes = plt.subplots(2, n_gens, figsize=(4 * n_gens, 8), squeeze=False)

    for row, trait_num in enumerate([1, 2]):
        liab_col = f"liability{trait_num}"
        aff_col = f"affected{trait_num}"

        for col, gen in enumerate(gens):
            ax = axes[row, col]
            gen_mask = df_samples["generation"] == gen
            df_gen = df_samples.loc[gen_mask]

            liab = df_gen[liab_col].values
            aff = df_gen[aff_col].values.astype(bool)

            if len(liab) > 1:
                draw_split_violin(ax, liab[~aff], liab[aff], pos=0)
                obs_prev = aff.mean()
                ax.set_xticks([0])
                ax.set_xticklabels([f"{obs_prev:.1%}"])
                if row == 0 and col == n_gens - 1:
                    from matplotlib.patches import Patch

                    ax.legend(
                        handles=[
                            Patch(facecolor=COLOR_UNAFFECTED, edgecolor="black", linewidth=0.8, label="0"),
                            Patch(facecolor=COLOR_AFFECTED, edgecolor="black", linewidth=0.8, label="1"),
                        ],
                        title="Affected",
                        fontsize=8,
                    )

                # Annotate means
                if aff.any():
                    mu = liab[aff].mean()
                    ax.plot(0.05, mu, "D", color="black", markersize=5, zorder=5)
                    ax.text(0.12, mu, f"\u03bc={mu:.2f}", ha="left", va="center", fontsize=8, fontweight="bold")
                if (~aff).any():
                    mu = liab[~aff].mean()
                    ax.plot(-0.05, mu, "D", color="black", markersize=5, zorder=5)
                    ax.text(-0.12, mu, f"\u03bc={mu:.2f}", ha="right", va="center", fontsize=8, fontweight="bold")

            if row == 0:
                label = f"Gen {gen}"
                if col == 0:
                    label += " (oldest)"
                elif col == n_gens - 1:
                    label += " (youngest)"
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nLiability", fontsize=10)
            else:
                ax.set_ylabel("")

    finalize_plot(output_path, subsample_note=subsample_note, scenario=scenario)


def plot_liability_violin_by_sex_generation(
    df_samples: pd.DataFrame,
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    subsample_note: str = "",
) -> None:
    """Split violin by affected status with side-by-side F|M panels per generation.

    Layout: 2 rows (traits) x N cols (generations). Each cell has two
    side-by-side sub-violins at x=-0.3 (female) and x=+0.3 (male).
    """
    if "generation" not in df_samples.columns:
        save_placeholder_plot(output_path, "No generation data")
        return

    gens = sorted(df_samples["generation"].unique())
    n_gens = len(gens)
    sex_arr = df_samples["sex"].values

    _fig, axes = plt.subplots(2, n_gens, figsize=(5 * n_gens, 8), squeeze=False)

    for row, trait_num in enumerate([1, 2]):
        liab_col = f"liability{trait_num}"
        aff_col = f"affected{trait_num}"

        for col, gen in enumerate(gens):
            ax = axes[row, col]
            gen_mask = df_samples["generation"] == gen
            sex_prev: dict[str, str] = {}

            for sex_val, sex_label, pos in [
                (0, "F", -0.3),
                (1, "M", 0.3),
            ]:
                mask = gen_mask & (sex_arr == sex_val)
                df_sub = df_samples.loc[mask]
                liab = df_sub[liab_col].values
                aff = df_sub[aff_col].values.astype(bool)

                if len(liab) > 1:
                    draw_split_violin(ax, liab[~aff], liab[aff], pos=pos, width=0.5)

                    # Annotate means
                    if aff.any():
                        mu = liab[aff].mean()
                        ax.plot(pos + 0.03, mu, "D", color="black", markersize=4, zorder=5)
                    if (~aff).any():
                        mu = liab[~aff].mean()
                        ax.plot(pos - 0.03, mu, "D", color="black", markersize=4, zorder=5)

                    sex_prev[sex_label] = f"{aff.mean():.0%}"

            ax.set_xticks([-0.3, 0.3])
            ax.set_xticklabels(
                [f"F\n{sex_prev.get('F', '')}", f"M\n{sex_prev.get('M', '')}"],
                fontsize=8,
            )

            if row == 0:
                label = f"Gen {gen}"
                if col == 0:
                    label += " (oldest)"
                elif col == n_gens - 1:
                    label += " (youngest)"
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nLiability", fontsize=10)
            else:
                ax.set_ylabel("")

            # Legend only once
            if row == 0 and col == n_gens - 1:
                from matplotlib.patches import Patch

                ax.legend(
                    handles=[
                        Patch(facecolor=COLOR_UNAFFECTED, edgecolor="black", linewidth=0.8, label="Unaffected"),
                        Patch(facecolor=COLOR_AFFECTED, edgecolor="black", linewidth=0.8, label="Affected"),
                    ],
                    fontsize=8,
                )

    finalize_plot(output_path, subsample_note=subsample_note, scenario=scenario)


def plot_liability_components_by_generation(
    df_samples: pd.DataFrame,
    output_path: str | Path,
    scenario: str = "",
    subsample_note: str = "",
) -> None:
    """Mean variance component by affected status across generations.

    2x3 grid: rows = traits, columns = A, C, E.  Each panel shows mean
    component value for affected (red), unaffected (grey), and overall (black)
    individuals per generation.  Prevalence annotated on x-tick labels.
    """
    gens = sorted(df_samples["generation"].unique())
    n_gen = len(gens)
    if n_gen == 0:
        save_placeholder_plot(output_path, "No generation data")
        return

    components = ["A", "C", "E"]
    _fig, axes = plt.subplots(2, 3, figsize=(max(14, n_gen * 3), 8))

    for row, trait_num in enumerate([1, 2]):
        aff_col = f"affected{trait_num}"
        if aff_col not in df_samples.columns:
            continue

        # Compute prevalence per generation (shared across columns)
        prev = []
        for gen in gens:
            g = df_samples[df_samples["generation"] == gen]
            prev.append(g[aff_col].mean() if len(g) > 0 else 0)

        for col_idx, comp in enumerate(components):
            comp_col = f"{comp}{trait_num}"
            if comp_col not in df_samples.columns:
                continue

            ax = axes[row, col_idx]
            mean_aff, mean_unaff, mean_all = [], [], []
            for gen in gens:
                g = df_samples[df_samples["generation"] == gen]
                aff = g[g[aff_col]]
                unaff = g[~g[aff_col]]
                mean_aff.append(aff[comp_col].mean() if len(aff) > 0 else float("nan"))
                mean_unaff.append(unaff[comp_col].mean() if len(unaff) > 0 else float("nan"))
                mean_all.append(g[comp_col].mean() if len(g) > 0 else float("nan"))

            ax.plot(gens, mean_aff, "o-", color=COLOR_AFFECTED, label="Affected", markersize=5)
            ax.plot(gens, mean_unaff, "s-", color=COLOR_UNAFFECTED, label="Unaffected", markersize=5)
            ax.plot(gens, mean_all, "D-", color="black", label="Overall", markersize=4, linewidth=1.0)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

            ax.set_xticks(gens)
            if col_idx == 0:
                # Show generation + prevalence in the A column for both traits
                ax.set_xticklabels([f"{g}\n{prev[i]:.1%}" for i, g in enumerate(gens)])
            else:
                ax.set_xticklabels([str(g) for g in gens])
            if row == 1:
                ax.set_xlabel("Generation")

            if col_idx == 0:
                ax.set_ylabel(f"Trait {trait_num}\nMean value")
            if row == 0:
                ax.set_title(comp, fontweight="bold")
                if col_idx == 2:
                    ax.legend(fontsize=8)

    finalize_plot(output_path, subsample_note=subsample_note, scenario=scenario)


def plot_censoring_confusion(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Per-trait 2x2 confusion matrix: true affected vs. observed affected.

    Uses pre-computed censoring_confusion stats from full (non-subsampled) data.
    """
    stats_with_data = [s for s in all_stats if s.get("censoring_confusion")]
    if not stats_with_data:
        save_placeholder_plot(output_path, "No censoring confusion data")
        return

    _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col, trait in enumerate([1, 2]):
        ax = axes[col]
        key = f"trait{trait}"

        # Average counts across reps
        rep_data = [s["censoring_confusion"][key] for s in stats_with_data if key in s["censoring_confusion"]]
        if not rep_data:
            ax.text(0.5, 0.5, f"No data for trait {trait}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Trait {trait}")
            continue

        tp = np.mean([d["tp"] for d in rep_data])
        fn = np.mean([d["fn"] for d in rep_data])
        fp = np.mean([d["fp"] for d in rep_data])
        tn = np.mean([d["tn"] for d in rep_data])
        n = np.mean([d["n"] for d in rep_data])

        props = np.array([[tp / n, fn / n], [fp / n, tn / n]])
        counts = np.array([[tp, fn], [fp, tn]])
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

        is_last = col == len(axes) - 1
        sns.heatmap(
            props,
            annot=False,
            cmap=HEATMAP_CMAP,
            ax=ax,
            xticklabels=["Observed Yes", "Observed No"],
            yticklabels=["True Yes", "True No"],
            vmin=0,
            vmax=1,
            cbar=is_last,
            cbar_kws={"label": "Proportion"} if is_last else {},
        )
        annotate_heatmap(ax, props, counts)

        metrics = (
            f"Sens: {sensitivity:.3f}   Spec: {specificity:.3f}   PPV: {ppv:.3f}   NPV: {npv:.3f}   n = {int(n):,}"
        )
        ax.set_title(f"Trait {trait}\n{metrics}", fontsize=11)

    finalize_plot(output_path, scenario=scenario)


def plot_censoring_cascade(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Per-trait stacked bar chart decomposing true cases by censoring fate per generation.

    Uses pre-computed censoring_cascade stats from full (non-subsampled) data.
    """
    stats_with_data = [s for s in all_stats if s.get("censoring_cascade")]
    if not stats_with_data:
        save_placeholder_plot(output_path, "No censoring cascade data")
        return

    from simace.plotting.plot_style import CENSORING_COLORS

    color_observed = CENSORING_COLORS["observed"]
    color_death = CENSORING_COLORS["death"]
    color_right = CENSORING_COLORS["right"]
    color_left = CENSORING_COLORS["left"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col, trait in enumerate([1, 2]):
        ax = axes[col]
        key = f"trait{trait}"

        rep_data = [s["censoring_cascade"][key] for s in stats_with_data if key in s["censoring_cascade"]]
        if not rep_data:
            ax.text(0.5, 0.5, f"No data for trait {trait}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Trait {trait}")
            continue

        # Discover generation keys from first rep
        gen_keys = sorted(rep_data[0].keys())
        if not gen_keys:
            ax.text(0.5, 0.5, "No generations", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Trait {trait}")
            continue

        counts_observed = []
        counts_death = []
        counts_right = []
        counts_left = []
        sensitivities = []
        x_labels = []

        for gk in gen_keys:
            gen_data = [r[gk] for r in rep_data if gk in r]
            if not gen_data:
                continue

            n_obs = np.mean([d["observed"] for d in gen_data])
            n_death = np.mean([d["death_censored"] for d in gen_data])
            n_right = np.mean([d["right_censored"] for d in gen_data])
            n_left = np.mean([d["left_truncated"] for d in gen_data])
            n_true = np.mean([d["true_affected"] for d in gen_data])
            window = gen_data[0]["window"]

            counts_observed.append(n_obs)
            counts_death.append(n_death)
            counts_right.append(n_right)
            counts_left.append(n_left)
            sensitivities.append(n_obs / n_true if n_true > 0 else float("nan"))
            gen_num = gk.replace("gen", "")
            x_labels.append(f"Gen {gen_num}\n[{window[0]:.0f}, {window[1]:.0f}]")

        n_bars = len(x_labels)
        x = np.arange(n_bars)
        bar_width = 0.6

        bottom = np.zeros(n_bars)
        bars_obs = np.array(counts_observed, dtype=float)
        bars_death = np.array(counts_death, dtype=float)
        bars_right = np.array(counts_right, dtype=float)
        bars_left = np.array(counts_left, dtype=float)

        ax.bar(x, bars_obs, bar_width, bottom=bottom, color=color_observed, label="Observed (TP)")
        bottom += bars_obs
        ax.bar(x, bars_death, bar_width, bottom=bottom, color=color_death, label="Death-censored")
        bottom += bars_death
        ax.bar(x, bars_right, bar_width, bottom=bottom, color=color_right, label="Right-censored")
        bottom += bars_right
        ax.bar(x, bars_left, bar_width, bottom=bottom, color=color_left, label="Left-truncated")
        bottom += bars_left

        # Annotate segments (skip if < 3% of bar height)
        for i in range(n_bars):
            total = bottom[i]
            if total == 0:
                continue
            cum = 0.0
            for count in [bars_obs[i], bars_death[i], bars_right[i], bars_left[i]]:
                if count > 0 and count / total >= 0.03:
                    mid = cum + count / 2
                    ax.text(x[i], mid, f"{int(count)}", ha="center", va="center", fontsize=8, fontweight="bold")
                cum += count

        # Fold sensitivity into x-axis tick labels (below bars, no overlap)
        for i, sens in enumerate(sensitivities):
            if not np.isnan(sens):
                x_labels[i] += f"\nsens={sens:.2f}"

        # Overall sensitivity
        total_obs = sum(counts_observed)
        total_true = total_obs + sum(counts_death) + sum(counts_right) + sum(counts_left)
        overall_sens = total_obs / total_true if total_true > 0 else float("nan")
        ax.set_title(f"Trait {trait}  (overall sensitivity: {overall_sens:.3f})", fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel("True affected count")

    # Shared legend above the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.98))

    finalize_plot(output_path, scenario=scenario)


def plot_joint_affection(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """2x2 heatmap of joint affection status (trait1 x trait2).

    Uses pre-computed joint_affection and cross_trait_tetrachoric stats.
    """
    # Average proportions/counts across reps
    keys = ["both", "trait1_only", "trait2_only", "neither"]
    avg_props = {}
    for k in keys:
        avg_props[k] = np.mean([s["joint_affection"]["proportions"][k] for s in all_stats])

    matrix = np.array(
        [
            [avg_props["both"], avg_props["trait1_only"]],
            [avg_props["trait2_only"], avg_props["neither"]],
        ]
    )

    avg_counts = {}
    for k in keys:
        avg_counts[k] = np.mean([s["joint_affection"]["counts"][k] for s in all_stats])

    count_matrix = np.array(
        [
            [avg_counts["both"], avg_counts["trait1_only"]],
            [avg_counts["trait2_only"], avg_counts["neither"]],
        ]
    )

    _fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=False,
        cmap=HEATMAP_CMAP,
        ax=ax,
        xticklabels=["Affected", "Unaffected"],
        yticklabels=["Affected", "Unaffected"],
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proportion"},
    )
    annotate_heatmap(ax, matrix, count_matrix)

    # Build subtitle from whichever correlation stats are present
    label_parts = []

    # Cross-trait tetrachoric correlation from pre-computed stats
    r_tet_vals = [s.get("cross_trait_tetrachoric", {}).get("same_person", {}).get("r") for s in all_stats]
    r_tet_vals = [v for v in r_tet_vals if v is not None]
    if r_tet_vals:
        label_parts.append(f"r_tet = {np.mean(r_tet_vals):.3f}")

    # Cross-trait frailty correlations (averaged across reps)
    uncens_vals = [
        s.get("frailty_cross_trait_uncensored", {}).get("r")
        for s in all_stats
        if s.get("frailty_cross_trait_uncensored", {}).get("r") is not None
    ]
    strat_vals = [
        s.get("frailty_cross_trait_stratified", {}).get("r")
        for s in all_stats
        if s.get("frailty_cross_trait_stratified", {}).get("r") is not None
    ]
    naive_vals = [
        s.get("frailty_cross_trait", {}).get("r")
        for s in all_stats
        if s.get("frailty_cross_trait", {}).get("r") is not None
    ]

    if uncens_vals:
        label_parts.append(f"r_frailty = {np.mean(uncens_vals):.3f}")
    if strat_vals:
        label_parts.append(f"stratified = {np.mean(strat_vals):.3f}")
    if naive_vals:
        label_parts.append(f"naive = {np.mean(naive_vals):.3f}")

    if not uncens_vals and not strat_vals and not naive_vals:
        label_parts.append("r_frailty: not computed")

    r_label = "  |  ".join(label_parts) if label_parts else ""

    ax.set_xlabel("Trait 1")
    ax.set_ylabel("Trait 2")
    title = "Joint Affected Status"
    if r_label:
        title += f"\n{r_label}"
    ax.set_title(title, fontsize=14)
    finalize_plot(output_path, scenario=scenario)


def plot_mate_correlation(
    all_stats: list[dict],
    output_path: str | Path,
    scenario: str = "",
    params: dict | None = None,
) -> None:
    """Plot 2x2 heatmap of empirical mate liability correlations with expected values."""
    from simace.simulation.mate_correlation import expected_mate_corr_matrix

    # Average observed matrices across replicates
    matrices = []
    for s in all_stats:
        mc = s.get("mate_correlation")
        if mc is not None:
            matrices.append(np.array(mc["matrix"]))
    if not matrices:
        save_placeholder_plot(output_path, "No mate correlation data")
        return

    obs = np.nanmean(np.stack(matrices), axis=0)

    # Compute expected matrix from params
    exp = np.zeros((2, 2))
    if params is not None:
        am = params.get("assort_matrix", None)
        exp = expected_mate_corr_matrix(
            assort1=float(params.get("assort1", 0)),
            assort2=float(params.get("assort2", 0)),
            rA=float(params.get("rA", 0)),
            rC=float(params.get("rC", 0)),
            A1=float(params.get("A1", 0)),
            C1=param_as_float(params.get("C1", 0)),
            A2=float(params.get("A2", 0)),
            C2=param_as_float(params.get("C2", 0)),
            assort_matrix=am,
            rE=float(params.get("rE", 0)),
            E1=param_as_float(params.get("E1", 0)),
            E2=param_as_float(params.get("E2", 0)),
        )

    xlabels = ["Male trait 1", "Male trait 2"]
    ylabels = ["Female trait 1", "Female trait 2"]

    _fig, (ax_exp, ax_obs) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: expected (parametric)
    sns.heatmap(
        exp,
        ax=ax_exp,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        square=True,
        cbar=False,
        xticklabels=xlabels,
        yticklabels=ylabels,
    )
    a1 = float(params.get("assort1", 0)) if params else 0
    a2 = float(params.get("assort2", 0)) if params else 0
    exp_title = "Expected"
    if a1 != 0 or a2 != 0:
        exp_title += f"\nassort1={a1}, assort2={a2}"
    ax_exp.set_title(exp_title, fontsize=13)

    # Right panel: observed (realized)
    sns.heatmap(
        obs,
        ax=ax_obs,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 16, "fontweight": "bold"},
        square=True,
        cbar_kws={"label": "Pearson r"},
        xticklabels=xlabels,
        yticklabels=[],
    )
    ax_obs.set_title("Observed", fontsize=13)

    finalize_plot(output_path, scenario=scenario)
