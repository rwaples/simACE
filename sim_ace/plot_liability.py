"""Liability-related phenotype plots.

Contains: plot_liability_joint, plot_liability_joint_affected,
plot_liability_violin, plot_liability_violin_by_generation, plot_joint_affection,
plot_censoring_confusion, plot_censoring_cascade, plot_mate_correlation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sim_ace.utils import HEATMAP_CMAP, annotate_heatmap, draw_split_violin, finalize_plot, save_placeholder_plot

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

    fig = plt.figure(figsize=(13, 13))
    fig.suptitle(f"Cross-Trait Correlations [{scenario}]", fontsize=14, y=1.01)
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
                (~affected, "C0", 0.2, "Unaffected"),
                (affected, "C3", 0.5, aff_label),
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
            ax_marg_x.hist(x[~affected], bins=bins_x.tolist(), edgecolor="none", alpha=0.5, color="C0")
            ax_marg_x.hist(x[affected], bins=bins_x.tolist(), edgecolor="none", alpha=0.7, color="C3")
            ax_marg_y.hist(
                y[~affected],
                bins=bins_y.tolist(),
                orientation="horizontal",
                edgecolor="none",
                alpha=0.5,
                color="C0",
            )
            ax_marg_y.hist(
                y[affected],
                bins=bins_y.tolist(),
                orientation="horizontal",
                edgecolor="none",
                alpha=0.7,
                color="C3",
            )
        else:
            ax_joint.plot(x, y, "o", ms=2, mew=0, alpha=0.3, rasterized=True)
            ax_marg_x.hist(x, bins=50, edgecolor="none", alpha=0.7)
            ax_marg_y.hist(y, bins=50, orientation="horizontal", edgecolor="none", alpha=0.7)

        r = np.corrcoef(x, y)[0, 1]
        ax_joint.text(
            0.05,
            0.95,
            f"r = {r:.4f}",
            transform=ax_joint.transAxes,
            va="top",
            fontsize=11,
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
            Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=8, label="Unaffected"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=8, label=aff_label),
        ]
        fig.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    finalize_plot(output_path, subsample_note=subsample_note)


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
    ax.set_xticklabels(["Trait 1", "Trait 2"])
    ax.set_ylabel("Liability")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="C0", edgecolor="black", linewidth=0.8, label="0"),
            Patch(facecolor="C1", edgecolor="black", linewidth=0.8, label="1"),
        ],
        title="Affected",
    )
    ax.set_title(f"Liability by Affected Status [{scenario}]")

    # Annotate mean liability for each trait x affected/unaffected group
    for i, trait_num in enumerate([1, 2]):
        liab = df_samples[f"liability{trait_num}"].values
        aff = df_samples[f"affected{trait_num}"].values.astype(bool)
        if aff.any():
            mean_aff = liab[aff].mean()
            ax.plot(i + 0.05, mean_aff, "D", color="black", markersize=6, zorder=5)
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
            ax.plot(i - 0.05, mean_unaff, "D", color="black", markersize=6, zorder=5)
            ax.text(
                i - 0.12,
                mean_unaff,
                f"\u03bc={mean_unaff:.2f}",
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    ax.text(
        0,
        ax.get_ylim()[0],
        f"Prevalence: {prev1:.1%}",
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
    )
    ax.text(
        1,
        ax.get_ylim()[0],
        f"Prevalence: {prev2:.1%}",
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
    )
    finalize_plot(output_path, subsample_note=subsample_note)


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

    fig, axes = plt.subplots(2, n_gens, figsize=(4 * n_gens, 8), squeeze=False)

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
                ax.set_xticks([0])
                ax.set_xticklabels([f"Trait {trait_num}"])
                if row == 0 and col == n_gens - 1:
                    from matplotlib.patches import Patch

                    ax.legend(
                        handles=[
                            Patch(facecolor="C0", edgecolor="black", linewidth=0.8, label="0"),
                            Patch(facecolor="C1", edgecolor="black", linewidth=0.8, label="1"),
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

                # Prevalence annotation
                obs_prev = aff.mean()
                ax.set_xlabel(f"prev: {obs_prev:.1%}", fontsize=8)

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

    fig.suptitle(
        f"Liability by Affected Status per Generation [{scenario}]",
        fontsize=14,
    )
    finalize_plot(output_path, subsample_note=subsample_note)


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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Censoring Confusion Matrix [{scenario}]", fontsize=14)

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

        metrics = f"Sensitivity: {sensitivity:.3f}   Specificity: {specificity:.3f}"
        ax.set_title(f"Trait {trait}\n{metrics}", fontsize=11)

    finalize_plot(output_path)


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

    # Colors — maximise contrast between censoring categories
    color_observed = "#4CAF50"  # green  — true positives
    color_death = "#E57373"  # red    — death-censored
    color_right = "#7E57C2"  # purple — right-censored
    color_left = "#FF9800"  # orange — left-truncated

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Censoring Cascade [{scenario}]", fontsize=14)

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

    # Shared legend above the subplots, below the suptitle
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.95), framealpha=0.9)

    finalize_plot(output_path, tight_rect=[0, 0, 1, 0.93])


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
    title = f"Joint Affected Status [{scenario}]"
    if r_label:
        title += f"\n{r_label}"
    ax.set_title(title, fontsize=14)
    finalize_plot(output_path)


def plot_mate_correlation(
    all_stats: list[dict],
    output_path: str | Path,
    scenario: str = "",
    params: dict | None = None,
) -> None:
    """Plot 2x2 heatmap of empirical mate liability correlations with expected values."""
    from sim_ace.utils import expected_mate_corr_matrix

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
            C1=float(params.get("C1", 0)),
            A2=float(params.get("A2", 0)),
            C2=float(params.get("C2", 0)),
            assort_matrix=am,
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

    _fig.suptitle(
        f"Mate Liability Correlation [{scenario}]",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    finalize_plot(output_path)
