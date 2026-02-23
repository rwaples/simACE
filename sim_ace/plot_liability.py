"""Liability-related phenotype plots.

Contains: plot_liability_joint, plot_liability_joint_affected,
plot_liability_violin, plot_liability_violin_by_generation, plot_joint_affection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sim_ace.stats import tetrachoric_corr

import logging
logger = logging.getLogger(__name__)


def plot_liability_joint(df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "") -> None:
    """2x2 grid of jointplots: Liability, A, C, E (trait 1 vs trait 2)."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    panels = [
        ("liability1", "liability2", "Liability"),
        ("A1", "A2", "A (Additive genetic)"),
        ("C1", "C2", "C (Common environment)"),
        ("E1", "E2", "E (Unique environment)"),
    ]
    panels = [
        (x, y, t)
        for x, y, t in panels
        if x in df_samples.columns and y in df_samples.columns
    ]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Cross-Trait Correlations [{scenario}]", fontsize=14, y=1.01)
    outer = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    for idx, (xcol, ycol, title) in enumerate(panels):
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[idx],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)

        x, y = df_samples[xcol].values, df_samples[ycol].values
        ax_joint.scatter(x, y, alpha=0.05, s=3, rasterized=True)
        ax_marg_x.hist(x, bins=50, edgecolor="none", alpha=0.7)
        ax_marg_y.hist(
            y, bins=50, orientation="horizontal", edgecolor="none", alpha=0.7
        )

        r = np.corrcoef(x, y)[0, 1]
        ax_joint.text(
            0.05, 0.95, f"r = {r:.4f}",
            transform=ax_joint.transAxes, va="top", fontsize=11,
        )
        ax_joint.set_xlabel(f"{title} (Trait 1)")
        ax_joint.set_ylabel(f"{title} (Trait 2)")

        ax_marg_x.set_title(f"{title}: Trait 1 vs Trait 2", fontsize=11)
        ax_marg_x.tick_params(labelbottom=False, labelleft=False)
        ax_marg_x.set_ylabel("")
        ax_marg_y.tick_params(labelleft=False, labelbottom=False)
        ax_marg_y.set_xlabel("")

        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_joint_affected(df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "") -> None:
    """2x2 grid of jointplots colored by Weibull affected status."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    panels = [
        ("liability1", "liability2", "Liability"),
        ("A1", "A2", "A (Additive genetic)"),
        ("C1", "C2", "C (Common environment)"),
        ("E1", "E2", "E (Unique environment)"),
    ]
    panels = [
        (x, y, t)
        for x, y, t in panels
        if x in df_samples.columns and y in df_samples.columns
    ]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Cross-Trait Correlations (Weibull) [{scenario}]", fontsize=14, y=1.01)
    outer = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    affected = df_samples["affected1"].values.astype(bool)

    for idx, (xcol, ycol, title) in enumerate(panels):
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[idx],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)

        x, y = df_samples[xcol].values, df_samples[ycol].values
        bins_x = np.linspace(x.min(), x.max(), 51)
        bins_y = np.linspace(y.min(), y.max(), 51)

        for mask, color, alpha, label in [
            (~affected, "C0", 0.03, "Unaffected"),
            (affected, "C3", 0.15, "Affected (T1)"),
        ]:
            ax_joint.scatter(
                x[mask], y[mask], c=color, alpha=alpha, s=3, rasterized=True, label=label,
            )

        ax_marg_x.hist(x[~affected], bins=bins_x.tolist(), edgecolor="none", alpha=0.5, color="C0")
        ax_marg_x.hist(x[affected], bins=bins_x.tolist(), edgecolor="none", alpha=0.7, color="C3")
        ax_marg_y.hist(
            y[~affected], bins=bins_y.tolist(), orientation="horizontal", edgecolor="none", alpha=0.5, color="C0",
        )
        ax_marg_y.hist(
            y[affected], bins=bins_y.tolist(), orientation="horizontal", edgecolor="none", alpha=0.7, color="C3",
        )

        r = np.corrcoef(x, y)[0, 1]
        ax_joint.text(
            0.05, 0.95, f"r = {r:.4f}",
            transform=ax_joint.transAxes, va="top", fontsize=11,
        )
        ax_joint.set_xlabel(f"{title} (Trait 1)")
        ax_joint.set_ylabel(f"{title} (Trait 2)")

        ax_marg_x.set_title(f"{title}: Trait 1 vs Trait 2", fontsize=11)
        ax_marg_x.tick_params(labelbottom=False, labelleft=False)
        ax_marg_x.set_ylabel("")
        ax_marg_y.tick_params(labelleft=False, labelbottom=False)
        ax_marg_y.set_xlabel("")

        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=8, label="Unaffected"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=8, label="Affected (T1)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_violin(df_samples: pd.DataFrame, all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str = "") -> None:
    """Split violin plot of liability by trait, split on affected status."""
    violin_data = pd.concat(
        [
            pd.DataFrame({
                "Trait": "Trait 1",
                "Liability": df_samples["liability1"],
                "Affected": df_samples["affected1"],
            }),
            pd.DataFrame({
                "Trait": "Trait 2",
                "Liability": df_samples["liability2"],
                "Affected": df_samples["affected2"],
            }),
        ],
        ignore_index=True,
    )

    # Use pre-computed prevalence averaged across reps
    prev1 = np.mean([s["prevalence"]["trait1"] for s in all_stats])
    prev2 = np.mean([s["prevalence"]["trait2"] for s in all_stats])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=violin_data, x="Trait", y="Liability", hue="Affected",
        split=True, ax=ax,
    )
    ax.set_title(
        f"Liability by Affected Status (Weibull) [{scenario}]"
    )

    # Annotate mean liability for each trait x affected/unaffected group
    for i, trait_num in enumerate([1, 2]):
        liab = df_samples[f"liability{trait_num}"].values
        aff = df_samples[f"affected{trait_num}"].values.astype(bool)
        if aff.any():
            mean_aff = liab[aff].mean()
            ax.plot(i + 0.05, mean_aff, "D", color="black", markersize=6, zorder=5)
            ax.text(
                i + 0.12, mean_aff, f"\u03bc={mean_aff:.2f}",
                ha="left", va="center", fontsize=9, fontweight="bold",
            )
        if (~aff).any():
            mean_unaff = liab[~aff].mean()
            ax.plot(i - 0.05, mean_unaff, "D", color="black", markersize=6, zorder=5)
            ax.text(
                i - 0.12, mean_unaff, f"\u03bc={mean_unaff:.2f}",
                ha="right", va="center", fontsize=9, fontweight="bold",
            )

    ax.text(
        0, ax.get_ylim()[0], f"Prevalence: {prev1:.1%}",
        ha="center", va="top", fontsize=10, fontstyle="italic",
    )
    ax.text(
        1, ax.get_ylim()[0], f"Prevalence: {prev2:.1%}",
        ha="center", va="top", fontsize=10, fontstyle="italic",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_violin_by_generation(df_samples: pd.DataFrame, all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str = "") -> None:
    """Split violin of liability by affected status, one column per generation (Weibull)."""
    if "generation" not in df_samples.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No generation data", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_path, dpi=150)
        plt.close()
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

            violin_data = pd.DataFrame({
                "Trait": f"Trait {trait_num}",
                "Liability": df_gen[liab_col].values,
                "Affected": df_gen[aff_col].values,
            })

            if len(violin_data) > 1:
                sns.violinplot(
                    data=violin_data, x="Trait", y="Liability",
                    hue="Affected", split=True,
                    legend=(row == 0 and col == n_gens - 1),
                    ax=ax, cut=0,
                )

                # Annotate means
                liab = df_gen[liab_col].values
                aff = df_gen[aff_col].values.astype(bool)
                if aff.any():
                    mu = liab[aff].mean()
                    ax.plot(0.05, mu, "D", color="black", markersize=5, zorder=5)
                    ax.text(0.12, mu, f"\u03bc={mu:.2f}",
                            ha="left", va="center", fontsize=8, fontweight="bold")
                if (~aff).any():
                    mu = liab[~aff].mean()
                    ax.plot(-0.05, mu, "D", color="black", markersize=5, zorder=5)
                    ax.text(-0.12, mu, f"\u03bc={mu:.2f}",
                            ha="right", va="center", fontsize=8, fontweight="bold")

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
        f"Liability by Affected Status per Generation (Weibull) [{scenario}]",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_joint_affection(df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "") -> None:
    """2x2 heatmap of joint affection status (trait1 x trait2)."""
    a1 = df_samples["affected1"].values.astype(bool)
    a2 = df_samples["affected2"].values.astype(bool)
    n = len(df_samples)

    props = {
        "both": np.sum(a1 & a2) / n,
        "trait1_only": np.sum(a1 & ~a2) / n,
        "trait2_only": np.sum(~a1 & a2) / n,
        "neither": np.sum(~a1 & ~a2) / n,
    }
    counts = {
        "both": int(np.sum(a1 & a2)),
        "trait1_only": int(np.sum(a1 & ~a2)),
        "trait2_only": int(np.sum(~a1 & a2)),
        "neither": int(np.sum(~a1 & ~a2)),
    }

    matrix = np.array([
        [props["both"], props["trait1_only"]],
        [props["trait2_only"], props["neither"]],
    ])
    labels = np.array([
        [f"{props['both']:.2f}\n(n={counts['both']})",
         f"{props['trait1_only']:.2f}\n(n={counts['trait1_only']})"],
        [f"{props['trait2_only']:.2f}\n(n={counts['trait2_only']})",
         f"{props['neither']:.2f}\n(n={counts['neither']})"],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix, annot=labels, fmt="", cmap="Blues", ax=ax,
        xticklabels=["Affected", "Unaffected"],
        yticklabels=["Affected", "Unaffected"],
        vmin=0, vmax=max(matrix.max(), 0.01),
        cbar_kws={"label": "Proportion"},
    )
    # Cross-trait tetrachoric correlation
    r_tet = tetrachoric_corr(a1, a2)
    r_label = f"r_tet = {r_tet:.3f}" if not np.isnan(r_tet) else "r_tet = N/A"

    ax.set_xlabel("Trait 1")
    ax.set_ylabel("Trait 2")
    ax.set_title(f"Joint Affected Status (Weibull) [{scenario}]\n{r_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
