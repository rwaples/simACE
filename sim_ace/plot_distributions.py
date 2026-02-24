"""Distribution-related phenotype plots.

Contains: plot_death_age_distribution, plot_trait_phenotype, plot_trait_regression,
plot_cumulative_incidence, plot_censoring_windows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def plot_death_age_distribution(all_stats: list[dict[str, Any]], censor_age: float, output_path: str | Path, scenario: str = "") -> None:
    """Plot mortality rate and cumulative mortality by decade, averaged across reps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average mortality rates across reps
    all_rates = np.array([s["mortality"]["rates"] for s in all_stats])
    mean_rates = all_rates.mean(axis=0)
    decade_labels = all_stats[0]["mortality"]["decade_labels"]

    # Left: mortality rate per decade
    axes[0].bar(decade_labels, mean_rates, edgecolor="black", alpha=0.7)
    axes[0].set_title("Mortality Rate by Decade")
    axes[0].set_xlabel("Age Decade")
    axes[0].set_ylabel("Mortality Rate")
    axes[0].tick_params(axis="x", rotation=45)

    # Right: cumulative mortality per decade with survival annotations
    survival = np.cumprod(1 - mean_rates)
    cumulative = 1 - survival
    bars = axes[1].bar(decade_labels, cumulative, edgecolor="black", alpha=0.7)
    for bar, s in zip(bars, survival):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"S={s:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_title("Cumulative Mortality by Decade")
    axes[1].set_xlabel("Age Decade")
    axes[1].set_ylabel("Cumulative Mortality")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"Death Age Distribution (Weibull) [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trait_phenotype(df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "") -> None:
    """Plot phenotype distributions for both traits in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, trait_num in enumerate([1, 2]):
        affected_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"
        death_censored_col = f"death_censored{trait_num}"

        affected = df_samples[df_samples[affected_col] == True]
        death_censored = df_samples[
            (df_samples[affected_col] == False) & (df_samples[death_censored_col] == True)
        ]

        axes[row, 0].hist(
            affected[t_col].dropna(), bins=50, density=True,
            edgecolor="black", alpha=0.7, color="C3",
        )
        axes[row, 0].set_title(f"Trait {trait_num}: Age at Onset (affected)")
        axes[row, 0].set_xlabel("Age")
        axes[row, 0].set_ylabel("Density")

        axes[row, 1].hist(
            death_censored[t_col].dropna(), bins=50, density=True,
            edgecolor="black", alpha=0.7, color="C0",
        )
        axes[row, 1].set_title(
            f"Trait {trait_num}: Age at Death (death-censored, unaffected)"
        )
        axes[row, 1].set_xlabel("Age")
        axes[row, 1].set_ylabel("Density")

    fig.suptitle(f"Phenotype Distributions (Weibull) [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trait_regression(df_samples: pd.DataFrame, all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str = "") -> None:
    """Plot liability vs age at onset for both traits as jointplots side by side."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"Liability vs Age at Onset (Weibull) [{scenario}]", fontsize=14, y=1.01
    )
    outer = GridSpec(1, 2, figure=fig, wspace=0.35)

    for i, trait_num in enumerate([1, 2]):
        affected_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"
        liability_col = f"liability{trait_num}"

        if liability_col not in df_samples.columns:
            continue

        affected = df_samples[df_samples[affected_col] == True].dropna(
            subset=[liability_col, t_col]
        )
        x = affected[liability_col].values
        y = affected[t_col].values

        # Get R^2 from pre-computed stats (averaged across reps)
        reg_stats = [
            s["regression"][f"trait{trait_num}"]
            for s in all_stats
            if s["regression"].get(f"trait{trait_num}") is not None
        ]
        if reg_stats:
            mean_r2 = np.mean([r["r2"] for r in reg_stats])
            mean_slope = np.mean([r["slope"] for r in reg_stats])
            mean_intercept = np.mean([r["intercept"] for r in reg_stats])
        elif len(x) >= 2:
            from scipy.stats import linregress
            reg = linregress(x, y)
            mean_r2 = reg.rvalue ** 2
            mean_slope = reg.slope
            mean_intercept = reg.intercept
        else:
            continue

        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[i],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)

        ax_joint.scatter(x, y, alpha=0.15, s=3, rasterized=True)
        x_line = np.array([x.min(), x.max()])
        ax_joint.plot(
            x_line, mean_slope * x_line + mean_intercept,
            color="C3", linewidth=2,
        )
        ax_joint.text(
            0.05, 0.95, f"R\u00b2 = {mean_r2:.4f}",
            transform=ax_joint.transAxes, va="top", fontsize=12,
        )
        ax_joint.set_xlabel("Liability")
        ax_joint.set_ylabel("Age at Onset")

        ax_marg_x.hist(x, bins=50, edgecolor="none", alpha=0.7)
        ax_marg_y.hist(y, bins=50, orientation="horizontal", edgecolor="none", alpha=0.7)
        ax_marg_x.set_title(f"Trait {trait_num}", fontsize=12)
        ax_marg_x.tick_params(labelbottom=False, labelleft=False)
        ax_marg_x.set_ylabel("")
        ax_marg_y.tick_params(labelleft=False, labelbottom=False)
        ax_marg_y.set_xlabel("")

        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative_incidence(all_stats: list[dict[str, Any]], censor_age: float, output_path: str | Path, scenario: str = "") -> None:
    """Plot cumulative incidence by age, mean +/- band across reps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for trait_num, ax in zip([1, 2], axes):
        key = f"trait{trait_num}"
        ages = np.array(all_stats[0]["cumulative_incidence"][key]["ages"])

        # Support both old ("values") and new ("observed_values"/"true_values") format
        mean_true = None
        if "observed_values" in all_stats[0]["cumulative_incidence"][key]:
            all_obs = np.array([
                s["cumulative_incidence"][key]["observed_values"] for s in all_stats
            ])
            all_true = np.array([
                s["cumulative_incidence"][key]["true_values"] for s in all_stats
            ])
            mean_true = all_true.mean(axis=0)

            # True incidence (gray)
            ax.plot(ages, mean_true, color="gray", alpha=0.7, linewidth=2, label="True")
            if len(all_stats) > 1:
                ax.fill_between(ages, all_true.min(axis=0), all_true.max(axis=0),
                                alpha=0.1, color="gray")
        else:
            all_obs = np.array([
                s["cumulative_incidence"][key]["values"] for s in all_stats
            ])

        mean_obs = all_obs.mean(axis=0)

        # Observed incidence (colored)
        ax.plot(ages, mean_obs, color="C0", linewidth=2, label="Observed")
        if len(all_stats) > 1:
            ax.fill_between(ages, all_obs.min(axis=0), all_obs.max(axis=0),
                            alpha=0.2, color="C0")

        # Find age when 50% of lifetime cases are diagnosed (from observed curve)
        lifetime_prev = mean_obs[-1]
        half_target = lifetime_prev / 2
        idx_50 = np.searchsorted(mean_obs, half_target)
        age_50 = ages[min(idx_50, len(ages) - 1)]

        ax.axhline(half_target, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(age_50, color="grey", linestyle="--", linewidth=0.8)
        ax.plot(age_50, half_target, "o", color="C3", markersize=6, zorder=5)
        ax.annotate(
            f"50% of cases\nby age {age_50:.0f}",
            xy=(age_50, half_target), xytext=(12, 12), textcoords="offset points",
            fontsize=10, ha="left", va="bottom",
        )
        # Annotation box: prevalence and censoring rates
        prev = np.mean([s["prevalence"][key] for s in all_stats])
        true_prev = mean_true[-1] if mean_true is not None else mean_obs[-1]
        censored_pct = (true_prev - prev) * 100
        ax.text(
            0.03, 0.95,
            f"Affected: {prev * 100:.1f}%\n"
            f"True prev: {true_prev * 100:.1f}%\n"
            f"Censored: {censored_pct:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_title(f"Trait {trait_num}")
        ax.set_xlabel("Age")
        ax.legend(loc="lower right", fontsize=9)

    axes[0].set_ylabel("Cumulative Incidence")
    fig.suptitle(f"Cumulative Incidence by Age (Weibull) [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_censoring_windows(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    gen_censoring: dict[int, list[float]] | None = None,
) -> None:
    """Plot per-generation censoring windows, mean +/- band across reps."""
    # Check that all reps have censoring data
    stats_with_censoring = [s for s in all_stats if s.get("censoring") is not None]
    if not stats_with_censoring:
        logger.warning("Skipping censoring_windows plot: no censoring data in stats")
        # Create empty plot to satisfy Snakemake output
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No censoring data", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    ages = np.array(stats_with_censoring[0]["censoring"]["censoring_ages"])

    # Discover generation keys from the stats YAML (e.g. "gen0", "gen1", ...)
    # Only include generations that have phenotyped individuals in any replicate
    all_gen_keys = sorted(stats_with_censoring[0]["censoring"]["generations"].keys())
    gen_keys = [
        gk for gk in all_gen_keys
        if any(
            s["censoring"]["generations"].get(gk, {}).get("n", 0) > 0
            for s in stats_with_censoring
        )
    ]
    if not gen_keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No phenotyped generations", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return
    if gen_censoring is None:
        gen_censoring = {}

    gen_labels = []
    for gk in gen_keys:
        gen_num = int(gk.replace("gen", ""))
        win = gen_censoring.get(gen_num)
        if win is not None:
            gen_labels.append(f"Gen {gen_num}\n[{win[0]}, {win[1]}]")
        else:
            gen_labels.append(f"Gen {gen_num}")

    traits = [1, 2]

    fig, axes = plt.subplots(
        len(traits), len(gen_keys),
        figsize=(5 * len(gen_keys), 4 * len(traits)),
        sharex=True, sharey=True, squeeze=False,
    )

    for col, (gen_key, label) in enumerate(zip(gen_keys, gen_labels)):
        # Check if any rep has data for this generation
        gen_data = [
            s["censoring"]["generations"][gen_key]
            for s in stats_with_censoring
            if s["censoring"]["generations"].get(gen_key, {}).get("n", 0) > 0
        ]
        if not gen_data:
            logger.warning("plot_censoring_windows: generation '%s' has 0 individuals", gen_key)
            for row in range(len(traits)):
                axes[row, col].text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=axes[row, col].transAxes,
                )
            continue

        for row, trait_num in enumerate(traits):
            ax = axes[row, col]
            key = f"trait{trait_num}"

            all_true = np.array([g[key]["true_incidence"] for g in gen_data])
            all_obs = np.array([g[key]["observed_incidence"] for g in gen_data])

            mean_true = all_true.mean(axis=0)
            mean_obs = all_obs.mean(axis=0)

            ax.plot(ages, mean_true, color="gray", alpha=0.7, linewidth=2, label="True")
            ax.fill_between(ages, mean_true, alpha=0.15, color="gray")
            ax.plot(ages, mean_obs, color="C0", linewidth=2, label="Observed")
            ax.fill_between(ages, mean_obs, alpha=0.2, color="C0")

            if len(stats_with_censoring) > 1:
                ax.fill_between(
                    ages, all_true.min(axis=0), all_true.max(axis=0),
                    alpha=0.08, color="gray",
                )
                ax.fill_between(
                    ages, all_obs.min(axis=0), all_obs.max(axis=0),
                    alpha=0.08, color="C0",
                )

            # Annotation stats (averaged)
            pct_affected = np.mean([g[key]["pct_affected"] for g in gen_data]) * 100
            left_cens = np.mean([g[key]["left_censored"] for g in gen_data]) * 100
            right_cens = np.mean([g[key]["right_censored"] for g in gen_data]) * 100
            death_cens = np.mean([g[key]["death_censored"] for g in gen_data]) * 100

            ax.text(
                0.03, 0.95,
                f"Affected: {pct_affected:.1f}%\n"
                f"Left-cens: {left_cens:.1f}%\n"
                f"Right-cens: {right_cens:.1f}%\n"
                f"Death-cens: {death_cens:.1f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            if row == 0:
                ax.set_title(label, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nCumulative Incidence")
            if row == len(traits) - 1:
                ax.set_xlabel("Age")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2, alpha=0.7, label="True"),
        Line2D([0], [0], color="C0", linewidth=2, label="Observed"),
    ]
    axes[0, -1].legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.suptitle(
        f"Censoring Windows by Generation (Weibull) [{scenario}]", fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
