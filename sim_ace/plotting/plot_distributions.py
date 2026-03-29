"""Distribution-related phenotype plots.

Contains: plot_death_age_distribution, plot_trait_phenotype, plot_trait_regression,
plot_cumulative_incidence, plot_cumulative_incidence_by_sex,
plot_cumulative_incidence_by_sex_generation, plot_censoring_windows.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

from sim_ace.plotting.plot_utils import finalize_plot, save_placeholder_plot

logger = logging.getLogger(__name__)

# Consistent sex colour scheme across all plots
COLOR_FEMALE = "#5ab4ac"
COLOR_MALE = "#4a90d9"


def plot_death_age_distribution(
    all_stats: list[dict[str, Any]], censor_age: float, output_path: str | Path, scenario: str = ""
) -> None:
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
    for bar, s in zip(bars, survival, strict=True):
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

    fig.suptitle(f"Death Age Distribution [{scenario}]", fontsize=14)
    finalize_plot(output_path)


def plot_trait_phenotype(
    df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "", subsample_note: str = ""
) -> None:
    """Plot phenotype distributions for both traits in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, trait_num in enumerate([1, 2]):
        affected_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"
        death_censored_col = f"death_censored{trait_num}"

        affected = df_samples[df_samples[affected_col]]
        death_censored = df_samples[~df_samples[affected_col] & df_samples[death_censored_col]]

        axes[row, 0].hist(
            affected[t_col].dropna(),
            bins=50,
            density=True,
            edgecolor="black",
            alpha=0.7,
            color="C3",
        )
        axes[row, 0].set_title(f"Trait {trait_num}: Age at Onset (affected)")
        axes[row, 0].set_xlabel("Age")
        axes[row, 0].set_ylabel("Density")

        axes[row, 1].hist(
            death_censored[t_col].dropna(),
            bins=50,
            density=True,
            edgecolor="black",
            alpha=0.7,
            color="C0",
        )
        axes[row, 1].set_title(f"Trait {trait_num}: Age at Death (death-censored, unaffected)")
        axes[row, 1].set_xlabel("Age")
        axes[row, 1].set_ylabel("Density")

    fig.suptitle(f"Phenotype Distributions [{scenario}]", fontsize=14)
    finalize_plot(output_path, subsample_note=subsample_note)


def plot_trait_regression(
    df_samples: pd.DataFrame,
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    subsample_note: str = "",
) -> None:
    """Plot liability vs age at onset for both traits as jointplots side by side."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Liability vs Age at Onset [{scenario}]", fontsize=14, y=1.01)
    outer = GridSpec(1, 2, figure=fig, wspace=0.35)

    for i, trait_num in enumerate([1, 2]):
        affected_col = f"affected{trait_num}"
        t_col = f"t_observed{trait_num}"
        liability_col = f"liability{trait_num}"

        if liability_col not in df_samples.columns:
            continue

        affected = df_samples[df_samples[affected_col]].dropna(subset=[liability_col, t_col])
        x = affected[liability_col].values
        y = affected[t_col].values

        # Get regression stats from pre-computed stats (averaged across reps)
        reg_stats = [
            s["regression"][f"trait{trait_num}"]
            for s in all_stats
            if s["regression"].get(f"trait{trait_num}") is not None
        ]
        if reg_stats:
            mean_r = np.mean([r["r"] for r in reg_stats])
            mean_slope = np.mean([r["slope"] for r in reg_stats])
            mean_intercept = np.mean([r["intercept"] for r in reg_stats])
            mean_n = int(np.mean([r["n"] for r in reg_stats]))
            stderr_vals = [r["stderr"] for r in reg_stats if r.get("stderr") is not None]
            mean_stderr = float(np.mean(stderr_vals)) if stderr_vals else None
        elif len(x) >= 2:
            from sim_ace.core.utils import fast_linregress

            mean_slope, mean_intercept, mean_r, mean_stderr, _mean_pvalue = fast_linregress(x, y)
            mean_n = len(x)
        else:
            continue

        inner = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer[i],
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)

        ax_joint.plot(x, y, "o", ms=2, mew=0, alpha=0.15, rasterized=True)
        x_line = np.array([x.min(), x.max()])
        ax_joint.plot(
            x_line,
            mean_slope * x_line + mean_intercept,
            color="C3",
            linewidth=2,
        )

        # 95% confidence band
        if mean_stderr is not None and mean_n > 2:
            from scipy.stats import t as t_dist

            x_smooth = np.linspace(x.min(), x.max(), 200)
            y_hat = mean_slope * x_smooth + mean_intercept
            x_mean = np.mean(x)
            ss_x = np.sum((x - x_mean) ** 2)
            if ss_x > 1e-12:
                s = mean_stderr * np.sqrt(ss_x)
                t_crit = t_dist.ppf(0.975, df=mean_n - 2)
                se_fit = s * np.sqrt(1.0 / mean_n + (x_smooth - x_mean) ** 2 / ss_x)
                ax_joint.fill_between(
                    x_smooth,
                    y_hat - t_crit * se_fit,
                    y_hat + t_crit * se_fit,
                    alpha=0.15,
                    color="C3",
                    zorder=2,
                )

        # Annotation: slope, r
        ann_lines = [f"slope = {mean_slope:.4f}", f"r = {mean_r:.4f}"]
        ax_joint.text(
            0.05,
            0.95,
            "\n".join(ann_lines),
            transform=ax_joint.transAxes,
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
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

    finalize_plot(output_path, subsample_note=subsample_note)


def plot_cumulative_incidence(
    all_stats: list[dict[str, Any]], censor_age: float, output_path: str | Path, scenario: str = ""
) -> None:
    """Plot cumulative incidence by age, mean +/- band across reps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for trait_num, ax in zip([1, 2], axes, strict=True):
        key = f"trait{trait_num}"
        ages = np.array(all_stats[0]["cumulative_incidence"][key]["ages"])

        # Support both old ("values") and new ("observed_values"/"true_values") format
        mean_true = None
        if "observed_values" in all_stats[0]["cumulative_incidence"][key]:
            all_obs = np.array([s["cumulative_incidence"][key]["observed_values"] for s in all_stats])
            all_true = np.array([s["cumulative_incidence"][key]["true_values"] for s in all_stats])
            mean_true = all_true.mean(axis=0)

            # True incidence (gray)
            ax.plot(ages, mean_true, color="gray", alpha=0.7, linewidth=2, label="True")
            if len(all_stats) > 1:
                ax.fill_between(ages, all_true.min(axis=0), all_true.max(axis=0), alpha=0.1, color="gray")
        else:
            all_obs = np.array([s["cumulative_incidence"][key]["values"] for s in all_stats])

        mean_obs = all_obs.mean(axis=0)

        # Observed incidence (colored)
        ax.plot(ages, mean_obs, color="C0", linewidth=2, label="Observed")
        if len(all_stats) > 1:
            ax.fill_between(ages, all_obs.min(axis=0), all_obs.max(axis=0), alpha=0.2, color="C0")

        # Annotate Q1, Q2 (median), Q3 on both observed and true curves
        quartile_points: dict[str, dict[str, tuple[float, float]]] = {}
        for curve, curve_color, y_offset, curve_key in [
            (mean_obs, "C0", -16, "obs"),
            (mean_true, "gray", 16, "true"),
        ]:
            if curve is None:
                continue
            lifetime = curve[-1]
            if lifetime <= 0:
                continue
            for frac, label, ms in [
                (0.25, "Q1", 4),
                (0.50, "Q2", 6),
                (0.75, "Q3", 4),
            ]:
                target = lifetime * frac
                idx_q = np.searchsorted(curve, target)
                age_q = ages[min(idx_q, len(ages) - 1)]

                ax.plot(age_q, target, "o", color=curve_color, markersize=ms, zorder=5)
                ax.annotate(
                    f"{label}: {age_q:.0f}",
                    xy=(age_q, target),
                    xytext=(10, y_offset),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    color=curve_color,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
                )
                quartile_points.setdefault(label, {})[curve_key] = (age_q, target)

        # Connect matching quartiles between observed and true curves
        for label in ["Q1", "Q2", "Q3"]:
            pts = quartile_points.get(label, {})
            if "obs" in pts and "true" in pts:
                ax.plot(
                    [pts["obs"][0], pts["true"][0]],
                    [pts["obs"][1], pts["true"][1]],
                    color="0.5",
                    linestyle="--",
                    linewidth=0.8,
                    zorder=4,
                )
        # Annotation box: prevalence and censoring rates
        prev = np.mean([s["prevalence"][key] for s in all_stats])
        true_prev = mean_true[-1] if mean_true is not None else mean_obs[-1]
        censored_pct = (true_prev - prev) * 100
        ax.text(
            0.03,
            0.95,
            f"Affected: {prev * 100:.1f}%\nTrue prev: {true_prev * 100:.1f}%\nCensored: {censored_pct:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_title(f"Trait {trait_num}")
        ax.set_xlabel("Age")
        ax.legend(loc="lower right", fontsize=9)

    axes[0].set_ylabel("Cumulative Incidence")
    fig.suptitle(f"Cumulative Incidence by Age [{scenario}]", fontsize=14)
    finalize_plot(output_path)


def plot_cumulative_incidence_by_sex(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot cumulative incidence curves split by sex, from pre-computed stats."""
    stats_with_data = [s for s in all_stats if s.get("cumulative_incidence_by_sex")]
    if not stats_with_data:
        logger.warning("Skipping cumulative_incidence_by_sex: no data in stats")
        save_placeholder_plot(output_path, "No sex-stratified incidence data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for trait_num, ax in zip([1, 2], axes, strict=True):
        key = f"trait{trait_num}"

        for sex_label, display, color in [
            ("female", "Female", COLOR_FEMALE),
            ("male", "Male", COLOR_MALE),
        ]:
            rep_data = [
                s["cumulative_incidence_by_sex"][key][sex_label]
                for s in stats_with_data
                if sex_label in s["cumulative_incidence_by_sex"].get(key, {})
            ]
            if not rep_data:
                continue

            ages = np.array(rep_data[0]["ages"])
            all_values = np.array([d["values"] for d in rep_data])
            mean_values = all_values.mean(axis=0)
            mean_n = np.mean([d["n"] for d in rep_data])
            mean_prev = np.mean([d["prevalence"] for d in rep_data])

            ax.plot(
                ages, mean_values, color=color, linewidth=2, label=f"{display} (n={int(mean_n)}, prev={mean_prev:.1%})"
            )

        ax.set_title(f"Trait {trait_num}")
        ax.set_xlabel("Age")
        ax.legend(loc="lower right", fontsize=9)

    axes[0].set_ylabel("Cumulative Incidence")
    fig.suptitle(f"Cumulative Incidence by Sex [{scenario}]", fontsize=14)
    finalize_plot(output_path)


def plot_cumulative_incidence_by_sex_generation(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot cumulative incidence by sex and generation, from pre-computed stats."""
    stats_with_data = [s for s in all_stats if s.get("cumulative_incidence_by_sex_generation")]
    if not stats_with_data:
        logger.warning("Skipping cumulative_incidence_by_sex_generation: no data in stats")
        save_placeholder_plot(output_path, "No sex/generation incidence data")
        return

    # Discover generation keys from first rep's first trait
    first_trait = stats_with_data[0]["cumulative_incidence_by_sex_generation"].get("trait1", {})
    gen_keys = sorted(first_trait.keys())
    if not gen_keys:
        save_placeholder_plot(output_path, "No generations")
        return

    traits = [1, 2]

    fig, axes = plt.subplots(
        len(traits),
        len(gen_keys),
        figsize=(5 * len(gen_keys), 4 * len(traits)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for col, gk in enumerate(gen_keys):
        gen_num = gk.replace("gen", "")

        for row, trait_num in enumerate(traits):
            ax = axes[row, col]
            key = f"trait{trait_num}"

            for sex_label, display, color in [
                ("female", "Female", COLOR_FEMALE),
                ("male", "Male", COLOR_MALE),
            ]:
                rep_data = [
                    s["cumulative_incidence_by_sex_generation"][key][gk][sex_label]
                    for s in stats_with_data
                    if sex_label in s["cumulative_incidence_by_sex_generation"].get(key, {}).get(gk, {})
                ]
                if not rep_data:
                    continue

                ages = np.array(rep_data[0]["ages"])
                all_values = np.array([d["values"] for d in rep_data])
                mean_values = all_values.mean(axis=0)
                mean_n = np.mean([d["n"] for d in rep_data])
                mean_prev = np.mean([d["prevalence"] for d in rep_data])

                ax.plot(
                    ages, mean_values, color=color, linewidth=2, label=f"{display} (n={int(mean_n)}, {mean_prev:.1%})"
                )

            if row == 0:
                ax.set_title(f"Gen {gen_num}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nCumulative Incidence")
            if row == len(traits) - 1:
                ax.set_xlabel("Age")
            if col == len(gen_keys) - 1:
                ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Cumulative Incidence by Sex and Generation [{scenario}]", fontsize=14, y=1.01)
    finalize_plot(output_path)


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
        save_placeholder_plot(output_path, "No censoring data")
        return

    ages = np.array(stats_with_censoring[0]["censoring"]["censoring_ages"])

    # Discover generation keys from the stats YAML (e.g. "gen0", "gen1", ...)
    # Only include generations that have phenotyped individuals in any replicate
    all_gen_keys = sorted(stats_with_censoring[0]["censoring"]["generations"].keys())
    gen_keys = [
        gk
        for gk in all_gen_keys
        if any(s["censoring"]["generations"].get(gk, {}).get("n", 0) > 0 for s in stats_with_censoring)
    ]
    if not gen_keys:
        save_placeholder_plot(output_path, "No phenotyped generations")
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
        len(traits),
        len(gen_keys),
        figsize=(5 * len(gen_keys), 4 * len(traits)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for col, (gen_key, label) in enumerate(zip(gen_keys, gen_labels, strict=True)):
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
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
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
                    ages,
                    all_true.min(axis=0),
                    all_true.max(axis=0),
                    alpha=0.08,
                    color="gray",
                )
                ax.fill_between(
                    ages,
                    all_obs.min(axis=0),
                    all_obs.max(axis=0),
                    alpha=0.08,
                    color="C0",
                )

            # Annotation stats (averaged)
            pct_affected = np.mean([g[key]["pct_affected"] for g in gen_data]) * 100
            left_cens = np.mean([g[key]["left_censored"] for g in gen_data]) * 100
            right_cens = np.mean([g[key]["right_censored"] for g in gen_data]) * 100
            death_cens = np.mean([g[key]["death_censored"] for g in gen_data]) * 100

            ax.text(
                0.03,
                0.95,
                f"Affected: {pct_affected:.1f}%\n"
                f"Left-cens: {left_cens:.1f}%\n"
                f"Right-cens: {right_cens:.1f}%\n"
                f"Death-cens: {death_cens:.1f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
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
    fig.suptitle(f"Censoring Windows by Generation [{scenario}]", fontsize=14, y=1.01)
    finalize_plot(output_path)


def plot_family_structure(all_stats: list[dict], output_path: str | Path, scenario: str = "") -> None:
    """Plot offspring and mate count distributions, averaged across replicates."""
    # Collect family_size dicts from each replicate
    fs_list = [s.get("family_size", {}) for s in all_stats if "family_size" in s]
    if not fs_list:
        save_placeholder_plot(output_path, "Family Structure", "No family_size data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Offspring per mating ---
    ax = axes[0]
    categories = ["1", "2", "3", "4+"]
    vals = np.array([[fs.get("size_dist", {}).get(c, 0) for c in categories] for fs in fs_list])
    mean_vals = vals.mean(axis=0)
    ax.bar(categories, mean_vals, color="C0", edgecolor="white")
    for i, v in enumerate(mean_vals):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", va="bottom", fontsize=10)
    mean_size = np.mean([fs.get("mean", 0) for fs in fs_list])
    ax.set_title("Offspring per Couple")
    ax.set_xlabel("Number of children")
    ax.set_ylabel("Fraction of couples")
    ax.text(
        0.97,
        0.95,
        f"mean = {mean_size:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # --- Panel 2: Offspring per person (by sex) ---
    ax = axes[1]
    categories2 = ["0", "1", "2", "3", "4+"]
    x2 = np.arange(len(categories2))
    w2 = 0.35
    # Try sex-stratified data first, fall back to pooled
    has_sex = any(fs.get("person_offspring_dist_by_sex") for fs in fs_list)
    if has_sex:
        vals_f = np.array(
            [
                [fs.get("person_offspring_dist_by_sex", {}).get("female", {}).get(c, 0) for c in categories2]
                for fs in fs_list
            ]
        )
        vals_m = np.array(
            [
                [fs.get("person_offspring_dist_by_sex", {}).get("male", {}).get(c, 0) for c in categories2]
                for fs in fs_list
            ]
        )
        mean_f = vals_f.mean(axis=0)
        mean_m = vals_m.mean(axis=0)
        ax.bar(x2 - w2 / 2, mean_f, w2, label="Female", color=COLOR_FEMALE, edgecolor="white")
        ax.bar(x2 + w2 / 2, mean_m, w2, label="Male", color=COLOR_MALE, edgecolor="white")
        # Annotate bars — merge label when F/M values are close
        for i in range(len(categories2)):
            fv, mv = mean_f[i], mean_m[i]
            if fv < 0.005 and mv < 0.005:
                continue
            if abs(fv - mv) < 0.01:
                # Values nearly equal — single centred label
                ax.text(x2[i], max(fv, mv) + 0.008, f"{(fv + mv) / 2:.0%}", ha="center", va="bottom", fontsize=9)
            else:
                if fv > 0.005:
                    ax.text(x2[i] - w2 / 2, fv + 0.005, f"{fv:.0%}", ha="center", va="bottom", fontsize=8)
                if mv > 0.005:
                    ax.text(x2[i] + w2 / 2, mv + 0.005, f"{mv:.0%}", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=9)
    else:
        vals2 = np.array([[fs.get("person_offspring_dist", {}).get(c, 0) for c in categories2] for fs in fs_list])
        mean_vals2 = vals2.mean(axis=0)
        ax.bar(x2, mean_vals2, 0.6, color="C1", edgecolor="white")
        for i, v in enumerate(mean_vals2):
            ax.text(i, v + 0.005, f"{v:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x2)
    ax.set_xticklabels(categories2)
    ax.set_title("Offspring per Person")
    ax.set_xlabel("Number of offspring")
    ax.set_ylabel("Fraction of individuals")

    # --- Panel 3: Mates per parent ---
    ax = axes[2]
    mates_list = [fs.get("mates_by_sex", {}) for fs in fs_list]
    f1 = np.mean([m.get("female_1", 0) for m in mates_list])
    f2 = np.mean([m.get("female_2+", 0) for m in mates_list])
    m1 = np.mean([m.get("male_1", 0) for m in mates_list])
    m2 = np.mean([m.get("male_2+", 0) for m in mates_list])
    x = np.arange(2)
    w = 0.35
    bars_f = ax.bar(x - w / 2, [f1, f2], w, label="Female", color=COLOR_FEMALE, edgecolor="white")
    bars_m = ax.bar(x + w / 2, [m1, m2], w, label="Male", color=COLOR_MALE, edgecolor="white")
    for bar in list(bars_f) + list(bars_m):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["1 partner", "2+ partners"])
    ax.set_ylabel("Fraction of parents")
    ax.set_title("Mates per Parent")
    ax.legend(fontsize=10)
    f_mean = np.mean([m.get("female_mean", 0) for m in mates_list])
    m_mean = np.mean([m.get("male_mean", 0) for m in mates_list])
    ax.text(
        0.97,
        0.95,
        f"mean F={f_mean:.2f}, M={m_mean:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.suptitle(f"Family Structure [{scenario}]", fontsize=14, y=1.01)
    finalize_plot(output_path)
