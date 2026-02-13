"""
Plot phenotype distributions from pre-computed per-rep statistics.

Reads phenotype_stats.yaml and phenotype_samples.parquet files (one per rep)
produced by compute_phenotype_stats.py. No full phenotype parquet loading needed.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

try:
    _yaml_loader = yaml.CSafeLoader
except AttributeError:
    _yaml_loader = yaml.SafeLoader
from pathlib import Path

MAX_PLOT_POINTS = 100_000

sys.path.insert(0, str(Path(__file__).parent))
from compute_phenotype_stats import tetrachoric_corr


def plot_death_age_distribution(all_stats, censor_age, output_path, scenario=""):
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


def plot_trait_phenotype(df_samples, output_path, scenario=""):
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


def plot_trait_regression(df_samples, all_stats, output_path, scenario=""):
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

        ax_joint.scatter(x, y, alpha=0.05, s=3, rasterized=True)
        x_line = np.array([x.min(), x.max()])
        ax_joint.plot(
            x_line, mean_slope * x_line + mean_intercept,
            color="C3", linewidth=2,
        )
        ax_joint.text(
            0.05, 0.95, f"R² = {mean_r2:.4f}",
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


def plot_liability_joint(df_samples, output_path, scenario=""):
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
    fig.suptitle(f"Cross-Trait Correlations (Weibull) [{scenario}]", fontsize=14, y=1.01)
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


def plot_liability_violin(df_samples, all_stats, output_path, scenario=""):
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
        mean_aff = liab[aff].mean()
        mean_unaff = liab[~aff].mean()
        ax.plot(i + 0.05, mean_aff, "D", color="black", markersize=6, zorder=5)
        ax.text(
            i + 0.12, mean_aff, f"\u03bc={mean_aff:.2f}",
            ha="left", va="center", fontsize=9, fontweight="bold",
        )
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


def plot_cumulative_incidence(all_stats, censor_age, output_path, scenario=""):
    """Plot cumulative incidence by age, mean +/- band across reps.

    Shows both true (uncensored) and observed (post-censoring) curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for trait_num, ax in zip([1, 2], axes):
        key = f"trait{trait_num}"
        ages = np.array(all_stats[0]["cumulative_incidence"][key]["ages"])

        # Support both old ("values") and new ("observed_values"/"true_values") format
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
        ax.set_title(f"Trait {trait_num}")
        ax.set_xlabel("Age")
        ax.legend(loc="lower right", fontsize=9)

    axes[0].set_ylabel("Cumulative Incidence")
    fig.suptitle(f"Cumulative Incidence by Age (Weibull) [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_censoring_windows(all_stats, output_path, scenario="",
                           old_gen_censoring=None, middle_gen_censoring=None,
                           young_gen_censoring=None):
    """Plot per-generation censoring windows, mean +/- band across reps."""
    # Check that all reps have censoring data
    stats_with_censoring = [s for s in all_stats if s.get("censoring") is not None]
    if not stats_with_censoring:
        print("Skipping censoring_windows plot: no censoring data in stats")
        # Create empty plot to satisfy Snakemake output
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No censoring data", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    ages = np.array(stats_with_censoring[0]["censoring"]["censoring_ages"])
    censor_age = stats_with_censoring[0]["censoring"]["censor_age"]

    gen_names = ["old", "middle", "young"]
    gen_windows = {
        "old": old_gen_censoring,
        "middle": middle_gen_censoring,
        "young": young_gen_censoring,
    }
    gen_labels = []
    for name in gen_names:
        label = name.capitalize()
        win = gen_windows.get(name)
        if win is not None:
            gen_labels.append(f"{label}\n[{win[0]}, {win[1]}]")
        else:
            gen_labels.append(label)

    # Infer censoring windows from the stats
    # We need them for the vertical lines - read from snakemake params passed to main()
    traits = [1, 2]

    fig, axes = plt.subplots(
        len(traits), len(gen_names),
        figsize=(5 * len(gen_names), 4 * len(traits)),
        sharex=True, sharey=True, squeeze=False,
    )

    for col, (gen_name, label) in enumerate(zip(gen_names, gen_labels)):
        # Check if any rep has data for this generation
        gen_data = [
            s["censoring"]["generations"][gen_name]
            for s in stats_with_censoring
            if s["censoring"]["generations"].get(gen_name, {}).get("n", 0) > 0
        ]
        if not gen_data:
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
    legend_ax = axes[0, 1]
    handles, labels = legend_ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="lower right", fontsize=8)
    fig.suptitle(
        f"Censoring Windows by Generation (Weibull) [{scenario}]", fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_joint_affection(df_samples, output_path, scenario=""):
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
    ax.set_title(f"Joint Affection Status (Weibull) [{scenario}]\n{r_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tetrachoric_sibling(all_stats, output_path, scenario):
    """Plot tetrachoric correlations by relationship type, mean with rep dots."""
    pair_types = ["MZ twin", "Full sib", "Mother-offspring", "Father-offspring", "Maternal half sib", "Paternal half sib", "1st cousin"]
    bar_colors = ["C0", "C1", "C3", "C5", "C2", "C6", "C4"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col_idx, trait_num in enumerate([1, 2]):
        ax = axes[col_idx]
        key = f"trait{trait_num}"

        mean_corrs = []
        se_bars = []
        total_pairs = []
        rep_corrs = {pt: [] for pt in pair_types}

        for ptype in pair_types:
            corrs = []
            n_total = 0
            for s in all_stats:
                entry = s["tetrachoric"][key].get(ptype, {})
                r = entry.get("r")
                n_p = entry.get("n_pairs", 0)
                if r is not None:
                    corrs.append(r)
                n_total += n_p
                rep_corrs[ptype].append(r)

            mean_corrs.append(np.mean(corrs) if corrs else np.nan)
            if len(corrs) > 1:
                se_bars.append(np.std(corrs) / np.sqrt(len(corrs)))
            else:
                se_bars.append(0)
            total_pairs.append(n_total)

        x = np.arange(len(pair_types))
        bars = ax.bar(
            x, mean_corrs, width=0.6, color=bar_colors,
            edgecolor="black", alpha=0.8, zorder=3,
        )

        # Error bars (cross-rep SE)
        ax.errorbar(
            x, mean_corrs, yerr=se_bars, fmt="none",
            ecolor="black", elinewidth=1.5, capsize=4, zorder=4,
        )

        # Overlay per-rep dots
        for i, ptype in enumerate(pair_types):
            rep_vals = [v for v in rep_corrs[ptype] if v is not None]
            if rep_vals:
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(rep_vals))
                ax.scatter(
                    x[i] + jitter, rep_vals, color="black", s=12,
                    alpha=0.4, zorder=5,
                )

        # Annotate mean N pairs per rep
        n_reps = len(all_stats)
        for i, (bar, n_p) in enumerate(zip(bars, total_pairs)):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + se_bars[i] + 0.01,
                    f"N={n_p // n_reps}", ha="center", va="bottom", fontsize=8,
                )

        # Liability correlation lines (averaged across reps)
        for i, ptype in enumerate(pair_types):
            liab_vals = [
                s.get("liability_correlations", {}).get(key, {}).get(ptype)
                for s in all_stats
            ]
            liab_vals = [v for v in liab_vals if v is not None]
            if liab_vals:
                mean_liab_r = np.mean(liab_vals)
                ax.hlines(
                    mean_liab_r, i - 0.35, i + 0.35, colors="black",
                    linestyles="dashed", linewidth=2, zorder=4,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(pair_types, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Tetrachoric Correlation")
        ax.set_title(f"Trait {trait_num}")
        ax.set_ylim(-0.1, 1.1)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Liability r"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    fig.suptitle(
        f"Tetrachoric Correlation (Weibull) [{scenario}]", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(stats_paths, sample_paths, output_dir, censor_age,
         young_gen_censoring=None, middle_gen_censoring=None,
         old_gen_censoring=None):
    """Generate all phenotype plots from pre-computed stats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = output_dir.parent.name
    sns.set_theme(style="whitegrid")

    # Load per-rep stats
    all_stats = []
    for p in stats_paths:
        with open(p) as f:
            all_stats.append(yaml.load(f, Loader=_yaml_loader))

    # Load and concatenate downsampled data
    df_samples = pd.concat(
        [pd.read_parquet(p) for p in sample_paths], ignore_index=True
    )

    # Subsample for plotting (scatter/violin are O(n) slow for >100K points)
    if len(df_samples) > MAX_PLOT_POINTS:
        df_samples = df_samples.sample(
            n=MAX_PLOT_POINTS, random_state=42
        ).reset_index(drop=True)

    plot_death_age_distribution(
        all_stats, censor_age, output_dir / "death_age_distribution.png", scenario
    )
    plot_trait_phenotype(
        df_samples, output_dir / "phenotype_traits.png", scenario
    )
    plot_trait_regression(
        df_samples, all_stats, output_dir / "liability_regression.png", scenario
    )
    plot_liability_joint(
        df_samples, output_dir / "liability_joint.png", scenario
    )
    plot_liability_violin(
        df_samples, all_stats, output_dir / "liability_violin.png", scenario
    )
    plot_cumulative_incidence(
        all_stats, censor_age, output_dir / "cumulative_incidence.png", scenario
    )
    plot_joint_affection(
        df_samples, output_dir / "joint_affection.png", scenario
    )
    if young_gen_censoring is not None:
        plot_censoring_windows(
            all_stats, output_dir / "censoring_windows.png", scenario,
            old_gen_censoring=old_gen_censoring,
            middle_gen_censoring=middle_gen_censoring,
            young_gen_censoring=young_gen_censoring,
        )
    else:
        # Create placeholder to satisfy Snakemake output
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No censoring windows configured",
                ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_dir / "censoring_windows.png", dpi=150)
        plt.close()

    plot_tetrachoric_sibling(
        all_stats, output_dir / "tetrachoric_relationship.png", scenario,
    )

    print(f"Phenotype plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    try:
        stats_paths = snakemake.input.stats
        sample_paths = snakemake.input.samples
        censor_age = snakemake.params.censor_age
        young_gen_censoring = snakemake.params.young_gen_censoring
        middle_gen_censoring = snakemake.params.middle_gen_censoring
        old_gen_censoring = snakemake.params.old_gen_censoring
        output_dir = Path(snakemake.output[0]).parent
    except NameError:
        print("Usage: run via snakemake or provide stats/sample paths")
        sys.exit(1)

    main(stats_paths, sample_paths, output_dir, censor_age,
         young_gen_censoring, middle_gen_censoring, old_gen_censoring)
