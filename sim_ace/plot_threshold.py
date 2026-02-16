"""
Plot liability threshold phenotype distributions from pre-computed stats.

Reads threshold_stats.yaml and threshold_samples.parquet files (one per rep)
produced by compute_threshold_stats.py.
"""

import argparse

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

from sim_ace.stats import tetrachoric_corr

MAX_PLOT_POINTS = 100_000


def plot_prevalence_by_generation(all_stats, prevalence1, prevalence2, output_path, scenario=""):
    """Grouped bar chart: observed prevalence per generation per trait."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect per-gen prevalence across reps
    gens = all_stats[0]["prevalence"]["trait1"]["generations"]
    n_gens = len(gens)

    for trait_num, color, expected in [(1, "C0", prevalence1), (2, "C3", prevalence2)]:
        key = f"trait{trait_num}"
        all_prev = np.array([s["prevalence"][key]["prevalence"] for s in all_stats])
        mean_prev = all_prev.mean(axis=0)

        x = np.arange(n_gens)
        offset = -0.2 if trait_num == 1 else 0.2
        bars = ax.bar(
            x + offset, mean_prev, width=0.35, color=color,
            edgecolor="black", alpha=0.8, label=f"Trait {trait_num}",
        )

        # Per-rep dots
        for rep_prev in all_prev:
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, n_gens)
            ax.scatter(
                x + offset + jitter, rep_prev, color="black",
                s=12, alpha=0.3, zorder=5,
            )

        # Expected prevalence line
        ax.hlines(
            expected, -0.5, n_gens - 0.5, colors=color,
            linestyles="dashed", linewidth=2, alpha=0.7,
        )

    ax.set_xticks(np.arange(n_gens))
    ax.set_xticklabels([f"Gen {g}" for g in gens])
    ax.set_ylabel("Prevalence")
    ax.set_ylim(0, max(0.3, ax.get_ylim()[1]))
    ax.set_title(f"Prevalence by Generation (Liability Threshold) [{scenario}]", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_liability_violin(df_samples, all_stats, output_path, scenario=""):
    """Split violin: liability distribution by affected status, per trait."""
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

    prev1 = np.mean([s["prevalence"]["trait1"]["overall"] for s in all_stats])
    prev2 = np.mean([s["prevalence"]["trait2"]["overall"] for s in all_stats])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=violin_data, x="Trait", y="Liability", hue="Affected",
        split=True, ax=ax,
    )
    ax.set_title(
        f"Liability by Affected Status (Liability Threshold) [{scenario}]"
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


def plot_tetrachoric(all_stats, output_path, scenario):
    """Tetrachoric correlations by relationship type with liability correlation lines."""
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

        ax.errorbar(
            x, mean_corrs, yerr=se_bars, fmt="none",
            ecolor="black", elinewidth=1.5, capsize=4, zorder=4,
        )

        # Per-rep dots
        for i, ptype in enumerate(pair_types):
            rep_vals = [v for v in rep_corrs[ptype] if v is not None]
            if rep_vals:
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(rep_vals))
                ax.scatter(
                    x[i] + jitter, rep_vals, color="black", s=12,
                    alpha=0.4, zorder=5,
                )

        # N annotation
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
        f"Tetrachoric Correlation (Liability Threshold) [{scenario}]", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_joint_affection(all_stats, df_samples, output_path, scenario=""):
    """2x2 heatmap of joint affection status (trait1 x trait2)."""
    # Average proportions across reps
    keys = ["both", "trait1_only", "trait2_only", "neither"]
    avg_props = {}
    for k in keys:
        avg_props[k] = np.mean([s["joint_affection"]["proportions"][k] for s in all_stats])

    matrix = np.array([
        [avg_props["both"], avg_props["trait1_only"]],
        [avg_props["trait2_only"], avg_props["neither"]],
    ])

    avg_counts = {}
    for k in keys:
        avg_counts[k] = np.mean([s["joint_affection"]["counts"][k] for s in all_stats])

    labels = np.array([
        [f"{avg_props['both']:.2f}\n(n={avg_counts['both']:.0f})",
         f"{avg_props['trait1_only']:.2f}\n(n={avg_counts['trait1_only']:.0f})"],
        [f"{avg_props['trait2_only']:.2f}\n(n={avg_counts['trait2_only']:.0f})",
         f"{avg_props['neither']:.2f}\n(n={avg_counts['neither']:.0f})"],
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
    a1 = df_samples["affected1"].values.astype(bool)
    a2 = df_samples["affected2"].values.astype(bool)
    r_tet = tetrachoric_corr(a1, a2)
    r_label = f"r_tet = {r_tet:.3f}" if not np.isnan(r_tet) else "r_tet = N/A"

    ax.set_xlabel("Trait 1")
    ax.set_ylabel("Trait 2")
    ax.set_title(f"Joint Affection Status (Liability Threshold) [{scenario}]\n{r_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_joint(df_samples, output_path, scenario=""):
    """2x2 jointplot grid: Liability, A, C, E (trait1 vs trait2), colored by affected1."""
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
    fig.suptitle(f"Cross-Trait Correlations (Liability Threshold) [{scenario}]", fontsize=14, y=1.01)
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

        # Plot unaffected first, then affected on top
        for mask, color, alpha, label in [
            (~affected, "C0", 0.03, "Unaffected"),
            (affected, "C3", 0.15, "Affected (T1)"),
        ]:
            ax_joint.scatter(
                x[mask], y[mask], c=color, alpha=alpha, s=3, rasterized=True, label=label,
            )

        ax_marg_x.hist(x[~affected], bins=50, edgecolor="none", alpha=0.5, color="C0")
        ax_marg_x.hist(x[affected], bins=50, edgecolor="none", alpha=0.7, color="C3")
        ax_marg_y.hist(
            y[~affected], bins=50, orientation="horizontal", edgecolor="none", alpha=0.5, color="C0",
        )
        ax_marg_y.hist(
            y[affected], bins=50, orientation="horizontal", edgecolor="none", alpha=0.7, color="C3",
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


def main(stats_paths, sample_paths, output_dir, prevalence1, prevalence2):
    """Generate all threshold phenotype plots from pre-computed stats."""
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

    plot_prevalence_by_generation(
        all_stats, prevalence1, prevalence2,
        output_dir / "prevalence_by_generation.png", scenario,
    )
    plot_liability_violin(
        df_samples, all_stats,
        output_dir / "liability_threshold_violin.png", scenario,
    )
    plot_joint_affection(
        all_stats, df_samples, output_dir / "threshold_joint_affection.png", scenario,
    )
    plot_liability_joint(
        df_samples, output_dir / "threshold_liability_joint.png", scenario,
    )

    plot_tetrachoric(
        all_stats, output_dir / "threshold_tetrachoric.png", scenario,
    )

    print(f"Threshold plots saved to {output_dir}")


def cli():
    """Command-line interface for generating threshold plots."""
    parser = argparse.ArgumentParser(description="Plot threshold phenotype distributions")
    parser.add_argument("--stats", nargs="+", required=True, help="Stats YAML paths")
    parser.add_argument("--samples", nargs="+", required=True, help="Sample parquet paths")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prevalence1", type=float, required=True)
    parser.add_argument("--prevalence2", type=float, required=True)
    args = parser.parse_args()

    main(args.stats, args.samples, args.output_dir, args.prevalence1, args.prevalence2)
