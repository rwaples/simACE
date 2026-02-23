"""
Plot liability threshold phenotype distributions from pre-computed stats.

Reads threshold_stats.yaml and threshold_samples.parquet files (one per rep)
produced by compute_threshold_stats.py.
"""

from __future__ import annotations

import argparse
from typing import Any

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

import logging
logger = logging.getLogger(__name__)

MAX_PLOT_POINTS = 100_000


def plot_prevalence_by_generation(all_stats: list[dict[str, Any]], prevalence1: float | dict[int, float], prevalence2: float | dict[int, float], output_path: str | Path, scenario: str = "") -> None:
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

        # Expected prevalence reference
        if isinstance(expected, dict):
            # Per-generation markers at each generation's expected value
            for i, gen in enumerate(gens):
                gen_prev = expected.get(int(gen))
                if gen_prev is not None:
                    ax.hlines(
                        gen_prev, i + offset - 0.17, i + offset + 0.17,
                        colors=color, linestyles="dashed", linewidth=2, alpha=0.7,
                    )
        else:
            # Single horizontal line across all generations
            ax.hlines(
                expected, -0.5, n_gens - 0.5, colors=color,
                linestyles="dashed", linewidth=2, alpha=0.7,
            )

    ax.set_xticks(np.arange(n_gens))
    gen_labels = []
    for i, g in enumerate(gens):
        label = f"Gen {g}"
        if i == 0:
            label += "\n(oldest)"
        elif i == n_gens - 1:
            label += "\n(youngest)"
        gen_labels.append(label)
    ax.set_xticklabels(gen_labels)
    ax.set_ylabel("Prevalence")
    # Adapt y-limit to expected prevalence values
    max_expected = 0.0
    for expected in [prevalence1, prevalence2]:
        if isinstance(expected, dict):
            max_expected = max(max_expected, max(expected.values()))
        else:
            max_expected = max(max_expected, expected)
    ax.set_ylim(0, max(max_expected * 1.5, ax.get_ylim()[1]))
    ax.set_title(f"Prevalence by Generation (Liability Threshold) [{scenario}]", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_liability_violin(df_samples: pd.DataFrame, all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str = "") -> None:
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


def plot_liability_violin_by_generation(df_samples: pd.DataFrame, all_stats: list[dict[str, Any]], prevalence1: float | dict[int, float], prevalence2: float | dict[int, float], output_path: str | Path, scenario: str = "") -> None:
    """Split violin of liability by affected status, one column per generation."""
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
        prevalence = prevalence1 if trait_num == 1 else prevalence2
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
            prev = prevalence[int(gen)] if isinstance(prevalence, dict) else prevalence
            obs_prev = df_gen[aff_col].mean() if len(df_gen) else float("nan")
            ax.set_xlabel(f"prev: {obs_prev:.1%} (exp {prev:.1%})", fontsize=8)

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
        f"Liability by Affected Status per Generation (Threshold) [{scenario}]",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tetrachoric(all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str) -> None:
    """Tetrachoric correlations by relationship type with liability correlation lines."""
    pair_types = ["MZ twin", "Full sib", "Mother-offspring", "Father-offspring", "Maternal half sib", "Paternal half sib", "1st cousin"]
    pair_colors = {"MZ twin": "C0", "Full sib": "C1", "Mother-offspring": "C3", "Father-offspring": "C5", "Maternal half sib": "C2", "Paternal half sib": "C6", "1st cousin": "C4"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col_idx, trait_num in enumerate([1, 2]):
        ax = axes[col_idx]
        key = f"trait{trait_num}"

        # Build long-format data for violin plot
        rows = []
        total_pairs = {}
        for ptype in pair_types:
            n_total = 0
            for s in all_stats:
                entry = s["tetrachoric"][key].get(ptype, {})
                r = entry.get("r")
                n_p = entry.get("n_pairs", 0)
                n_total += n_p
                if r is not None:
                    rows.append({"pair_type": ptype, "r": r})
            total_pairs[ptype] = n_total

        df_plot = pd.DataFrame(rows)

        if not df_plot.empty:
            sns.violinplot(
                data=df_plot, x="pair_type", y="r", hue="pair_type", ax=ax,
                order=pair_types, palette=pair_colors, legend=False,
                inner=None, cut=0, alpha=0.6, zorder=3,
            )

            # Per-rep dots
            x = np.arange(len(pair_types))
            for i, ptype in enumerate(pair_types):
                rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                if len(rep_vals):
                    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rep_vals))
                    ax.scatter(
                        i + jitter, rep_vals, color="black", s=15,
                        alpha=0.6, zorder=5,
                    )

            # N annotation
            n_reps = len(all_stats)
            for i, ptype in enumerate(pair_types):
                rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                if len(rep_vals):
                    top = rep_vals.max()
                    ax.text(
                        i, top + 0.03,
                        f"N={total_pairs[ptype] // n_reps}", ha="center", va="bottom", fontsize=8,
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

        ax.set_xticks(range(len(pair_types)))
        ax.set_xticklabels(pair_types, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Tetrachoric Correlation")
        ax.set_title(f"Trait {trait_num}")
        ax.set_ylim(-0.1, 1.1)

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor="C7", alpha=0.6, label="Tetrachoric r (violin)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                   markersize=6, alpha=0.6, label="Per-replicate r"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Liability r"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.suptitle(
        f"Tetrachoric Correlation (Liability Threshold) [{scenario}]", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_joint_affection(all_stats: list[dict[str, Any]], df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "") -> None:
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
    ax.set_title(f"Joint Affected Status (Liability Threshold) [{scenario}]\n{r_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_joint(df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "") -> None:
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
        bins_x = np.linspace(x.min(), x.max(), 51)
        bins_y = np.linspace(y.min(), y.max(), 51)

        # Plot unaffected first, then affected on top
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


def main(stats_paths: list[str], sample_paths: list[str], output_dir: str, prevalence1: float | dict[int, float], prevalence2: float | dict[int, float]) -> None:
    """Generate all threshold phenotype plots from pre-computed stats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = out_dir.parent.name
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
        out_dir / "prevalence_by_generation.png", scenario,
    )
    plot_liability_violin(
        df_samples, all_stats,
        out_dir / "liability_violin.threshold.png", scenario,
    )
    plot_liability_violin_by_generation(
        df_samples, all_stats, prevalence1, prevalence2,
        out_dir / "liability_violin.threshold.by_generation.png", scenario,
    )
    plot_joint_affection(
        all_stats, df_samples, out_dir / "joint_affected.threshold.png", scenario,
    )
    plot_liability_joint(
        df_samples, out_dir / "cross_trait.threshold.png", scenario,
    )

    plot_tetrachoric(
        all_stats, out_dir / "tetrachoric.threshold.png", scenario,
    )

    logger.info("Threshold plots saved to %s", out_dir)


def cli() -> None:
    """Command-line interface for generating threshold plots."""
    import json
    from sim_ace import setup_logging
    parser = argparse.ArgumentParser(description="Plot threshold phenotype distributions")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG output")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING+ only")
    parser.add_argument("--stats", nargs="+", required=True, help="Stats YAML paths")
    parser.add_argument("--samples", nargs="+", required=True, help="Sample parquet paths")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prevalence1", type=float, default=None, help="Disease prevalence for trait 1")
    parser.add_argument("--prevalence2", type=float, default=None, help="Disease prevalence for trait 2")
    parser.add_argument("--prevalence1-by-gen", type=str, default=None,
                        help='Per-gen prevalence for trait 1 as JSON, e.g. \'{"0":0.05,"1":0.10}\'')
    parser.add_argument("--prevalence2-by-gen", type=str, default=None,
                        help='Per-gen prevalence for trait 2 as JSON, e.g. \'{"0":0.05,"1":0.10}\'')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=level)

    def _resolve(scalar, by_gen_json):
        if by_gen_json is not None:
            return {int(k): float(v) for k, v in json.loads(by_gen_json).items()}
        if scalar is not None:
            return scalar
        parser.error("Either --prevalenceN or --prevalenceN-by-gen is required")

    p1 = _resolve(args.prevalence1, args.prevalence1_by_gen)
    p2 = _resolve(args.prevalence2, args.prevalence2_by_gen)

    main(args.stats, args.samples, args.output_dir, p1, p2)
