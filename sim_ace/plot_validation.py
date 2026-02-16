"""
Plot validation results summarized across replicates per scenario.
"""

import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def stripplot(df, ax, y, expected=None, expected_func=None):
    """Stripplot of observed values with optional expected markers.

    Args:
        expected: column name for per-scenario expected values, or a fixed number.
        expected_func: callable(scenario_df) -> expected value.
    """
    scenarios = df["scenario"].unique()
    positions = {s: i for i, s in enumerate(scenarios)}

    sns.stripplot(data=df, x="scenario", y=y, ax=ax, alpha=0.7, color="C0")

    if expected_func is not None or expected is not None:
        for scenario in scenarios:
            sdf = df[df["scenario"] == scenario]
            if expected_func:
                val = expected_func(sdf)
            elif isinstance(expected, str):
                val = sdf[expected].iloc[0]
            else:
                val = expected
            ax.scatter(
                positions[scenario], val,
                marker="_", s=200, linewidths=3, color="C1", zorder=10,
            )

    ax.tick_params(axis="x", rotation=45)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.05, ymax + 0.05)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variance_components(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, t in enumerate([1, 2]):
        for col, comp in enumerate(["A", "C", "E"]):
            ax = axes[row, col]
            stripplot(df, ax, f"variance_{comp}{t}", expected=f"{comp}{t}")
            ax.set_title(f"Trait {t}: {comp}{t}")
            ax.set_ylabel("Variance Proportion")
    save(fig, out / "variance_components.png")


def plot_twin_rate(df, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    stripplot(df, ax, "observed_twin_rate", expected="p_mztwin")
    ax.set_title("MZ Twin Rate: Observed vs Expected")
    ax.set_ylabel("Twin Rate")
    save(fig, out / "twin_rate.png")


def plot_A_correlations(df, out):
    panels = [
        ("mz_twin_A1_corr", 1.0, "MZ Twin A1 Correlation"),
        ("dz_sibling_A1_corr", 0.5, "DZ Sibling A1 Correlation"),
        ("half_sib_A1_corr", 0.25, "Half-Sibling A1 Correlation"),
        ("parent_offspring_A1_r2", 0.5, "Midparent-Offspring A1 R²"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (col, exp, title) in zip(axes.flat, panels):
        stripplot(df, ax, col, expected=exp)
        ax.axhline(y=exp, color="C1", linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / "correlations_A.png")


def plot_phenotype_correlations(df, out):
    panels = [
        ("mz_twin_liability1_corr", lambda d: d["A1"].iloc[0] + d["C1"].iloc[0],
         "MZ Twin Liability1 Corr"),
        ("dz_sibling_liability1_corr", lambda d: 0.5 * d["A1"].iloc[0] + d["C1"].iloc[0],
         "DZ Sibling Liability1 Corr"),
        ("half_sib_liability1_corr", lambda d: 0.25 * d["A1"].iloc[0],
         "Half-Sib Liability1 Corr"),
        ("parent_offspring_liability1_slope", lambda d: d["A1"].iloc[0],
         "Midparent-Offspring Liability1 Slope"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (col, efn, title) in zip(axes.flat, panels):
        stripplot(df, ax, col, expected_func=efn)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / "correlations_phenotype.png")


def plot_heritability_estimates(df, out):
    panels = [
        ("falconer_h2_trait1", "A1", "Falconer h² Trait 1", "Heritability"),
        ("parent_offspring_liability1_slope", "A1", "Midparent-Offspring Liability1", "Slope"),
        ("falconer_h2_trait2", "A2", "Falconer h² Trait 2", "Heritability"),
        ("parent_offspring_liability2_slope", "A2", "Midparent-Offspring Liability2", "Slope"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (col, exp, title, ylabel) in zip(axes.flat, panels):
        stripplot(df, ax, col, expected=exp)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    save(fig, out / "heritability_estimates.png")


def plot_half_sib_proportions(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stripplot(df, axes[0], "half_sib_prop_observed", expected="half_sib_prop_expected")
    axes[0].set_title("Half-Sibling Pair Proportion")
    axes[0].set_ylabel("Proportion")

    stripplot(df, axes[1], "offspring_with_half_sib_observed")
    axes[1].set_title("Proportion of Offspring with Half-Siblings")
    axes[1].set_ylabel("Proportion")
    save(fig, out / "half_sib_proportions.png")


def plot_cross_trait_correlations(df, out):
    panels = [
        ("observed_rA", "rA", "Cross-Trait rA"),
        ("observed_rC", "rC", "Cross-Trait rC"),
        ("observed_rE", None, "Cross-Trait rE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (obs, exp, title) in zip(axes, panels):
        if exp:
            stripplot(df, ax, obs, expected=exp)
        else:
            stripplot(df, ax, obs)
            ax.axhline(y=0, color="C1", linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / "cross_trait_correlations.png")


def plot_family_size(df, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = df["scenario"].unique()
    positions = {s: i for i, s in enumerate(scenarios)}
    width = 0.3

    for scenario in scenarios:
        sdf = df[df["scenario"] == scenario]
        x = positions[scenario]
        ax.scatter(
            [x - width / 2] * len(sdf), sdf["mother_mean_offspring"],
            color="C0", alpha=0.7, s=30, zorder=5,
        )
        ax.scatter(
            [x + width / 2] * len(sdf), sdf["father_mean_offspring"],
            color="C3", alpha=0.7, s=30, zorder=5,
        )
        # Expected fam_size marker
        expected = sdf["fam_size"].iloc[0]
        ax.scatter(
            x, expected, marker="_", s=200, linewidths=3, color="C1", zorder=10,
        )

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylabel("Mean Offspring per Parent")
    ax.set_title("Family Size: Mean Offspring per Mother and Father (parents with children only)")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=8, label="Mother"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=8, label="Father"),
        Line2D([0], [0], marker="_", color="C1", markersize=12, linewidth=3, label="Parametric Poisson family size"),
    ]
    ax.legend(handles=legend, loc="lower left")
    save(fig, out / "family_size.png")


def plot_summary_bias(df, out):
    dp = df.copy()
    dp["A1 Bias"] = dp["variance_A1"] - dp["A1"]
    dp["C1 Bias"] = dp["variance_C1"] - dp["C1"]
    dp["E1 Bias"] = dp["variance_E1"] - dp["E1"]
    dp["Twin Rate Bias"] = dp["observed_twin_rate"] - dp["p_mztwin"]
    dp["DZ A1 Corr Bias"] = dp["dz_sibling_A1_corr"] - 0.5
    dp["Half-sib A1 Corr Bias"] = dp["half_sib_A1_corr"] - 0.25

    panels = [
        "A1 Bias", "C1 Bias", "E1 Bias",
        "Twin Rate Bias", "DZ A1 Corr Bias", "Half-sib A1 Corr Bias",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, col in zip(axes.flat, panels):
        sns.stripplot(data=dp, x="scenario", y=col, ax=ax, alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    save(fig, out / "summary_bias.png")


def main(tsv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")
    df = pd.read_csv(tsv_path, sep="\t")

    plot_variance_components(df, output_dir)
    plot_twin_rate(df, output_dir)
    plot_A_correlations(df, output_dir)
    plot_phenotype_correlations(df, output_dir)
    plot_heritability_estimates(df, output_dir)
    plot_half_sib_proportions(df, output_dir)
    plot_cross_trait_correlations(df, output_dir)
    plot_family_size(df, output_dir)
    plot_summary_bias(df, output_dir)


def cli():
    """Command-line interface for generating validation plots."""
    parser = argparse.ArgumentParser(description="Plot validation results")
    parser.add_argument("tsv", help="Validation summary TSV path")
    parser.add_argument("output_dir", help="Output directory")
    args = parser.parse_args()

    main(args.tsv, args.output_dir)
