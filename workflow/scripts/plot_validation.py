"""
Plot validation results summarized across replicates per scenario.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_validation_data(tsv_path):
    """Load validation summary TSV."""
    df = pd.read_csv(tsv_path, sep="\t")
    df["E_expected"] = 1 - df["A"] - df["C"]
    return df


def get_scenario_positions(df):
    """Get x positions for each scenario."""
    scenarios = df["scenario"].unique()
    return scenarios, {s: i for i, s in enumerate(scenarios)}


def set_yaxis_padding(ax, padding=0.05):
    """Extend y-axis limits by at least `padding` units beyond min/max."""
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - padding, ymax + padding)


def plot_expected_markers(
    ax, df, scenarios, positions, expected_col=None, expected_func=None
):
    """Plot expected values as horizontal line markers per scenario.

    Use expected_col for a column name, or expected_func for a callable that
    takes a scenario's dataframe subset and returns the expected value.
    """
    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario]
        if expected_col:
            exp_val = scenario_df[expected_col].iloc[0]
        else:
            exp_val = expected_func(scenario_df)
        ax.scatter(
            positions[scenario],
            exp_val,
            marker="_",
            s=200,
            linewidths=3,
            color="C1",
            zorder=10,
        )


def setup_ax(ax, title, ylabel, xlabel="Scenario"):
    """Apply common axis setup."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    set_yaxis_padding(ax)


def plot_variance_components(df, output_dir):
    """Plot observed vs expected variance components (A, C, E) per scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    scenarios, positions = get_scenario_positions(df)

    components = [
        ("variance_A", "A", "Additive Genetic (A)"),
        ("variance_C", "C", "Common Environment (C)"),
        ("variance_E", "E_expected", "Unique Environment (E)"),
    ]

    for ax, (obs_col, exp_col, title) in zip(axes, components):
        sns.stripplot(data=df, x="scenario", y=obs_col, ax=ax, alpha=0.7, color="C0")
        plot_expected_markers(ax, df, scenarios, positions, expected_col=exp_col)
        setup_ax(ax, title, "Variance Proportion")

    plt.tight_layout()
    plt.savefig(output_dir / "variance_components.png", dpi=150)
    plt.close()


def plot_twin_rate(df, output_dir):
    """Plot observed vs expected MZ twin rate per scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios, positions = get_scenario_positions(df)

    sns.stripplot(
        data=df, x="scenario", y="observed_twin_rate", ax=ax, alpha=0.7, color="C0"
    )
    plot_expected_markers(ax, df, scenarios, positions, expected_col="p_mztwin")
    setup_ax(ax, "MZ Twin Rate: Observed vs Expected", "Twin Rate")

    plt.tight_layout()
    plt.savefig(output_dir / "twin_rate.png", dpi=150)
    plt.close()


def plot_A_correlations(df, output_dir):
    """Plot A (genetic) correlation estimates per scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        (axes[0, 0], "mz_twin_A_corr", 1.0, "MZ Twin A Correlation", "Correlation"),
        (
            axes[0, 1],
            "dz_sibling_A_corr",
            0.5,
            "DZ Sibling A Correlation",
            "Correlation",
        ),
        (
            axes[1, 0],
            "half_sib_A_corr",
            0.25,
            "Half-Sibling A Correlation",
            "Correlation",
        ),
        (axes[1, 1], "parent_offspring_A_r2", 0.5, "Midparent-Offspring A R²", "R²"),
    ]

    for ax, col, expected, title, ylabel in panels:
        sns.stripplot(data=df, x="scenario", y=col, ax=ax, alpha=0.7, color="C0")
        ax.axhline(y=expected, color="C1", linestyle="--", alpha=0.7)
        setup_ax(ax, title, ylabel)

    plt.tight_layout()
    plt.savefig(output_dir / "correlations_A.png", dpi=150)
    plt.close()


def plot_phenotype_correlations(df, output_dir):
    """Plot phenotype correlation estimates per scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    scenarios, positions = get_scenario_positions(df)

    panels = [
        (
            axes[0, 0],
            "mz_twin_pheno_corr",
            lambda d: d["A"].iloc[0] + d["C"].iloc[0],
            "MZ Twin Phenotype Correlation",
            "Expected (A+C)",
        ),
        (
            axes[0, 1],
            "dz_sibling_pheno_corr",
            lambda d: 0.5 * d["A"].iloc[0] + d["C"].iloc[0],
            "DZ Sibling Phenotype Correlation",
            "Expected (0.5A+C)",
        ),
        (
            axes[1, 0],
            "half_sib_pheno_corr",
            lambda d: 0.25 * d["A"].iloc[0],
            "Half-Sibling Phenotype Correlation",
            "Expected (0.25A)",
        ),
        (
            axes[1, 1],
            "parent_offspring_pheno_slope",
            lambda d: d["A"].iloc[0],
            "Midparent-Offspring Phenotype Correlation",
            "Expected (A)",
        ),
    ]

    for ax, col, exp_func, title, label in panels:
        sns.stripplot(data=df, x="scenario", y=col, ax=ax, alpha=0.7, color="C0")
        plot_expected_markers(ax, df, scenarios, positions, expected_func=exp_func)
        ax.scatter([], [], marker="_", s=200, linewidths=3, color="C1", label=label)
        ax.legend(loc="best")
        setup_ax(ax, title, "Correlation")

    plt.tight_layout()
    plt.savefig(output_dir / "correlations_phenotype.png", dpi=150)
    plt.close()


def plot_heritability_estimates(df, output_dir):
    """Plot heritability estimates vs expected per scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scenarios, positions = get_scenario_positions(df)

    panels = [
        (axes[0], "falconer_h2", "Falconer Heritability Estimate", "Heritability"),
        (
            axes[1],
            "parent_offspring_pheno_slope",
            "Midparent-Offspring Phenotype Regression",
            "Slope",
        ),
    ]

    for ax, col, title, ylabel in panels:
        sns.stripplot(data=df, x="scenario", y=col, ax=ax, alpha=0.7, color="C0")
        plot_expected_markers(ax, df, scenarios, positions, expected_col="A")
        setup_ax(ax, title, ylabel)

    plt.tight_layout()
    plt.savefig(output_dir / "heritability_estimates.png", dpi=150)
    plt.close()


def plot_half_sib_proportions(df, output_dir):
    """Plot observed vs expected half-sibling proportions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scenarios, positions = get_scenario_positions(df)

    # Half-sib pair proportion (with expected)
    ax = axes[0]
    sns.stripplot(
        data=df, x="scenario", y="half_sib_prop_observed", ax=ax, alpha=0.7, color="C0"
    )
    plot_expected_markers(
        ax, df, scenarios, positions, expected_col="half_sib_prop_expected"
    )
    setup_ax(ax, "Half-Sibling Pair Proportion", "Proportion")

    # Offspring with half-sib (no expected)
    ax = axes[1]
    sns.stripplot(
        data=df,
        x="scenario",
        y="offspring_with_half_sib_observed",
        ax=ax,
        alpha=0.7,
        color="C0",
    )
    setup_ax(ax, "Proportion of Offspring with Half-Siblings", "Proportion")

    plt.tight_layout()
    plt.savefig(output_dir / "half_sib_proportions.png", dpi=150)
    plt.close()


def plot_summary_stripplot(df, output_dir):
    """Create a summary stripplot showing bias (observed - expected) per scenario."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    df_plot = df.copy()
    df_plot["A Bias"] = df_plot["variance_A"] - df_plot["A"]
    df_plot["C Bias"] = df_plot["variance_C"] - df_plot["C"]
    df_plot["E Bias"] = df_plot["variance_E"] - df_plot["E_expected"]
    df_plot["Twin Rate Bias"] = df_plot["observed_twin_rate"] - df_plot["p_mztwin"]
    df_plot["DZ A Corr Bias"] = df_plot["dz_sibling_A_corr"] - 0.5
    df_plot["Half-sib A Corr Bias"] = df_plot["half_sib_A_corr"] - 0.25

    panels = [
        (axes[0, 0], "A Bias", "Variance A Bias (Observed - Expected)"),
        (axes[0, 1], "C Bias", "Variance C Bias (Observed - Expected)"),
        (axes[0, 2], "E Bias", "Variance E Bias (Observed - Expected)"),
        (axes[1, 0], "Twin Rate Bias", "MZ Twin Rate Bias (Observed - Expected)"),
        (axes[1, 1], "DZ A Corr Bias", "DZ Sibling A Correlation Bias (vs 0.5)"),
        (
            axes[1, 2],
            "Half-sib A Corr Bias",
            "Half-Sibling A Correlation Bias (vs 0.25)",
        ),
    ]

    for ax, col, title in panels:
        sns.stripplot(data=df_plot, x="scenario", y=col, ax=ax, alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        setup_ax(ax, title, col)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_bias.png", dpi=150)
    plt.close()


def main(tsv_path, output_dir):
    """Generate all validation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="Set2")
    df = load_validation_data(tsv_path)

    plot_variance_components(df, output_dir)
    plot_twin_rate(df, output_dir)
    plot_A_correlations(df, output_dir)
    plot_phenotype_correlations(df, output_dir)
    plot_heritability_estimates(df, output_dir)
    plot_half_sib_proportions(df, output_dir)
    plot_summary_stripplot(df, output_dir)

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    try:
        tsv_path = snakemake.input.tsv
        output_dir = Path(snakemake.output[0]).parent
    except NameError:
        if len(sys.argv) >= 3:
            tsv_path = sys.argv[1]
            output_dir = sys.argv[2]
        else:
            tsv_path = "results/validation_summary.tsv"
            output_dir = "results/plots"

    main(tsv_path, output_dir)
