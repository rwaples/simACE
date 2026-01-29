"""
Plot phenotype distributions aggregated across replicates for a scenario.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_phenotypes(parquet_paths):
    """Load and concatenate phenotype parquet files."""
    dfs = [pd.read_parquet(p) for p in parquet_paths]
    return pd.concat(dfs, ignore_index=True)


def plot_death_age_distribution(df, censor_age, output_path):
    """Plot mortality rate and cumulative mortality by decade."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute mortality rate per decade
    decade_edges = np.arange(0, censor_age + 10, 10)
    mortality_rates = []
    decade_labels = []
    for i in range(len(decade_edges) - 1):
        lo, hi = decade_edges[i], decade_edges[i + 1]
        if lo >= censor_age:
            break
        alive_at_start = len(df[df["death_age"] >= lo])
        died_in_decade = len(df[(df["death_age"] >= lo) & (df["death_age"] < hi) & (df["death_age"] < censor_age)])
        rate = died_in_decade / alive_at_start if alive_at_start > 0 else 0
        mortality_rates.append(rate)
        decade_labels.append(f"{int(lo)}-{int(hi - 1)}")

    mortality_rates = np.array(mortality_rates)

    # Left: mortality rate per decade
    axes[0].bar(decade_labels, mortality_rates, edgecolor="black", alpha=0.7)
    axes[0].set_title("Mortality Rate by Decade")
    axes[0].set_xlabel("Age Decade")
    axes[0].set_ylabel("Mortality Rate")
    axes[0].tick_params(axis="x", rotation=45)

    # Right: cumulative mortality per decade with survival annotations
    survival = np.cumprod(1 - mortality_rates)
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trait_phenotype(df, trait_num, output_path):
    """Plot phenotype distributions for a trait (affected vs censored)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    affected_col = f"affected{trait_num}"
    t_col = f"t_observed{trait_num}"

    death_censored_col = f"death_censored{trait_num}"

    affected = df[df[affected_col] == True]
    death_censored = df[(df[affected_col] == False) & (df[death_censored_col] == True)]

    axes[0].hist(affected[t_col].dropna(), bins=50, density=True, edgecolor="black", alpha=0.7, color="C3")
    axes[0].set_title(f"Trait {trait_num}: Age at Onset (affected)")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Density")

    axes[1].hist(death_censored[t_col].dropna(), bins=50, density=True, edgecolor="black", alpha=0.7, color="C0")
    axes[1].set_title(f"Trait {trait_num}: Age at Death (death-censored, unaffected)")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trait_regression(df, trait_num, output_path):
    """Plot liability vs age at onset for affected individuals as a jointplot."""
    affected_col = f"affected{trait_num}"
    t_col = f"t_observed{trait_num}"
    liability_col = f"liability{trait_num}"

    affected = df[df[affected_col] == True].dropna(subset=[liability_col, t_col])
    x = affected[liability_col].values
    y = affected[t_col].values

    # Compute regression for annotation
    coeffs = np.polyfit(x, y, 1)
    ss_res = np.sum((y - np.polyval(coeffs, x)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    g = sns.jointplot(
        x=affected[liability_col],
        y=affected[t_col],
        kind="scatter",
        joint_kws={"alpha": 0.1, "s": 5},
        marginal_kws={"bins": 50},
        height=8,
    )
    # Overlay regression line
    x_line = np.array([x.min(), x.max()])
    g.ax_joint.plot(x_line, np.polyval(coeffs, x_line), color="C3", linewidth=2)
    g.ax_joint.text(
        0.05, 0.95, f"R² = {r2:.4f}", transform=g.ax_joint.transAxes, va="top", fontsize=12
    )

    g.ax_joint.set_xlabel("Liability")
    g.ax_joint.set_ylabel("Age at Onset")
    g.figure.suptitle(f"Trait {trait_num}: Liability vs Age at Onset (affected)", y=1.02)

    g.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_joint(df, output_path):
    """Jointplot of liability1 vs liability2."""
    g = sns.jointplot(
        x=df["liability1"],
        y=df["liability2"],
        kind="scatter",
        joint_kws={"alpha": 0.1, "s": 5},
        marginal_kws={"bins": 50},
        height=8,
    )
    r = df[["liability1", "liability2"]].corr().iloc[0, 1]
    g.ax_joint.text(
        0.05, 0.95, f"r = {r:.4f}", transform=g.ax_joint.transAxes, va="top", fontsize=12
    )
    g.ax_joint.set_xlabel("Liability (Trait 1)")
    g.ax_joint.set_ylabel("Liability (Trait 2)")
    g.figure.suptitle("Liability: Trait 1 vs Trait 2", y=1.02)
    g.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_liability_violin(df, output_path):
    """Split violin plot of liability by trait, split on affected status."""
    violin_data = pd.concat(
        [
            pd.DataFrame({
                "Trait": "Trait 1",
                "Liability": df["liability1"],
                "Affected": df["affected1"],
            }),
            pd.DataFrame({
                "Trait": "Trait 2",
                "Liability": df["liability2"],
                "Affected": df["affected2"],
            }),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=violin_data, x="Trait", y="Liability", hue="Affected",
        split=True, ax=ax,
    )
    ax.set_title("Liability Distribution by Trait and Affected Status")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(parquet_paths, censor_age, output_dir):
    """Generate all phenotype plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = load_phenotypes(parquet_paths)

    plot_death_age_distribution(df, censor_age, output_dir / "death_age_distribution.png")
    plot_trait_phenotype(df, 1, output_dir / "phenotype_trait1.png")
    plot_trait_phenotype(df, 2, output_dir / "phenotype_trait2.png")
    plot_trait_regression(df, 1, output_dir / "liability_regression_trait1.png")
    plot_trait_regression(df, 2, output_dir / "liability_regression_trait2.png")
    plot_liability_joint(df, output_dir / "liability_joint.png")
    plot_liability_violin(df, output_dir / "liability_violin.png")

    print(f"Phenotype plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    try:
        parquet_paths = snakemake.input.phenotypes
        censor_age = snakemake.params.censor_age
        output_dir = Path(snakemake.output[0]).parent
    except NameError:
        if len(sys.argv) >= 4:
            censor_age = float(sys.argv[1])
            output_dir = sys.argv[2]
            parquet_paths = sys.argv[3:]
        else:
            print("Usage: plot_phenotype.py <censor_age> <output_dir> <phenotype1.parquet> ...")
            sys.exit(1)

    main(parquet_paths, censor_age, output_dir)
