"""
Plot validation results summarized across replicates per scenario.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

import pandas as pd

logger = logging.getLogger(__name__)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path


def stripplot(df: pd.DataFrame, ax: Axes, y: str, expected: str | float | None = None, expected_func: Callable[[pd.DataFrame], float] | None = None) -> None:
    """Stripplot of observed values with optional expected markers.

    Args:
        expected: column name for per-scenario expected values, or a fixed number.
        expected_func: callable(scenario_df) -> expected value.
    """
    scenarios = df["scenario"].unique()
    positions = {s: i for i, s in enumerate(scenarios)}

    sns.stripplot(data=df, x="scenario", y=y, ax=ax, alpha=0.7, color="C0", jitter=0.15)

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

    ax.set_xlabel("")
    if len(scenarios) > 4:
        ax.tick_params(axis="x", rotation=45)
    if len(scenarios) == 1:
        ax.set_xlim(-0.5, 0.5)
    ymin, ymax = ax.get_ylim()
    pad = max(0.02, (ymax - ymin) * 0.05)
    ax.set_ylim(ymin - pad, ymax + pad)


def save(fig: Figure, path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_width(n_scenarios: int, ncols: int = 1, per_scenario: float = 1.5, min_col: float = 3.0) -> float:
    """Scale figure width by scenario count and number of subplot columns."""
    return ncols * max(min_col, n_scenarios * per_scenario)


def plot_variance_components(df: pd.DataFrame, out: Path) -> None:
    n = df["scenario"].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(_fig_width(n, ncols=3), 10))
    for row, t in enumerate([1, 2]):
        for col, comp in enumerate(["A", "C", "E"]):
            ax = axes[row, col]
            stripplot(df, ax, f"variance_{comp}{t}", expected=f"{comp}{t}")
            ax.set_title(f"Trait {t}: {comp}{t}")
            ax.set_ylabel("Variance Proportion")
    save(fig, out / "variance_components.png")


def plot_twin_rate(df: pd.DataFrame, out: Path) -> None:
    n = df["scenario"].nunique()
    fig, ax = plt.subplots(figsize=(_fig_width(n), 5))
    stripplot(df, ax, "observed_twin_rate", expected="p_mztwin")
    ax.set_title("MZ Twin Rate: Observed vs Expected")
    ax.set_ylabel("Twin Rate")
    save(fig, out / "twin_rate.png")


def plot_A_correlations(df: pd.DataFrame, out: Path) -> None:
    panels = [
        ("mz_twin_A1_corr", 1.0, "MZ Twin A1 Correlation"),
        ("dz_sibling_A1_corr", 0.5, "DZ Sibling A1 Correlation"),
        ("half_sib_A1_corr", 0.25, "Half-Sibling A1 Correlation"),
        ("parent_offspring_A1_r2", 0.5, "Midparent-Offspring A1 R²"),
    ]
    n = df["scenario"].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(_fig_width(n, ncols=2), 10))
    for ax, (col, exp, title) in zip(axes.flat, panels):
        stripplot(df, ax, col, expected=exp)
        ax.axhline(y=exp, color="C1", linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / "correlations_A.png")


def plot_phenotype_correlations(df: pd.DataFrame, out: Path) -> None:
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
    n = df["scenario"].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(_fig_width(n, ncols=2), 10))
    for ax, (col, efn, title) in zip(axes.flat, panels):
        stripplot(df, ax, col, expected_func=efn)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / "correlations_phenotype.png")


def plot_heritability_estimates(df: pd.DataFrame, out: Path) -> None:
    panels = [
        ("falconer_h2_trait1", "A1", "Falconer h² Trait 1", "Heritability"),
        ("parent_offspring_liability1_slope", "A1", "Midparent-Offspring Liability1", "Slope"),
        ("falconer_h2_trait2", "A2", "Falconer h² Trait 2", "Heritability"),
        ("parent_offspring_liability2_slope", "A2", "Midparent-Offspring Liability2", "Slope"),
    ]
    n = df["scenario"].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(_fig_width(n, ncols=2), 10))
    for ax, (col, exp, title, ylabel) in zip(axes.flat, panels):
        stripplot(df, ax, col, expected=exp)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    save(fig, out / "heritability_estimates.png")


def plot_half_sib_proportions(df: pd.DataFrame, out: Path) -> None:
    n = df["scenario"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(_fig_width(n, ncols=2), 5))
    stripplot(df, axes[0], "half_sib_prop_observed", expected="half_sib_prop_expected")
    axes[0].set_title("Half-Sibling Pair Proportion")
    axes[0].set_ylabel("Proportion")

    stripplot(df, axes[1], "offspring_with_half_sib_observed")
    axes[1].set_title("Proportion of Offspring with Half-Siblings")
    axes[1].set_ylabel("Proportion")
    save(fig, out / "half_sib_proportions.png")


def plot_cross_trait_correlations(df: pd.DataFrame, out: Path) -> None:
    panels = [
        ("observed_rA", "rA", "Cross-Trait rA"),
        ("observed_rC", "rC", "Cross-Trait rC"),
        ("observed_rE", None, "Cross-Trait rE"),
    ]
    n = df["scenario"].nunique()
    fig, axes = plt.subplots(1, 3, figsize=(_fig_width(n, ncols=3), 5))
    for ax, (obs, exp, title) in zip(axes, panels):
        if exp:
            stripplot(df, ax, obs, expected=exp)
        else:
            stripplot(df, ax, obs)
            ax.axhline(y=0, color="C1", linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / "cross_trait_correlations.png")


def plot_family_size(df: pd.DataFrame, out: Path) -> None:
    n = df["scenario"].nunique()
    fig, ax = plt.subplots(figsize=(_fig_width(n), 5))
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
    if len(scenarios) > 4:
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
    else:
        ax.set_xticklabels(scenarios)
    if len(scenarios) == 1:
        ax.set_xlim(-0.5, 0.5)
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


def plot_summary_bias(df: pd.DataFrame, out: Path) -> None:
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
    n = dp["scenario"].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(_fig_width(n, ncols=3), 10))
    for ax, col in zip(axes.flat, panels):
        sns.stripplot(data=dp, x="scenario", y=col, ax=ax, alpha=0.7, jitter=0.15)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(col)
        ax.set_xlabel("")
        if n > 4:
            ax.tick_params(axis="x", rotation=45)
        if n == 1:
            ax.set_xlim(-0.5, 0.5)
    save(fig, out / "summary_bias.png")


def plot_runtime(df: pd.DataFrame, out: Path) -> None:
    sub = df.dropna(subset=["simulate_seconds"])
    if sub.empty:
        logger.warning("No simulate_seconds data; skipping runtime plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scenarios = sub["scenario"].unique()
    palette = sns.color_palette("Set2", len(scenarios))
    color_map = dict(zip(scenarios, palette))

    for scenario in scenarios:
        sdf = sub[sub["scenario"] == scenario]
        ax.scatter(
            sdf["N"], sdf["simulate_seconds"],
            color=color_map[scenario], label=scenario, alpha=0.7, s=40,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import LogLocator, ScalarFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=12))
        axis.set_major_formatter(ScalarFormatter())
        axis.get_major_formatter().set_scientific(False)
    ax.set_xlabel("Population Size (N)")
    ax.set_ylabel("Simulate Time (seconds)")
    ax.set_title("Simulation Runtime vs Population Size")
    ax.legend()
    save(fig, out / "runtime.png")


def plot_memory(df: pd.DataFrame, out: Path) -> None:
    sub = df.dropna(subset=["simulate_max_rss_mb"])
    if sub.empty:
        logger.warning("No simulate_max_rss_mb data; skipping memory plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scenarios = sub["scenario"].unique()
    palette = sns.color_palette("Set2", len(scenarios))
    color_map = dict(zip(scenarios, palette))

    for scenario in scenarios:
        sdf = sub[sub["scenario"] == scenario]
        ax.scatter(
            sdf["N"], sdf["simulate_max_rss_mb"],
            color=color_map[scenario], label=scenario, alpha=0.7, s=40,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import LogLocator, ScalarFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=12))
        axis.set_major_formatter(ScalarFormatter())
        axis.get_major_formatter().set_scientific(False)
    ax.set_xlabel("Population Size (N)")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Simulation Memory Usage vs Population Size")
    ax.legend()
    save(fig, out / "memory.png")


def main(tsv_path: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating validation plots in %s", output_dir)
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
    plot_runtime(df, output_dir)
    plot_memory(df, output_dir)


def cli() -> None:
    """Command-line interface for generating validation plots."""
    from sim_ace import setup_logging
    parser = argparse.ArgumentParser(description="Plot validation results")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG output")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING+ only")
    parser.add_argument("tsv", help="Validation summary TSV path")
    parser.add_argument("output_dir", help="Output directory")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=level)

    main(args.tsv, args.output_dir)
