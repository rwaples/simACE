"""Plot validation results summarized across replicates per scenario."""

from __future__ import annotations

__all__ = [
    "plot_A_correlations",
    "plot_consanguineous_matings",
    "plot_cross_trait_correlations",
    "plot_family_size",
    "plot_half_sib_proportions",
    "plot_heritability_estimates",
    "plot_memory",
    "plot_phenotype_correlations",
    "plot_runtime",
    "plot_summary_bias",
    "plot_twin_rate",
    "plot_variance_components",
    "save",
    "stripplot",
]

import argparse
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def stripplot(
    df: pd.DataFrame,
    ax: Axes,
    y: str,
    expected: str | float | None = None,
    expected_func: Callable[[pd.DataFrame], float] | None = None,
) -> None:
    """Stripplot of observed values with optional expected markers.

    Args:
        df: Gathered metrics DataFrame with ``scenario`` column.
        ax: Matplotlib axes to plot on.
        y: Column name for the observed metric to plot.
        expected: Column name for per-scenario expected values, or a fixed number.
        expected_func: Callable(scenario_df) returning expected value.
    """
    scenarios = df["scenario"].unique()
    positions = {s: i for i, s in enumerate(scenarios)}

    # Guard against all-NaN y column (metric not computed)
    if df[y].isna().all():
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="0.5")
        ax.set_ylabel(y)
        return

    sns.stripplot(data=df, x="scenario", y=y, ax=ax, alpha=0.9, color="C0", jitter=0.15)

    if expected_func is not None or expected is not None:
        for scenario in scenarios:
            sdf = df[df["scenario"] == scenario]
            if expected_func:
                val = expected_func(sdf)
            elif isinstance(expected, str):
                val = sdf[expected].iloc[0]
            else:
                assert expected is not None
                val = expected
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = None
            if val is not None and np.isfinite(val):
                ax.scatter(
                    positions[scenario],
                    val,
                    marker="_",
                    s=200,
                    linewidths=3,
                    color="C1",
                    zorder=10,
                )

    ax.set_xlabel("")
    _long = max((len(str(s)) for s in scenarios), default=0) > 12
    if len(scenarios) > 3 or (len(scenarios) > 1 and _long):
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    if len(scenarios) == 1:
        ax.set_xlim(-0.5, 0.5)

    # Tight y-axis padding based on actual data range
    data_vals = df[y].dropna().values
    all_vals = list(data_vals)
    if expected_func is not None:
        for scenario in scenarios:
            sdf = df[df["scenario"] == scenario]
            all_vals.append(expected_func(sdf))
    elif expected is not None:
        if isinstance(expected, str):
            all_vals.extend(df[expected].dropna().values)
        else:
            all_vals.append(expected)
    # Filter out non-numeric values (e.g. per-generation dict strings)
    numeric_vals = []
    for v in all_vals:
        try:
            numeric_vals.append(float(v))
        except (TypeError, ValueError):
            pass
    all_vals = np.array(numeric_vals, dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) > 0:
        lo, hi = float(all_vals.min()), float(all_vals.max())
        span = hi - lo
        pad = max(span * 0.15, max(0.002, abs(lo + hi) / 2 * 0.01))
        ax.set_ylim(lo - pad, hi + pad)


def save(fig: Figure, path: str | Path) -> None:
    """Save figure to disk and close it."""
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _figsize(nrows: int = 1, ncols: int = 1) -> tuple[float, float]:
    """Fixed figure size tuned for landscape-letter atlas pages.

    Single-column plots are narrower to avoid stretching; multi-column
    plots fill the available ~10.5 × 6.5-inch image area.
    """
    width = 10.0 if ncols >= 2 else 6.0
    height = 6.0 if nrows >= 2 else 4.0
    return (width, height)


def plot_variance_components(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot observed vs expected A, C, E variance components per trait."""
    fig, axes = plt.subplots(2, 3, figsize=_figsize(nrows=2, ncols=3))
    for row, t in enumerate([1, 2]):
        for col, comp in enumerate(["A", "C", "E"]):
            ax = axes[row, col]
            stripplot(df, ax, f"variance_{comp}{t}", expected=f"{comp}{t}")
            ax.set_title(f"Trait {t}: {comp}{t}")
            ax.set_ylabel("Variance Proportion")
    save(fig, out / f"variance_components.{ext}")


def plot_twin_rate(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot observed MZ twin rate vs expected across scenarios."""
    fig, ax = plt.subplots(figsize=_figsize())
    stripplot(df, ax, "observed_twin_rate", expected="p_mztwin")
    ax.set_title("MZ Twin Rate: Observed vs Expected")
    ax.set_ylabel("Twin Rate")
    save(fig, out / f"twin_rate.{ext}")


def plot_A_correlations(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot MZ twin and full-sib additive genetic correlations."""
    panels = [
        ("mz_twin_A1_corr", 1.0, "MZ Twin A1 Correlation"),
        ("dz_sibling_A1_corr", 0.5, "DZ Sibling A1 Correlation"),
        ("half_sib_A1_corr", 0.25, "Half-Sibling A1 Correlation"),
        ("parent_offspring_A1_r2", 0.5, "Midparent-Offspring A1 R²"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=_figsize(nrows=2, ncols=2))
    for ax, (col, exp, title) in zip(axes.flat, panels, strict=True):
        stripplot(df, ax, col, expected=exp)
        ax.axhline(y=exp, color="C1", linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / f"correlations_A.{ext}")


def plot_phenotype_correlations(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot MZ twin and full-sib liability correlations vs expected."""
    panels = [
        ("mz_twin_liability1_corr", lambda d: d["A1"].iloc[0] + d["C1"].iloc[0], "MZ Twin Liability1 Corr"),
        ("dz_sibling_liability1_corr", lambda d: 0.5 * d["A1"].iloc[0] + d["C1"].iloc[0], "DZ Sibling Liability1 Corr"),
        ("half_sib_liability1_corr", lambda d: 0.25 * d["A1"].iloc[0], "Half-Sib Liability1 Corr"),
        ("parent_offspring_liability1_slope", lambda d: d["A1"].iloc[0], "Midparent-Offspring Liability1 Slope"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=_figsize(nrows=2, ncols=2))
    for ax, (col, efn, title) in zip(axes.flat, panels, strict=True):
        stripplot(df, ax, col, expected_func=efn)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / f"correlations_phenotype.{ext}")


def plot_heritability_estimates(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot Falconer heritability estimates vs configured A values."""
    panels = [
        ("falconer_h2_trait1", "A1", "Falconer h² Trait 1", "Heritability"),
        ("parent_offspring_liability1_slope", "A1", "Midparent-Offspring Liability1", "Slope"),
        ("falconer_h2_trait2", "A2", "Falconer h² Trait 2", "Heritability"),
        ("parent_offspring_liability2_slope", "A2", "Midparent-Offspring Liability2", "Slope"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=_figsize(nrows=2, ncols=2))
    for ax, (col, exp, title, ylabel) in zip(axes.flat, panels, strict=True):
        stripplot(df, ax, col, expected=exp)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    save(fig, out / f"heritability_estimates.{ext}")


def plot_half_sib_proportions(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot observed vs expected half-sib proportions."""
    fig, axes = plt.subplots(1, 2, figsize=_figsize(ncols=2))
    stripplot(df, axes[0], "half_sib_prop_observed", expected="half_sib_prop_expected")
    axes[0].set_title("Half-Sibling Pair Proportion")
    axes[0].set_ylabel("Proportion")

    stripplot(df, axes[1], "offspring_with_half_sib_observed")
    axes[1].set_title("Proportion of Offspring with Half-Siblings")
    axes[1].set_ylabel("Proportion")
    save(fig, out / f"half_sib_proportions.{ext}")


def plot_cross_trait_correlations(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot cross-trait genetic and environmental correlations vs expected."""
    panels = [
        ("observed_rA", "rA", "Cross-Trait rA"),
        ("observed_rC", "rC", "Cross-Trait rC"),
        ("observed_rE", None, "Cross-Trait rE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=_figsize(ncols=3))
    for ax, (obs, exp, title) in zip(axes, panels, strict=True):
        if exp:
            stripplot(df, ax, obs, expected=exp)
        else:
            stripplot(df, ax, obs)
            ax.axhline(y=0, color="C1", linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
    save(fig, out / f"cross_trait_correlations.{ext}")


def plot_family_size(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot mean family size distribution across scenarios."""
    fig, ax = plt.subplots(figsize=_figsize())
    scenarios = df["scenario"].unique()
    positions = {s: i for i, s in enumerate(scenarios)}
    width = 0.3

    for scenario in scenarios:
        sdf = df[df["scenario"] == scenario]
        x = positions[scenario]
        ax.scatter(
            [x - width / 2] * len(sdf),
            sdf["mother_mean_offspring"],
            color="C0",
            alpha=0.9,
            s=30,
            zorder=5,
        )
        ax.scatter(
            [x + width / 2] * len(sdf),
            sdf["father_mean_offspring"],
            color="C3",
            alpha=0.9,
            s=30,
            zorder=5,
        )
        # Expected mean offspring per mother marker (~2.0 for balanced sex)
        expected = 2.0
        ax.scatter(
            x,
            expected,
            marker="_",
            s=200,
            linewidths=3,
            color="C1",
            zorder=10,
        )

    ax.set_xticks(range(len(scenarios)))
    _long = max((len(str(s)) for s in scenarios), default=0) > 12
    if len(scenarios) > 3 or (len(scenarios) > 1 and _long):
        ax.set_xticklabels(scenarios, rotation=30, ha="right")
    else:
        ax.set_xticklabels(scenarios)
    if len(scenarios) == 1:
        ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("Mean Offspring per Parent")
    ax.set_title("Family Size: Mean Offspring per Mother and Father (parents with children only)")

    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=6, label="Mother"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=6, label="Father"),
        Line2D([0], [0], marker="_", color="C1", markersize=10, linewidth=2, label="Expected (Poisson)"),
    ]
    ax.legend(handles=legend, loc="best", fontsize="small")
    save(fig, out / f"family_size.{ext}")


def plot_summary_bias(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot bias heatmap for variance components and correlations."""
    dp = df.copy()
    # Coerce to numeric; per-generation dicts become NaN (bias undefined)
    for col in ["A1", "C1", "E1"]:
        dp[col] = pd.to_numeric(dp[col], errors="coerce")
    dp["A1 Bias"] = dp["variance_A1"] - dp["A1"]
    dp["C1 Bias"] = dp["variance_C1"] - dp["C1"]
    dp["E1 Bias"] = dp["variance_E1"] - dp["E1"]
    dp["Twin Rate Bias"] = dp["observed_twin_rate"] - dp["p_mztwin"]
    dp["DZ A1 Corr Bias"] = dp["dz_sibling_A1_corr"] - 0.5
    dp["Half-sib A1 Bias"] = dp["half_sib_A1_corr"] - 0.25

    panels = [
        "A1 Bias",
        "C1 Bias",
        "E1 Bias",
        "Twin Rate Bias",
        "DZ A1 Corr Bias",
        "Half-sib A1 Bias",
    ]
    scenarios = dp["scenario"].unique()
    n = len(scenarios)
    _long = max((len(str(s)) for s in scenarios), default=0) > 12
    fig, axes = plt.subplots(2, 3, figsize=_figsize(nrows=2, ncols=3))

    for ax, col in zip(axes.flat, panels, strict=True):
        if dp[col].isna().all():
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="0.5")
            ax.set_title(col)
            ax.set_xlabel("")
            continue
        sns.stripplot(data=dp, x="scenario", y=col, ax=ax, alpha=0.9, jitter=0.15)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(col)
        ax.set_xlabel("")
        if n > 3 or (n > 1 and _long):
            ax.tick_params(axis="x", rotation=30)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
        if n == 1:
            ax.set_xlim(-0.5, 0.5)
        # Tight y-axis: include zero (the reference line) in span
        vals = dp[col].dropna().values
        all_v = np.concatenate([vals, [0.0]])
        lo, hi = float(all_v.min()), float(all_v.max())
        span = hi - lo
        pad = max(span * 0.15, 0.002)
        ax.set_ylim(lo - pad, hi + pad)
    save(fig, out / f"summary_bias.{ext}")


def _format_log_axes(ax: Axes) -> None:
    """Add readable tick marks to log-scaled axes.

    Places major ticks at 1, 2, 5 × 10^n so that intermediate values are
    visible, and adds minor ticks at the remaining integers for grid context.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator

    subs_major = [1.0, 2.0, 5.0]  # labelled ticks at 1, 2, 5 per decade
    subs_minor = [3.0, 4.0, 6.0, 7.0, 8.0, 9.0]  # unlabelled grid ticks

    def _smart_fmt(val, _pos):
        if val == 0:
            return "0"
        if val >= 1:
            if val == int(val):
                return f"{int(val):,}"
            return f"{val:,.1f}"
        return f"{val:g}"

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=subs_major, numticks=30))
        axis.set_major_formatter(FuncFormatter(_smart_fmt))
        axis.set_minor_locator(LogLocator(base=10, subs=subs_minor, numticks=30))
        axis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))

    ax.tick_params(axis="both", which="minor", length=3, color="0.7")
    ax.tick_params(axis="both", which="major", length=6)
    ax.grid(True, which="major", linewidth=0.8, alpha=0.5)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.3)


def plot_runtime(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot simulation runtime per scenario."""
    sub = df.dropna(subset=["simulate_seconds"])
    if sub.empty:
        logger.warning("No simulate_seconds data; skipping runtime plot")
        return

    unique_n = sub["N"].nunique()
    if unique_n <= 1:
        # Single N value — use stripplot instead of log-log scatter
        fig, ax = plt.subplots(figsize=_figsize())
        stripplot(sub, ax, "simulate_seconds")
        ax.set_ylabel("Simulate Time (seconds)")
        ax.set_title("Simulation Runtime")
        save(fig, out / f"runtime.{ext}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scenarios = sub["scenario"].unique()
    palette = sns.color_palette("colorblind", len(scenarios))
    color_map = dict(zip(scenarios, palette, strict=True))

    for scenario in scenarios:
        sdf = sub[sub["scenario"] == scenario]
        ax.scatter(
            sdf["N"],
            sdf["simulate_seconds"],
            color=color_map[scenario],
            label=scenario,
            alpha=0.9,
            s=40,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    _format_log_axes(ax)
    ax.set_xlabel("Population Size (N)")
    ax.set_ylabel("Simulate Time (seconds)")
    ax.set_title("Simulation Runtime vs Population Size")
    ax.legend()
    save(fig, out / f"runtime.{ext}")


def plot_memory(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot simulation peak memory usage per scenario."""
    sub = df.dropna(subset=["simulate_max_rss_mb"])
    if sub.empty:
        logger.warning("No simulate_max_rss_mb data; skipping memory plot")
        return

    unique_n = sub["N"].nunique()
    if unique_n <= 1:
        # Single N value — use stripplot instead of log-log scatter
        fig, ax = plt.subplots(figsize=_figsize())
        stripplot(sub, ax, "simulate_max_rss_mb")
        ax.set_ylabel("Peak RSS (MB)")
        ax.set_title("Simulation Memory Usage")
        save(fig, out / f"memory.{ext}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scenarios = sub["scenario"].unique()
    palette = sns.color_palette("colorblind", len(scenarios))
    color_map = dict(zip(scenarios, palette, strict=True))

    for scenario in scenarios:
        sdf = sub[sub["scenario"] == scenario]
        ax.scatter(
            sdf["N"],
            sdf["simulate_max_rss_mb"],
            color=color_map[scenario],
            label=scenario,
            alpha=0.9,
            s=40,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    _format_log_axes(ax)
    ax.set_xlabel("Population Size (N)")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Simulation Memory Usage vs Population Size")
    ax.legend()
    save(fig, out / f"memory.{ext}")


def plot_consanguineous_matings(df: pd.DataFrame, out: Path, ext: str = "png") -> None:
    """Plot consanguineous mating counts and inbreeding coefficients."""
    fig, axes = plt.subplots(1, 2, figsize=_figsize(ncols=2))
    stripplot(df, axes[0], "n_half_sib_matings", expected=0)
    axes[0].set_title("Half-Sib Matings")
    axes[0].set_ylabel("Count")

    stripplot(df, axes[1], "missing_gp_links", expected=0)
    axes[1].set_title("Missing Grandparent Links")
    axes[1].set_ylabel("Count")
    save(fig, out / f"consanguineous_matings.{ext}")


def main(tsv_path: str, output_dir: str | Path, plot_ext: str = "png") -> None:
    """Generate all validation plots from a gathered metrics TSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Generating validation plots in %s", out)
    sns.set_theme(style="whitegrid", palette="colorblind")
    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")

    # Sort scenarios by increasing N so x-axes read left-to-right by size
    if "N" in df.columns:
        scenario_order = df.groupby("scenario")["N"].first().sort_values().index
        df["scenario"] = pd.Categorical(df["scenario"], categories=scenario_order, ordered=True)
        df = df.sort_values("scenario").reset_index(drop=True)

    plot_variance_components(df, out, ext=plot_ext)
    plot_twin_rate(df, out, ext=plot_ext)
    plot_A_correlations(df, out, ext=plot_ext)
    plot_phenotype_correlations(df, out, ext=plot_ext)
    plot_heritability_estimates(df, out, ext=plot_ext)
    plot_half_sib_proportions(df, out, ext=plot_ext)
    plot_cross_trait_correlations(df, out, ext=plot_ext)
    plot_family_size(df, out, ext=plot_ext)
    plot_summary_bias(df, out, ext=plot_ext)
    plot_runtime(df, out, ext=plot_ext)
    plot_memory(df, out, ext=plot_ext)
    plot_consanguineous_matings(df, out, ext=plot_ext)

    # Assemble validation atlas PDF
    from sim_ace.plotting.plot_atlas import VALIDATION_CAPTIONS, assemble_atlas

    _VALIDATION_BASENAMES = [
        "family_size",
        "twin_rate",
        "half_sib_proportions",
        "consanguineous_matings",
        "variance_components",
        "correlations_A",
        "correlations_phenotype",
        "heritability_estimates",
        "cross_trait_correlations",
        "summary_bias",
        "runtime",
        "memory",
    ]
    atlas_paths = [out / f"{name}.{plot_ext}" for name in _VALIDATION_BASENAMES]
    assemble_atlas(atlas_paths, VALIDATION_CAPTIONS, out / "atlas.pdf")


def cli() -> None:
    """Command-line interface for generating validation plots."""
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Plot validation results")
    add_logging_args(parser)
    parser.add_argument("tsv", help="Validation summary TSV path")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--plot-format", choices=["png", "pdf"], default="png", help="Output plot format (default: png)"
    )
    args = parser.parse_args()

    init_logging(args)

    main(args.tsv, args.output_dir, plot_ext=args.plot_format)
