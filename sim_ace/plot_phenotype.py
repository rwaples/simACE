"""
Plot phenotype distributions from pre-computed per-rep statistics.

Reads phenotype_stats.yaml and phenotype_samples.parquet files (one per rep)
produced by compute_phenotype_stats.py. No full phenotype parquet loading needed.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

try:
    _yaml_loader = yaml.CSafeLoader
except AttributeError:
    _yaml_loader = yaml.SafeLoader
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

MAX_PLOT_POINTS = 100_000

# -- Distribution plots --
from sim_ace.plot_distributions import (
    plot_death_age_distribution,
    plot_trait_phenotype,
    plot_trait_regression,
    plot_cumulative_incidence,
    plot_censoring_windows,
)

# -- Liability plots --
from sim_ace.plot_liability import (
    plot_liability_joint,
    plot_liability_joint_affected,
    plot_liability_violin,
    plot_liability_violin_by_generation,
    plot_joint_affection,
)

# -- Correlation plots --
from sim_ace.plot_correlations import (
    plot_tetrachoric_sibling,
    plot_tetrachoric_by_generation,
    plot_parent_offspring_liability,
)


def main(
    stats_paths: list[str],
    sample_paths: list[str],
    output_dir: str,
    censor_age: float,
    gen_censoring: dict[int, list[float]] | None = None,
    plot_ext: str = "png",
) -> None:
    """Generate all phenotype plots from pre-computed stats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = out_dir.parent.name
    sns.set_theme(style="whitegrid", palette="colorblind")

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

    ext = plot_ext

    plot_death_age_distribution(
        all_stats, censor_age, out_dir / f"mortality.{ext}", scenario
    )
    plot_trait_phenotype(
        df_samples, out_dir / f"age_at_onset_death.{ext}", scenario
    )
    plot_trait_regression(
        df_samples, all_stats, out_dir / f"liability_vs_aoo.{ext}", scenario
    )
    plot_liability_joint(
        df_samples, out_dir / f"cross_trait.{ext}", scenario
    )
    plot_liability_joint_affected(
        df_samples, out_dir / f"cross_trait.weibull.{ext}", scenario
    )
    plot_liability_violin(
        df_samples, all_stats, out_dir / f"liability_violin.weibull.{ext}", scenario
    )
    plot_liability_violin_by_generation(
        df_samples, all_stats, out_dir / f"liability_violin.weibull.by_generation.{ext}", scenario
    )
    plot_cumulative_incidence(
        all_stats, censor_age, out_dir / f"cumulative_incidence.weibull.{ext}", scenario
    )
    plot_joint_affection(
        df_samples, out_dir / f"joint_affected.weibull.{ext}", scenario
    )
    if gen_censoring is not None:
        plot_censoring_windows(
            all_stats, out_dir / f"censoring.{ext}", scenario,
            gen_censoring=gen_censoring,
        )
    else:
        # Create placeholder to satisfy Snakemake output
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No censoring windows configured",
                ha="center", va="center", transform=ax.transAxes)
        plt.savefig(out_dir / f"censoring.{ext}", dpi=150)
        plt.close()

    plot_tetrachoric_sibling(
        all_stats, out_dir / f"tetrachoric.weibull.{ext}", scenario,
    )
    plot_tetrachoric_by_generation(
        all_stats, out_dir / f"tetrachoric.weibull.by_generation.{ext}", scenario,
    )
    plot_parent_offspring_liability(
        df_samples, all_stats, out_dir / f"parent_offspring_liability.by_generation.{ext}", scenario,
    )

    logger.info("Phenotype plots saved to %s", out_dir)


def cli() -> None:
    """Command-line interface for generating phenotype plots."""
    from sim_ace.cli_base import add_logging_args, init_logging
    parser = argparse.ArgumentParser(description="Plot phenotype distributions")
    add_logging_args(parser)
    parser.add_argument("--stats", nargs="+", required=True, help="Stats YAML paths")
    parser.add_argument("--samples", nargs="+", required=True, help="Sample parquet paths")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--censor-age", type=float, required=True, help="Maximum follow-up age")
    parser.add_argument("--gen-censoring", type=str, default=None, help="Per-generation censoring windows as JSON dict")
    parser.add_argument("--plot-format", choices=["png", "pdf"], default="png", help="Output plot format (default: png)")
    args = parser.parse_args()

    init_logging(args)

    import json
    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    main(args.stats, args.samples, args.output_dir, args.censor_age,
         gen_censoring=gen_censoring, plot_ext=args.plot_format)
