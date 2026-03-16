"""
Plot phenotype distributions from pre-computed per-rep statistics.

Reads phenotype_stats.yaml and phenotype_samples.parquet files (one per rep)
produced by compute_phenotype_stats.py. No full phenotype parquet loading needed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

from sim_ace.utils import yaml_loader
_yaml_loader = yaml_loader()

import logging
logger = logging.getLogger(__name__)

MAX_PLOT_POINTS = 200_000

from sim_ace.plot_distributions import (
    plot_death_age_distribution,
    plot_trait_phenotype,
    plot_trait_regression,
    plot_cumulative_incidence,
    plot_cumulative_incidence_by_sex,
    plot_cumulative_incidence_by_sex_generation,
    plot_censoring_windows,
)
from sim_ace.plot_liability import (
    plot_liability_joint,
    plot_liability_joint_affected,
    plot_liability_violin,
    plot_liability_violin_by_generation,
    plot_joint_affection,
    plot_censoring_confusion,
    plot_censoring_cascade,
)
from sim_ace.plot_pedigree_counts import plot_pedigree_relationship_counts
from sim_ace.plot_correlations import (
    plot_tetrachoric_sibling,
    plot_tetrachoric_by_generation,
    plot_cross_trait_tetrachoric,
    plot_parent_offspring_liability,
    plot_heritability_by_generation,
    plot_broad_heritability_by_generation,
    plot_cross_trait_frailty_by_generation,
)


def main(
    stats_paths: list[str],
    sample_paths: list[str],
    output_dir: str,
    censor_age: float,
    gen_censoring: dict[int, list[float]] | None = None,
    plot_ext: str = "png",
    validation_paths: list[str] | None = None,
) -> None:
    """Generate all phenotype plots from pre-computed stats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = out_dir.parent.name
    sns.set_theme(style="whitegrid", palette="colorblind")

    all_stats = []
    for p in stats_paths:
        with open(p, encoding="utf-8") as f:
            all_stats.append(yaml.load(f, Loader=_yaml_loader))

    df_samples = pd.concat(
        [pd.read_parquet(p) for p in sample_paths], ignore_index=True
    )
    if len(df_samples) > MAX_PLOT_POINTS:
        df_samples = df_samples.sample(
            n=MAX_PLOT_POINTS, random_state=42
        ).reset_index(drop=True)

    ext = plot_ext

    # Pedigree relationship pair counts
    plot_pedigree_relationship_counts(
        all_stats, out_dir / f"pedigree_counts.ped.{ext}", scenario,
        stats_key="pair_counts_ped",
        generations_label="G_ped",
    )
    plot_pedigree_relationship_counts(
        all_stats, out_dir / f"pedigree_counts.{ext}", scenario,
        generations_label="G_pheno",
    )

    # Distribution plots
    plot_death_age_distribution(
        all_stats, censor_age, out_dir / f"mortality.{ext}", scenario,
    )
    plot_trait_phenotype(
        df_samples, out_dir / f"age_at_onset_death.{ext}", scenario,
    )
    plot_trait_regression(
        df_samples, all_stats, out_dir / f"liability_vs_aoo.{ext}", scenario,
    )

    # Liability plots
    plot_liability_joint(
        df_samples, out_dir / f"cross_trait.{ext}", scenario,
    )
    plot_liability_joint_affected(
        df_samples, out_dir / f"cross_trait.frailty.{ext}", scenario,
    )
    plot_liability_violin(
        df_samples, all_stats, out_dir / f"liability_violin.frailty.{ext}", scenario,
    )
    plot_liability_violin_by_generation(
        df_samples, all_stats,
        out_dir / f"liability_violin.frailty.by_generation.{ext}", scenario,
    )

    # Survival / incidence plots
    plot_cumulative_incidence(
        all_stats, censor_age,
        out_dir / f"cumulative_incidence.frailty.{ext}", scenario,
    )
    plot_cumulative_incidence_by_sex(
        df_samples, censor_age,
        out_dir / f"cumulative_incidence.by_sex.{ext}", scenario,
    )
    plot_cumulative_incidence_by_sex_generation(
        df_samples, censor_age,
        out_dir / f"cumulative_incidence.by_sex.by_generation.{ext}", scenario,
    )
    plot_joint_affection(
        df_samples, out_dir / f"joint_affected.frailty.{ext}", scenario,
        all_stats=all_stats,
    )

    # Censoring
    if gen_censoring is not None:
        plot_censoring_windows(
            all_stats, out_dir / f"censoring.{ext}", scenario,
            gen_censoring=gen_censoring,
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No censoring windows configured",
                ha="center", va="center", transform=ax.transAxes)
        plt.savefig(out_dir / f"censoring.{ext}", dpi=150)
        plt.close()

    plot_censoring_confusion(
        df_samples, censor_age, out_dir / f"censoring_confusion.{ext}",
        scenario, gen_censoring=gen_censoring,
    )
    plot_censoring_cascade(
        df_samples, censor_age, out_dir / f"censoring_cascade.{ext}",
        scenario, gen_censoring=gen_censoring,
    )

    # Correlation plots
    plot_tetrachoric_sibling(
        all_stats, out_dir / f"tetrachoric.frailty.{ext}", scenario,
    )
    plot_tetrachoric_by_generation(
        all_stats, out_dir / f"tetrachoric.frailty.by_generation.{ext}", scenario,
    )
    plot_cross_trait_tetrachoric(
        all_stats, out_dir / f"cross_trait_tetrachoric.{ext}", scenario,
    )
    plot_parent_offspring_liability(
        df_samples, all_stats,
        out_dir / f"parent_offspring_liability.by_generation.{ext}", scenario,
    )
    plot_cross_trait_frailty_by_generation(
        all_stats, out_dir / f"cross_trait_frailty.by_generation.{ext}", scenario,
    )

    # Per-generation heritability (requires validation data)
    if validation_paths:
        all_validations = []
        for p in validation_paths:
            with open(p, encoding="utf-8") as f:
                all_validations.append(yaml.load(f, Loader=_yaml_loader))
        plot_heritability_by_generation(
            all_validations,
            out_dir / f"heritability.by_generation.{ext}", scenario,
        )
        plot_broad_heritability_by_generation(
            all_validations,
            out_dir / f"broad_heritability.by_generation.{ext}", scenario,
        )
    else:
        for name in ["heritability.by_generation", "broad_heritability.by_generation"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No validation data available",
                    ha="center", va="center", transform=ax.transAxes)
            plt.savefig(out_dir / f"{name}.{ext}", dpi=150)
            plt.close()

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
    parser.add_argument("--validations", nargs="*", default=None, help="Validation YAML paths")
    args = parser.parse_args()

    init_logging(args)

    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    main(
        args.stats, args.samples, args.output_dir, args.censor_age,
        gen_censoring=gen_censoring, plot_ext=args.plot_format,
        validation_paths=args.validations,
    )