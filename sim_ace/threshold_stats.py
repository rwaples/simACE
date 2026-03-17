"""
Compute per-rep statistics for the liability threshold phenotype model.

Reads a single phenotype.liability_threshold.parquet and produces:
  - threshold_stats.yaml: prevalence, tetrachoric correlations, joint affection
  - threshold_samples.parquet: downsampled rows for plotting
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sim_ace.pedigree_graph import extract_relationship_pairs
from sim_ace.stats import (
    compute_cross_trait_tetrachoric,
    compute_joint_affection,
    compute_liability_correlations,
    compute_tetrachoric,
    create_sample,
)
from sim_ace.utils import save_parquet

logger = logging.getLogger(__name__)


def compute_prevalence_by_generation(df: pd.DataFrame) -> dict[str, Any]:
    """Compute per-trait, per-generation prevalence."""
    result = {}
    for trait_num in [1, 2]:
        col = f"affected{trait_num}"
        gen_prev = df.groupby("generation")[col].mean()
        result[f"trait{trait_num}"] = {
            "generations": gen_prev.index.tolist(),
            "prevalence": gen_prev.values.tolist(),
            "overall": float(df[col].mean()),
        }
    return result


def compute_liability_by_status(df: pd.DataFrame) -> dict[str, Any]:
    """Compute mean/std of liability for affected vs unaffected, per trait."""
    result = {}
    for trait_num in [1, 2]:
        affected = df[df[f"affected{trait_num}"]][f"liability{trait_num}"]
        unaffected = df[~df[f"affected{trait_num}"]][f"liability{trait_num}"]
        result[f"trait{trait_num}"] = {
            "affected_mean": float(affected.mean()) if len(affected) > 0 else None,
            "affected_std": float(affected.std()) if len(affected) > 1 else None,
            "unaffected_mean": float(unaffected.mean()) if len(unaffected) > 0 else None,
            "unaffected_std": float(unaffected.std()) if len(unaffected) > 1 else None,
        }
    return result


def main(
    phenotype_path: str,
    stats_output: str,
    samples_output: str,
    seed: int = 42,
    extra_tetrachoric: bool = True,
    pedigree_path: str | None = None,
) -> None:  # extra_tetrachoric kept for API compat
    """Compute all threshold stats for a single rep and write outputs."""
    df = pd.read_parquet(phenotype_path)

    logger.info("Computing threshold stats for %s (%d rows)", phenotype_path, len(df))

    stats: dict[str, Any] = {}
    stats["n_individuals"] = len(df)

    # Prevalence by generation
    stats["prevalence"] = compute_prevalence_by_generation(df)

    # Joint affection
    stats["joint_affection"] = compute_joint_affection(df)

    # Liability by status
    stats["liability_by_status"] = compute_liability_by_status(df)

    # Extract relationship pairs once
    full_ped = pd.read_parquet(pedigree_path) if pedigree_path is not None else None
    logger.info("Extracting relationship pairs...")
    t_pairs = time.perf_counter()
    pairs = extract_relationship_pairs(df, seed=seed, full_pedigree=full_ped)
    del full_ped
    logger.info(
        "Relationship pairs extracted in %.1fs: %s",
        time.perf_counter() - t_pairs,
        ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
    )

    # Liability correlations
    logger.info("Computing liability correlations...")
    stats["liability_correlations"] = compute_liability_correlations(df, seed=seed, pairs=pairs)

    # Tetrachoric correlations (always run — fast O(N) scalar MLEs)
    logger.info("Computing tetrachoric correlations...")
    t_tet = time.perf_counter()
    stats["tetrachoric"] = compute_tetrachoric(df, seed=seed, pairs=pairs)
    logger.info("Tetrachoric correlations computed in %.1fs", time.perf_counter() - t_tet)

    logger.info("Computing cross-trait tetrachoric correlations...")
    t_ct = time.perf_counter()
    stats["cross_trait_tetrachoric"] = compute_cross_trait_tetrachoric(df, seed=seed, pairs=pairs)
    logger.info("Cross-trait tetrachoric computed in %.1fs", time.perf_counter() - t_ct)

    # Write stats YAML
    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    logger.info("Stats written to %s", stats_path)

    # Write downsampled parquet
    sample_df = create_sample(df, seed=seed)
    samples_path = Path(samples_output)
    save_parquet(sample_df, samples_path)
    logger.info("Sample (%d rows) written to %s", len(sample_df), samples_path)


def cli() -> None:
    """Command-line interface for computing threshold statistics."""
    from sim_ace.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Compute threshold phenotype statistics")
    add_logging_args(parser)
    parser.add_argument("phenotype", help="Input phenotype parquet")
    parser.add_argument("stats_output", help="Output stats YAML")
    parser.add_argument("samples_output", help="Output samples parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--pedigree", default=None, help="Full pedigree parquet for complete pair extraction")
    parser.add_argument(
        "--no-extra-tetrachoric",
        dest="extra_tetrachoric",
        action="store_false",
        default=True,
        help="No-op (kept for CLI compatibility; basic tetrachoric always runs)",
    )
    args = parser.parse_args()

    init_logging(args)

    main(
        args.phenotype,
        args.stats_output,
        args.samples_output,
        seed=args.seed,
        extra_tetrachoric=args.extra_tetrachoric,
        pedigree_path=args.pedigree,
    )
