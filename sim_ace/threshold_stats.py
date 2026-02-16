"""
Compute per-rep statistics for the liability threshold phenotype model.

Reads a single phenotype.liability_threshold.parquet and produces:
  - threshold_stats.yaml: prevalence, tetrachoric correlations, joint affection
  - threshold_samples.parquet: downsampled rows for plotting
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from sim_ace.stats import (
    tetrachoric_corr_se,
    extract_relationship_pairs,
    compute_liability_correlations,
    create_sample,
)

import logging
import time
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


def compute_joint_affection(df: pd.DataFrame) -> dict[str, Any]:
    """Compute 2x2 contingency table for trait1 x trait2 affection status."""
    a1 = df["affected1"].values.astype(bool)
    a2 = df["affected2"].values.astype(bool)
    n = len(df)

    counts = {
        "both": int(np.sum(a1 & a2)),
        "trait1_only": int(np.sum(a1 & ~a2)),
        "trait2_only": int(np.sum(~a1 & a2)),
        "neither": int(np.sum(~a1 & ~a2)),
    }
    proportions = {k: v / n for k, v in counts.items()}

    return {"counts": counts, "proportions": proportions, "n": n}


def compute_liability_by_status(df: pd.DataFrame) -> dict[str, Any]:
    """Compute mean/std of liability for affected vs unaffected, per trait."""
    result = {}
    for trait_num in [1, 2]:
        affected = df[df[f"affected{trait_num}"]]["liability{0}".format(trait_num)]
        unaffected = df[~df[f"affected{trait_num}"]]["liability{0}".format(trait_num)]
        result[f"trait{trait_num}"] = {
            "affected_mean": float(affected.mean()) if len(affected) > 0 else None,
            "affected_std": float(affected.std()) if len(affected) > 1 else None,
            "unaffected_mean": float(unaffected.mean()) if len(unaffected) > 0 else None,
            "unaffected_std": float(unaffected.std()) if len(unaffected) > 1 else None,
        }
    return result


def compute_tetrachoric(df: pd.DataFrame, seed: int = 42, pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> dict[str, Any]:
    """Compute tetrachoric correlations for all relationship types."""
    if pairs is None:
        pairs = extract_relationship_pairs(df, seed=seed)
    pair_types = ["MZ twin", "Full sib", "Mother-offspring", "Father-offspring", "Maternal half sib", "Paternal half sib", "1st cousin"]
    result = {}

    for trait_num in [1, 2]:
        affected = df[f"affected{trait_num}"].values.astype(bool)
        trait_result = {}
        for ptype in pair_types:
            idx1, idx2 = pairs[ptype]
            n_p = len(idx1)
            if n_p < 10:
                trait_result[ptype] = {"r": None, "se": None, "n_pairs": int(n_p)}
                continue
            a_vals = affected[idx1]
            b_vals = affected[idx2]
            r_tet, se = tetrachoric_corr_se(a_vals, b_vals)
            trait_result[ptype] = {
                "r": float(r_tet) if not np.isnan(r_tet) else None,
                "se": float(se) if not np.isnan(se) else None,
                "n_pairs": int(n_p),
            }
        result[f"trait{trait_num}"] = trait_result

    return result


def main(phenotype_path: str, stats_output: str, samples_output: str, seed: int = 42) -> None:
    """Compute all threshold stats for a single rep and write outputs."""
    df = pd.read_parquet(phenotype_path)

    logger.info("Computing threshold stats for %s (%d rows)", phenotype_path, len(df))

    stats = {}
    stats["n_individuals"] = int(len(df))

    # Prevalence by generation
    stats["prevalence"] = compute_prevalence_by_generation(df)

    # Joint affection
    stats["joint_affection"] = compute_joint_affection(df)

    # Liability by status
    stats["liability_by_status"] = compute_liability_by_status(df)

    # Extract relationship pairs once
    logger.info("Extracting relationship pairs...")
    t_pairs = time.perf_counter()
    pairs = extract_relationship_pairs(df, seed=seed)
    logger.info(
        "Relationship pairs extracted in %.1fs: %s",
        time.perf_counter() - t_pairs,
        ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
    )

    # Liability correlations
    logger.info("Computing liability correlations...")
    stats["liability_correlations"] = compute_liability_correlations(df, seed=seed, pairs=pairs)

    # Tetrachoric correlations
    logger.info("Computing tetrachoric correlations...")
    t_tet = time.perf_counter()
    stats["tetrachoric"] = compute_tetrachoric(df, seed=seed, pairs=pairs)
    logger.info("Tetrachoric correlations computed in %.1fs", time.perf_counter() - t_tet)

    # Write stats YAML
    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    logger.info("Stats written to %s", stats_path)

    # Write downsampled parquet
    sample_df = create_sample(df, seed=seed)
    samples_path = Path(samples_output)
    sample_df.to_parquet(samples_path, index=False)
    logger.info("Sample (%d rows) written to %s", len(sample_df), samples_path)


def cli() -> None:
    """Command-line interface for computing threshold statistics."""
    from sim_ace import setup_logging
    parser = argparse.ArgumentParser(description="Compute threshold phenotype statistics")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG output")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING+ only")
    parser.add_argument("phenotype", help="Input phenotype parquet")
    parser.add_argument("stats_output", help="Output stats YAML")
    parser.add_argument("samples_output", help="Output samples parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=level)

    main(args.phenotype, args.stats_output, args.samples_output, seed=args.seed)
