"""
Compute per-rep statistics for the liability threshold phenotype model.

Reads a single phenotype.liability_threshold.parquet and produces:
  - threshold_stats.yaml: prevalence, tetrachoric correlations, joint affection
  - threshold_samples.parquet: downsampled rows for plotting
"""

import argparse

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


def compute_prevalence_by_generation(df):
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


def compute_joint_affection(df):
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


def compute_liability_by_status(df):
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


def compute_tetrachoric(df, seed=42, pairs=None):
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


def main(phenotype_path, stats_output, samples_output, seed=42):
    """Compute all threshold stats for a single rep and write outputs."""
    df = pd.read_parquet(phenotype_path)

    print(f"Computing threshold stats for {phenotype_path} ({len(df)} rows)")

    stats = {}
    stats["n_individuals"] = int(len(df))

    # Prevalence by generation
    stats["prevalence"] = compute_prevalence_by_generation(df)

    # Joint affection
    stats["joint_affection"] = compute_joint_affection(df)

    # Liability by status
    stats["liability_by_status"] = compute_liability_by_status(df)

    # Extract relationship pairs once
    print("Extracting relationship pairs...")
    pairs = extract_relationship_pairs(df, seed=seed)

    # Liability correlations
    print("Computing liability correlations...")
    stats["liability_correlations"] = compute_liability_correlations(df, seed=seed, pairs=pairs)

    # Tetrachoric correlations
    print("Computing tetrachoric correlations...")
    stats["tetrachoric"] = compute_tetrachoric(df, seed=seed, pairs=pairs)

    # Write stats YAML
    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    print(f"Stats written to {stats_path}")

    # Write downsampled parquet
    sample_df = create_sample(df, seed=seed)
    samples_path = Path(samples_output)
    sample_df.to_parquet(samples_path, index=False)
    print(f"Sample ({len(sample_df)} rows) written to {samples_path}")


def cli():
    """Command-line interface for computing threshold statistics."""
    parser = argparse.ArgumentParser(description="Compute threshold phenotype statistics")
    parser.add_argument("phenotype", help="Input phenotype parquet")
    parser.add_argument("stats_output", help="Output stats YAML")
    parser.add_argument("samples_output", help="Output samples parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.phenotype, args.stats_output, args.samples_output, seed=args.seed)
