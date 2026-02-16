"""
Liability threshold phenotype model for two correlated traits.

Converts liability to binary affected status using a per-generation
prevalence threshold. Liability is standardized within each generation,
then the top X% (determined by prevalence) are classified as affected.

No time-to-event or censoring -- purely binary outcome.
"""

import argparse

import numpy as np
import pandas as pd


def apply_threshold(liability, generation, prevalence):
    """Apply liability threshold model per generation.

    Within each generation, standardize liability (mean=0, std=1),
    then classify the top `prevalence` fraction as affected.

    Args:
        liability: array of liability values
        generation: array of generation labels (same length as liability)
        prevalence: fraction of individuals affected per generation (0-1)

    Returns:
        affected: boolean array (True = affected)

    Raises:
        ValueError: if prevalence is not in (0, 1)
    """
    if not (0 < prevalence < 1):
        raise ValueError(f"prevalence must be between 0 and 1 (exclusive), got {prevalence}")

    affected = np.zeros(len(liability), dtype=bool)
    for gen in np.unique(generation):
        mask = generation == gen
        liab_gen = liability[mask]
        # Standardize within generation
        mean = liab_gen.mean()
        std = liab_gen.std()
        if std > 0:
            standardized = (liab_gen - mean) / std
        else:
            standardized = liab_gen - mean
        # Threshold: top prevalence fraction are affected
        threshold = np.percentile(standardized, 100 * (1 - prevalence))
        affected[mask] = standardized >= threshold
    return affected


def run_threshold(pedigree, params):
    """Orchestrate threshold phenotype from pedigree and parameter dict.

    Args:
        pedigree: DataFrame with pedigree data
        params: dict with keys: G_pheno, prevalence1, prevalence2

    Returns:
        phenotype DataFrame
    """
    # Filter to last G_pheno generations
    max_gen = pedigree["generation"].max()
    min_pheno_gen = max_gen - params["G_pheno"] + 1
    assert min_pheno_gen >= 0, (
        f"G_pheno ({params['G_pheno']}) > available generations ({max_gen + 1})"
    )
    pedigree = pedigree[pedigree["generation"] >= min_pheno_gen].reset_index(drop=True)

    generation = pedigree["generation"].values

    affected1 = apply_threshold(
        pedigree["liability1"].values, generation, params["prevalence1"]
    )
    affected2 = apply_threshold(
        pedigree["liability2"].values, generation, params["prevalence2"]
    )

    phenotype = pd.DataFrame(
        {
            "id": pedigree["id"].values,
            "generation": generation,
            "mother": pedigree["mother"].values,
            "father": pedigree["father"].values,
            "twin": pedigree["twin"].values,
            "A1": pedigree["A1"].values,
            "C1": pedigree["C1"].values,
            "E1": pedigree["E1"].values,
            "liability1": pedigree["liability1"].values,
            "A2": pedigree["A2"].values,
            "C2": pedigree["C2"].values,
            "E2": pedigree["E2"].values,
            "liability2": pedigree["liability2"].values,
            "affected1": affected1,
            "affected2": affected2,
        }
    )

    return phenotype


def cli():
    """Command-line interface for threshold phenotype simulation."""
    parser = argparse.ArgumentParser(description="Apply liability threshold model")
    parser.add_argument("--pedigree", required=True, help="Input pedigree parquet")
    parser.add_argument("--output", required=True, help="Output phenotype parquet")
    parser.add_argument("--G-pheno", type=int, default=3)
    parser.add_argument("--prevalence1", type=float, default=0.1)
    parser.add_argument("--prevalence2", type=float, default=0.1)
    args = parser.parse_args()

    pedigree = pd.read_parquet(args.pedigree)
    params = {
        "G_pheno": args.G_pheno,
        "prevalence1": args.prevalence1,
        "prevalence2": args.prevalence2,
    }
    phenotype = run_threshold(pedigree, params)
    phenotype.to_parquet(args.output, index=False)
