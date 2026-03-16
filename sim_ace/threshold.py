"""
Liability threshold phenotype model for two correlated traits.

Converts liability to binary affected status using a per-generation
prevalence threshold. Liability is standardized within each generation,
then the top X% (determined by prevalence) are classified as affected.

No time-to-event or censoring -- purely binary outcome.
"""

from __future__ import annotations

import argparse
from typing import Any

import logging
import time

import numpy as np
import pandas as pd

from sim_ace.utils import save_parquet

logger = logging.getLogger(__name__)


def apply_threshold(liability: np.ndarray, generation: np.ndarray, prevalence: float | dict[int, float]) -> np.ndarray:
    """Apply liability threshold model per generation.

    Within each generation, standardize liability (mean=0, std=1),
    then classify the top `prevalence` fraction as affected.

    Args:
        liability: array of liability values
        generation: array of generation labels (same length as liability)
        prevalence: fraction affected per generation — either a single float
            applied to all generations, or a dict mapping generation number
            to prevalence (e.g. {0: 0.05, 1: 0.08, 2: 0.10})

    Returns:
        affected: boolean array (True = affected)

    Raises:
        ValueError: if any prevalence value is not in (0, 1), or if a dict
            is provided but is missing entries for observed generations
    """
    unique_gens = np.unique(generation)

    if isinstance(prevalence, dict):
        # Validate: every phenotyped generation must have an entry
        missing = [int(g) for g in unique_gens if int(g) not in prevalence]
        if missing:
            raise ValueError(
                f"prevalence dict is missing entries for generations: {missing}. "
                f"Dict has keys {sorted(prevalence.keys())}, "
                f"but data contains generations {sorted(int(g) for g in unique_gens)}"
            )
        for gen_key, prev_val in prevalence.items():
            if not (0 < prev_val < 1):
                raise ValueError(
                    f"prevalence must be between 0 and 1 (exclusive), "
                    f"got {prev_val} for generation {gen_key}"
                )
    else:
        if not (0 < prevalence < 1):
            raise ValueError(f"prevalence must be between 0 and 1 (exclusive), got {prevalence}")

    affected = np.zeros(len(liability), dtype=bool)
    for gen in unique_gens:
        mask = generation == gen
        liab_gen = liability[mask]
        # Standardize within generation
        mean = liab_gen.mean()
        std = liab_gen.std()
        if std > 0:
            standardized = (liab_gen - mean) / std
        else:
            standardized = liab_gen - mean
        # Look up per-gen prevalence
        prev = prevalence[int(gen)] if isinstance(prevalence, dict) else prevalence
        # Threshold: top prevalence fraction are affected
        threshold = np.percentile(standardized, 100 * (1 - prev))
        affected[mask] = standardized >= threshold
    return affected


def _apply_threshold_sex_aware(
    liability: np.ndarray,
    generation: np.ndarray,
    sex: np.ndarray,
    params: dict[str, Any],
    trait_num: int,
) -> np.ndarray:
    """Apply threshold with optional sex-specific prevalence.

    When ``prevalence{N}`` is a dict with ``"female"`` and ``"male"`` keys,
    thresholds are applied separately per sex.  Each sex value may itself
    be a scalar or a per-generation dict (composing naturally with
    ``apply_threshold``).  Otherwise falls back to the standard
    ``apply_threshold`` with the scalar/dict prevalence directly.
    """
    prev = params[f"prevalence{trait_num}"]
    if isinstance(prev, dict) and "female" in prev and "male" in prev:
        affected = np.zeros(len(liability), dtype=bool)
        for sex_val, key in [(0, "female"), (1, "male")]:
            mask = sex == sex_val
            affected[mask] = apply_threshold(
                liability[mask], generation[mask], prev[key]
            )
        return affected
    return apply_threshold(liability, generation, prev)


def run_threshold(pedigree: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Orchestrate threshold phenotype from pedigree and parameter dict.

    Args:
        pedigree: DataFrame with pedigree data
        params: dict with keys: G_pheno, prevalence1, prevalence2

    Returns:
        phenotype DataFrame
    """
    p1, p2 = params["prevalence1"], params["prevalence2"]
    if isinstance(p1, dict) or isinstance(p2, dict):
        logger.info("Applying threshold model: prevalence1=%s, prevalence2=%s", p1, p2)
    else:
        logger.info("Applying threshold model: prevalence1=%.3f, prevalence2=%.3f", p1, p2)
    t0 = time.perf_counter()
    # Filter to last G_pheno generations
    max_gen = pedigree["generation"].max()
    min_pheno_gen = max_gen - params["G_pheno"] + 1
    assert min_pheno_gen >= 0, (
        f"G_pheno ({params['G_pheno']}) > available generations ({max_gen + 1})"
    )
    pedigree = pedigree[pedigree["generation"] >= min_pheno_gen].reset_index(drop=True)

    generation = pedigree["generation"].values
    sex = pedigree["sex"].values

    affected1 = _apply_threshold_sex_aware(
        pedigree["liability1"].values, generation, sex, params, trait_num=1,
    )
    affected2 = _apply_threshold_sex_aware(
        pedigree["liability2"].values, generation, sex, params, trait_num=2,
    )

    phenotype = pd.DataFrame(
        {
            "id": pedigree["id"].values,
            "sex": pedigree["sex"].values,
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

    elapsed = time.perf_counter() - t0
    logger.info("Threshold model complete in %.1fs: %d individuals", elapsed, len(phenotype))

    return phenotype


def _parse_prevalence_arg(scalar: float | None, by_gen_json: str | None) -> float | dict[int, float] | None:
    """Resolve prevalence from scalar flag or JSON by-gen flag."""
    import json
    if by_gen_json is not None:
        raw = json.loads(by_gen_json)
        return {int(k): float(v) for k, v in raw.items()}
    return scalar


def cli() -> None:
    """Command-line interface for threshold phenotype simulation."""
    from sim_ace.cli_base import add_logging_args, init_logging
    parser = argparse.ArgumentParser(description="Apply liability threshold model")
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True, help="Input pedigree parquet")
    parser.add_argument("--output", required=True, help="Output phenotype parquet")
    parser.add_argument("--G-pheno", type=int, default=3, help="Number of generations to assign phenotypes")
    parser.add_argument("--prevalence1", type=float, default=0.1, help="Disease prevalence for trait 1")
    parser.add_argument("--prevalence2", type=float, default=0.1, help="Disease prevalence for trait 2")
    parser.add_argument("--prevalence1-by-gen", type=str, default=None,
                        help='Per-gen prevalence for trait 1 as JSON, e.g. \'{"0":0.05,"1":0.10}\'')
    parser.add_argument("--prevalence2-by-gen", type=str, default=None,
                        help='Per-gen prevalence for trait 2 as JSON, e.g. \'{"0":0.05,"1":0.10}\'')
    args = parser.parse_args()

    init_logging(args)

    pedigree = pd.read_parquet(args.pedigree)
    params = {
        "G_pheno": args.G_pheno,
        "prevalence1": _parse_prevalence_arg(args.prevalence1, args.prevalence1_by_gen),
        "prevalence2": _parse_prevalence_arg(args.prevalence2, args.prevalence2_by_gen),
    }
    phenotype = run_threshold(pedigree, params)
    save_parquet(phenotype, args.output)
