"""Orchestration entry point for per-rep phenotype statistics.

Reads a single phenotype.parquet, runs every stats computation, and writes
``phenotype_stats.yaml`` plus ``phenotype_samples.parquet``.
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pedigree_graph import PedigreeGraph

from simace.core.parquet import save_parquet
from simace.core.yaml_io import to_native

from .censoring import (
    compute_censoring_cascade,
    compute_censoring_confusion,
    compute_censoring_windows,
    compute_person_years,
)
from .correlations import (
    compute_affected_correlations,
    compute_cross_trait_tetrachoric,
    compute_liability_correlations,
    compute_mate_correlation,
    compute_observed_h2_estimators,
    compute_parent_offspring_affected_corr,
    compute_parent_offspring_corr,
    compute_parent_offspring_corr_by_sex,
    compute_tetrachoric,
    compute_tetrachoric_by_generation,
    compute_tetrachoric_by_sex,
)
from .incidence import (
    compute_cumulative_incidence,
    compute_cumulative_incidence_by_sex,
    compute_cumulative_incidence_by_sex_generation,
    compute_joint_affection,
    compute_mortality,
    compute_prevalence,
    compute_regression,
)
from .pedigree import compute_mean_family_size, compute_parent_status
from .sampling import create_sample

logger = logging.getLogger(__name__)


def main(
    phenotype_path: str,
    censor_age: float,
    stats_output: str,
    samples_output: str,
    seed: int = 42,
    gen_censoring: dict[int, list[float]] | None = None,
    pedigree_path: str | None = None,
    max_degree: int = 2,
    case_ascertainment_ratio: float = 1.0,
) -> None:
    """Compute all stats for a single rep and write outputs."""
    df = pd.read_parquet(phenotype_path)
    logger.info("Computing stats for %s (%d rows)", phenotype_path, len(df))

    stats: dict[str, Any] = {
        "n_individuals": len(df),
        "n_generations": int(df["generation"].nunique()) if "generation" in df.columns else 1,
    }

    if case_ascertainment_ratio != 1.0:
        stats["case_ascertainment_ratio"] = case_ascertainment_ratio

    stats["prevalence"] = compute_prevalence(df)
    stats["mortality"] = compute_mortality(df, censor_age)
    stats["regression"] = compute_regression(df)
    stats["cumulative_incidence"] = compute_cumulative_incidence(df, censor_age)
    stats["joint_affection"] = compute_joint_affection(df)
    stats["cumulative_incidence_by_sex"] = compute_cumulative_incidence_by_sex(df, censor_age)
    stats["cumulative_incidence_by_sex_generation"] = compute_cumulative_incidence_by_sex_generation(df, censor_age)

    if gen_censoring is not None:
        stats["censoring"] = compute_censoring_windows(df, censor_age, gen_censoring)
        stats["censoring_confusion"] = compute_censoring_confusion(df, censor_age, gen_censoring)
        stats["censoring_cascade"] = compute_censoring_cascade(df, censor_age, gen_censoring)

    stats["person_years"] = compute_person_years(df, censor_age, gen_censoring)
    stats["family_size"] = compute_mean_family_size(df)

    # Read full pedigree once (used for both pair extraction and pair counts)
    df_ped = pd.read_parquet(pedigree_path) if pedigree_path is not None else None

    logger.info("Extracting relationship pairs...")
    t0 = time.perf_counter()
    if df_ped is not None:
        pg = PedigreeGraph.from_subsample(df_ped, df)
        pairs = pg.extract_pairs(max_degree=max_degree)
        full_counts = pg.count_pairs(max_degree=max_degree, scope="full")
    else:
        pg = PedigreeGraph(df)
        pairs = pg.extract_pairs(max_degree=max_degree)
        full_counts = None
    logger.info(
        "Relationship pairs extracted in %.1fs: %s",
        time.perf_counter() - t0,
        ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
    )

    stats["pair_counts"] = {k: len(v[0]) for k, v in pairs.items()}

    stats["parent_status"] = compute_parent_status(df, df_ped)

    if df_ped is not None and full_counts is not None:
        stats["pair_counts_ped"] = full_counts
        stats["n_individuals_ped"] = len(df_ped)
        stats["n_generations_ped"] = int(df_ped["generation"].nunique()) if "generation" in df_ped.columns else 1
        logger.info(
            "Pedigree pair counts (from same graph): %s",
            ", ".join(f"{k}: {v}" for k, v in full_counts.items()),
        )

    if df_ped is not None:
        logger.info("Computing mate liability correlations...")
        stats["mate_correlation"] = compute_mate_correlation(df_ped)
    del df_ped

    # Fast sequential computations
    stats["liability_correlations"] = compute_liability_correlations(df, seed=seed, pairs=pairs)
    stats["affected_correlations"] = compute_affected_correlations(df, seed=seed, pairs=pairs)
    stats["parent_offspring_corr"] = compute_parent_offspring_corr(df)
    stats["parent_offspring_corr_by_sex"] = compute_parent_offspring_corr_by_sex(df)
    stats["parent_offspring_affected_corr"] = compute_parent_offspring_affected_corr(df)
    stats["observed_h2_estimators"] = compute_observed_h2_estimators(stats)

    # Expensive MLE computations — run in parallel (scipy.optimize releases the GIL)
    logger.info("Computing tetrachoric correlations in parallel...")
    t_mle = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:
        fut_tetra = pool.submit(compute_tetrachoric, df, seed=seed, pairs=pairs)
        fut_tetra_gen = pool.submit(compute_tetrachoric_by_generation, df, seed=seed, pairs=pairs)
        fut_cross = pool.submit(compute_cross_trait_tetrachoric, df, seed=seed, pairs=pairs)
        fut_tetra_sex = pool.submit(compute_tetrachoric_by_sex, df, seed=seed, pairs=pairs)

        stats["tetrachoric"] = fut_tetra.result()
        stats["tetrachoric_by_generation"] = fut_tetra_gen.result()
        stats["cross_trait_tetrachoric"] = fut_cross.result()
        stats["tetrachoric_by_sex"] = fut_tetra_sex.result()
    logger.info("All MLE correlations computed in %.1fs", time.perf_counter() - t_mle)

    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        yaml.dump(to_native(stats), fh, default_flow_style=False, sort_keys=False)
    logger.info("Stats written to %s", stats_path)

    sample_df = create_sample(df, seed=seed)
    save_parquet(sample_df, Path(samples_output))
    logger.info("Sample (%d rows) written to %s", len(sample_df), samples_output)


def cli() -> None:
    """Command-line interface for phenotype statistics computation."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Compute phenotype statistics")
    add_logging_args(parser)
    parser.add_argument("phenotype", help="Input phenotype parquet")
    parser.add_argument("censor_age", type=float)
    parser.add_argument("stats_output", help="Output stats YAML")
    parser.add_argument("samples_output", help="Output samples parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-censoring", type=str, default=None, help="Per-generation censoring windows as JSON dict")
    parser.add_argument("--pedigree", default=None, help="Full pedigree parquet for G_ped pair counts")
    parser.add_argument(
        "--max-degree",
        dest="max_degree",
        type=int,
        default=2,
        help="Maximum kinship degree for pair extraction (1-5, default 2)",
    )

    args = parser.parse_args()
    init_logging(args)

    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    main(
        args.phenotype,
        args.censor_age,
        args.stats_output,
        args.samples_output,
        seed=args.seed,
        gen_censoring=gen_censoring,
        pedigree_path=args.pedigree,
        max_degree=args.max_degree,
    )
