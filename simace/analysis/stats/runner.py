"""Orchestration entry point for per-replicate stats reports.

Reads outcomes-only ``trait.parquet`` plus ``pedigree.parquet``, hydrates the
trait rows for computations, and writes ``stats_report.yaml`` plus
``plotting_sample.parquet``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from pedigree_graph import PedigreeGraph

from simace.core.parquet import load_parquet, save_parquet
from simace.core.relationships import DEFAULT_MAX_DEGREE
from simace.core.trait_schema import hydrate_trait
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
    compute_cumulative_incidence_aj,
    compute_cumulative_incidence_aj_by_sex,
    compute_cumulative_incidence_aj_by_sex_generation,
    compute_cumulative_incidence_by_sex,
    compute_cumulative_incidence_by_sex_generation,
    compute_joint_affection,
    compute_mortality,
    compute_prevalence,
    compute_regression,
)
from .pedigree import compute_mean_family_size, compute_parent_status
from .sampling import create_sample

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    type _Frame = pd.DataFrame | pl.DataFrame

logger = logging.getLogger(__name__)

PEDIGREE_REPORT_COLUMNS = ["id", "mother", "father", "twin", "sex", "generation", "liability1", "liability2"]
# Per-trait additive (A), common-environment (C), and unique-environment (E)
# liability components. Stored in the pedigree (the trait file is outcomes-only),
# hydrated onto the plotting sample so the A/C/E component figures render.
LIABILITY_COMPONENT_COLUMNS = ["A1", "A2", "C1", "C2", "E1", "E2"]
REPORT_GROUPS = ("metadata", "incidence", "censoring", "pedigree", "correlations", "heritability")


@dataclass(frozen=True)
class RelationshipContext:
    """Extracted relationship pairs and per-relation full-pedigree pair counts."""

    pairs: dict[str, tuple[Any, Any]]
    full_counts: dict[str, int] | None


def _log_elapsed(label: str, start: float) -> None:
    logger.info("%s completed in %.1fs", label, time.perf_counter() - start)


def _read_pedigree(path: str | None) -> pl.DataFrame | None:
    if path is None:
        return None
    return load_parquet(path, columns=PEDIGREE_REPORT_COLUMNS)


def _same_ordered_ids(left: _Frame, right: _Frame) -> bool:
    """Return True when two frames contain the same ordered ``id`` column."""
    if len(left) != len(right):
        return False
    return bool(np.array_equal(left["id"].to_numpy(), right["id"].to_numpy()))


def _n_unique_generations(df: _Frame) -> int:
    return len(np.unique(df["generation"].to_numpy())) if "generation" in df.columns else 1


def _build_relationship_context(
    df: _Frame,
    df_ped: _Frame | None,
    max_degree: int,
) -> RelationshipContext:
    logger.info("Extracting relationship pairs...")
    # TODO(performance): design streaming/sample-based relationship stats in
    # pedigree-graph/simACE so large-N reports can compute counts and
    # correlations without materializing 100M+ pair arrays. Coordinate with
    # fitACE consumers before changing relationship-extraction semantics.
    t0 = time.perf_counter()
    if df_ped is not None:
        if _same_ordered_ids(df_ped, df):
            # Fast path for the common no-ascertainment case: the phenotype
            # and pedigree tables are the same ordered individuals, so a
            # subsample mask/remap would be pure overhead.
            pg = PedigreeGraph(df)
        else:
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
    return RelationshipContext(pairs=pairs, full_counts=full_counts)


def build_stats_report(
    df: _Frame,
    censor_age: float,
    *,
    seed: int = 42,
    gen_censoring: dict[int, list[float]] | None = None,
    df_ped: _Frame | None = None,
    max_degree: int = DEFAULT_MAX_DEGREE,
    case_ascertainment_ratio: float = 1.0,
) -> dict[str, Any]:
    """Build the grouped per-replicate stats report in memory."""
    report: dict[str, Any] = {group: {} for group in REPORT_GROUPS}

    t0 = time.perf_counter()
    metadata = {
        "n_individuals": len(df),
        "n_generations": _n_unique_generations(df),
    }
    if case_ascertainment_ratio != 1.0:
        metadata["case_ascertainment_ratio"] = case_ascertainment_ratio
    report["metadata"] = metadata
    _log_elapsed("Metadata stats", t0)

    t0 = time.perf_counter()
    incidence = {
        "prevalence": compute_prevalence(df),
        "mortality": compute_mortality(df, censor_age),
        "regression": compute_regression(df),
        "cumulative_incidence": compute_cumulative_incidence(df, censor_age),
        "cumulative_incidence_by_sex": compute_cumulative_incidence_by_sex(df, censor_age),
        "cumulative_incidence_by_sex_generation": compute_cumulative_incidence_by_sex_generation(df, censor_age),
        "cumulative_incidence_aj": compute_cumulative_incidence_aj(df, censor_age, gen_censoring=gen_censoring),
        "cumulative_incidence_aj_by_sex": compute_cumulative_incidence_aj_by_sex(
            df, censor_age, gen_censoring=gen_censoring
        ),
        "cumulative_incidence_aj_by_sex_generation": compute_cumulative_incidence_aj_by_sex_generation(
            df, censor_age, gen_censoring=gen_censoring
        ),
    }
    _log_elapsed("Incidence stats", t0)

    t0 = time.perf_counter()
    censoring = {
        "person_years": compute_person_years(df, censor_age, gen_censoring),
    }
    if gen_censoring is not None:
        censoring["windows"] = compute_censoring_windows(df, censor_age, gen_censoring)
        censoring["confusion"] = compute_censoring_confusion(df, censor_age, gen_censoring)
        censoring["cascade"] = compute_censoring_cascade(df, censor_age, gen_censoring)
    _log_elapsed("Censoring stats", t0)

    t0 = time.perf_counter()
    relationship_context = _build_relationship_context(df, df_ped, max_degree)
    _log_elapsed("Relationship context", t0)

    pairs = relationship_context.pairs
    t0 = time.perf_counter()
    pedigree: dict[str, Any] = {
        "family_size": compute_mean_family_size(df),
        "relationship_pair_counts": {k: len(v[0]) for k, v in pairs.items()},
        "parent_status": compute_parent_status(df, df_ped),
    }
    if df_ped is not None and relationship_context.full_counts is not None:
        pedigree["full"] = {
            "relationship_pair_counts": relationship_context.full_counts,
            "n_individuals": len(df_ped),
            "n_generations": _n_unique_generations(df_ped),
        }
        logger.info(
            "Pedigree pair counts (from same graph): %s",
            ", ".join(f"{k}: {v}" for k, v in relationship_context.full_counts.items()),
        )
    _log_elapsed("Pedigree stats", t0)

    t0 = time.perf_counter()
    correlations = {
        "liability_correlations": compute_liability_correlations(df, seed=seed, pairs=pairs),
        "affected_correlations": compute_affected_correlations(df, seed=seed, pairs=pairs),
        "parent_offspring_corr": compute_parent_offspring_corr(df),
        "parent_offspring_corr_by_sex": compute_parent_offspring_corr_by_sex(df),
        "parent_offspring_affected_corr": compute_parent_offspring_affected_corr(df),
        "joint_affection": compute_joint_affection(df),
    }
    if df_ped is not None:
        logger.info("Computing mate liability correlations...")
        correlations["mate_correlation"] = compute_mate_correlation(df_ped)
    _log_elapsed("Fast correlation stats", t0)

    report["heritability"] = {
        "observed_h2_estimators": compute_observed_h2_estimators(
            correlations["affected_correlations"],
            correlations["parent_offspring_affected_corr"],
        )
    }

    logger.info("Computing tetrachoric correlations in parallel...")
    t_mle = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:
        fut_tetra = pool.submit(compute_tetrachoric, df, seed=seed, pairs=pairs)
        fut_tetra_gen = pool.submit(compute_tetrachoric_by_generation, df, seed=seed, pairs=pairs)
        fut_cross = pool.submit(compute_cross_trait_tetrachoric, df, seed=seed, pairs=pairs)
        fut_tetra_sex = pool.submit(compute_tetrachoric_by_sex, df, seed=seed, pairs=pairs)

        correlations["tetrachoric"] = fut_tetra.result()
        correlations["tetrachoric_by_generation"] = fut_tetra_gen.result()
        correlations["cross_trait_tetrachoric"] = fut_cross.result()
        correlations["tetrachoric_by_sex"] = fut_tetra_sex.result()
    logger.info("All MLE correlations computed in %.1fs", time.perf_counter() - t_mle)

    report["incidence"] = incidence
    report["censoring"] = censoring
    report["pedigree"] = pedigree
    report["correlations"] = correlations
    return report


def main(
    phenotype_path: str,
    censor_age: float,
    stats_output: str,
    samples_output: str,
    seed: int = 42,
    gen_censoring: dict[int, list[float]] | None = None,
    pedigree_path: str | None = None,
    max_degree: int = DEFAULT_MAX_DEGREE,
    case_ascertainment_ratio: float = 1.0,
) -> None:
    """Compute all stats for a single replicate and write outputs."""
    t0 = time.perf_counter()
    df_trait = load_parquet(phenotype_path)
    logger.info("Computing stats for %s (%d rows)", phenotype_path, len(df_trait))
    df_ped = _read_pedigree(pedigree_path)
    df = (
        hydrate_trait(df_trait, df_ped, kind="censored", columns=PEDIGREE_REPORT_COLUMNS)
        if df_ped is not None
        else df_trait
    )
    _log_elapsed("Input load", t0)

    stats = build_stats_report(
        df,
        censor_age,
        seed=seed,
        gen_censoring=gen_censoring,
        df_ped=df_ped,
        max_degree=max_degree,
        case_ascertainment_ratio=case_ascertainment_ratio,
    )
    del df_trait, df_ped

    stats_path = Path(stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(stats_path, "w", encoding="utf-8") as fh:
        yaml.dump(to_native(stats), fh, default_flow_style=False, sort_keys=False)
    logger.info("Stats written to %s", stats_path)
    _log_elapsed("YAML write", t0)

    t0 = time.perf_counter()
    sample_df = create_sample(df, seed=seed)
    save_parquet(sample_df, Path(samples_output))
    logger.info("Plotting sample (%d rows) written to %s", len(sample_df), samples_output)
    _log_elapsed("Plotting sample write", t0)


def cli() -> None:
    """Command-line interface for phenotype statistics computation."""
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging

    parser = argparse.ArgumentParser(description="Build per-replicate stats report")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
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
        default=DEFAULT_MAX_DEGREE,
        help="Maximum kinship degree for pair extraction (0-5, default 3; includes 1C)",
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
