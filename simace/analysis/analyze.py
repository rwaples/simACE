"""Combined Analyze stage: Validate + Stats in one job.

Runs the two analysis halves sequentially within a single process:

1. **Validate** — ground-truth sanity checks on the full, pre-ascertainment
   pedigree (``pedigree.full.parquet`` + ``params.yaml``) → ``validation.yaml``.
2. **Stats** — descriptive statistics on the post-ascertainment subsample
   (``trait.parquet`` + ``pedigree.parquet``) → ``stats_report.yaml`` +
   ``plotting_sample.parquet``.

The two halves read disjoint inputs over different pedigree scopes, so there is
no cross-stage graph/pair sharing here — that efficiency work is deferred (see
ADR 0006). The one concrete benefit is one scheduler job / process / import
instead of two. Phase 1's full-pedigree frame and graph are explicitly freed
before Phase 2 loads its inputs, so peak memory is ``max(validate, stats)``
rather than their sum.
"""

from __future__ import annotations

__all__ = ["cli", "run_analysis"]

import argparse
import gc
import json
import logging
from typing import Any

import pandas as pd

from simace.core.parquet import save_parquet
from simace.core.yaml_io import dump_yaml, load_yaml

from .stats.runner import PEDIGREE_REPORT_COLUMNS, build_stats_report, create_sample
from .validate import build_validation_report

logger = logging.getLogger(__name__)


def run_analysis(
    *,
    pedigree_full_path: str,
    params_path: str,
    trait_path: str,
    pedigree_path: str,
    validation_output: str,
    stats_output: str,
    samples_output: str,
    seed: int = 42,
    censor_age: float,
    gen_censoring: dict[int, list[float]] | None = None,
    max_degree: int = 2,
    case_ascertainment_ratio: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Validate then Stats in one process and write all three artifacts.

    Args:
        pedigree_full_path: Full, pre-ascertainment pedigree parquet (Validate).
        params_path: Scenario parameters YAML (Validate).
        trait_path: Post-ascertainment trait parquet (Stats).
        pedigree_path: Post-ascertainment pedigree parquet (Stats).
        validation_output: Output path for ``validation.yaml``.
        stats_output: Output path for ``stats_report.yaml``.
        samples_output: Output path for ``plotting_sample.parquet``.
        seed: Random seed for stats sampling / correlations.
        censor_age: Administrative censoring age.
        gen_censoring: Optional per-generation censoring windows.
        max_degree: Maximum kinship degree for stats pair extraction.
        case_ascertainment_ratio: Recorded in stats metadata when != 1.0.

    Returns:
        ``(validation_report, stats_report)`` for in-process callers and tests.
    """
    # --- Phase 1: Validate (full, pre-ascertainment pedigree) ---
    logger.info("Analyze phase 1/2: validating %s", pedigree_full_path)
    df_full = pd.read_parquet(pedigree_full_path)
    params = load_yaml(params_path)
    validation_report = build_validation_report(df_full, params)
    dump_yaml(validation_report, validation_output)
    logger.info("Validation written to %s", validation_output)

    # Free the full-pedigree frame (and the graph/pairs that build_validation_report
    # built and has now released) before loading Stats inputs, so peak memory
    # stays at max(validate, stats), not their sum (ADR 0006).
    del df_full
    gc.collect()

    # --- Phase 2: Stats (post-ascertainment subsample) ---
    logger.info("Analyze phase 2/2: stats on %s", trait_path)
    df = pd.read_parquet(trait_path)
    df_ped = pd.read_parquet(pedigree_path, columns=PEDIGREE_REPORT_COLUMNS)

    stats_report = build_stats_report(
        df,
        censor_age,
        seed=seed,
        gen_censoring=gen_censoring,
        df_ped=df_ped,
        max_degree=max_degree,
        case_ascertainment_ratio=case_ascertainment_ratio,
    )
    del df_ped
    dump_yaml(stats_report, stats_output)
    logger.info("Stats written to %s", stats_output)

    sample_df = create_sample(df, seed=seed)
    save_parquet(sample_df, samples_output)
    logger.info("Plotting sample (%d rows) written to %s", len(sample_df), samples_output)

    return validation_report, stats_report


def cli() -> None:
    """Command-line interface for the combined Analyze stage (debug parity)."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Run combined Validate + Stats analysis")
    add_logging_args(parser)
    parser.add_argument("--pedigree-full", required=True, help="Full pre-ascertainment pedigree parquet")
    parser.add_argument("--params", required=True, help="Scenario params YAML")
    parser.add_argument("--trait", required=True, help="Post-ascertainment trait parquet")
    parser.add_argument("--pedigree", required=True, help="Post-ascertainment pedigree parquet")
    parser.add_argument("--validation-output", required=True, help="Output validation YAML")
    parser.add_argument("--stats-output", required=True, help="Output stats YAML")
    parser.add_argument("--samples-output", required=True, help="Output plotting sample parquet")
    parser.add_argument("--censor-age", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-censoring", default=None, help="Per-generation censoring windows as JSON dict")
    parser.add_argument("--max-degree", dest="max_degree", type=int, default=2)
    parser.add_argument("--case-ascertainment-ratio", dest="case_ascertainment_ratio", type=float, default=1.0)

    args = parser.parse_args()
    init_logging(args)

    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    run_analysis(
        pedigree_full_path=args.pedigree_full,
        params_path=args.params,
        trait_path=args.trait,
        pedigree_path=args.pedigree,
        validation_output=args.validation_output,
        stats_output=args.stats_output,
        samples_output=args.samples_output,
        seed=args.seed,
        censor_age=args.censor_age,
        gen_censoring=gen_censoring,
        max_degree=args.max_degree,
        case_ascertainment_ratio=args.case_ascertainment_ratio,
    )
