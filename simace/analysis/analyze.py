"""Combined Analyze stage: produce the curated v2 ``report.yaml`` in one job.

Runs three phases sequentially within a single process (ADR 0007/0008), each
freeing its large frame before the next so peak memory is the max of the three
phases rather than their sum (ADR 0006):

1. **Validate** — ground-truth checks on the full, pre-ascertainment recorded
   pedigree (``pedigree.full.parquet`` + ``params.yaml``).
2. **Phenotyped population** — lightweight prevalence summaries on the full
   pre-ascertainment phenotyped rows (``trait.full.parquet``), used to quantify
   ascertainment distortion.
3. **Analysis sample** — descriptive statistics on the post-ascertainment
   subsample (``trait.parquet`` + ``pedigree.parquet``), plus
   ``plotting_sample.parquet``.

These are re-homed into the v2 scientific report (``schema``, ``replicate``,
``inputs``, ``scopes``, ``quality_checks``, ``truth``, ``observed``,
``estimators``) by :mod:`simace.analysis.report`. Dense plot-only arrays go to a
companion ``plot_payload.yaml`` so the report stays scalar-only.
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
from simace.core.relationships import DEFAULT_MAX_DEGREE
from simace.core.trait_schema import hydrate_trait
from simace.core.yaml_io import dump_yaml, load_yaml

from .report import assemble_report
from .stats.incidence import compute_prevalence
from .stats.runner import PEDIGREE_REPORT_COLUMNS, build_stats_report, create_sample
from .validate import build_validation_report

logger = logging.getLogger(__name__)


def _n_generations(df: pd.DataFrame) -> int:
    return int(df["generation"].nunique()) if "generation" in df.columns else 1


def run_analysis(
    *,
    pedigree_full_path: str,
    params_path: str,
    trait_full_path: str,
    trait_path: str,
    pedigree_path: str,
    report_output: str,
    plot_payload_output: str,
    samples_output: str,
    folder: str = "",
    scenario: str = "",
    rep: int = 1,
    seed: int = 42,
    censor_age: float,
    gen_censoring: dict[int, list[float]] | None = None,
    max_degree: int = DEFAULT_MAX_DEGREE,
    case_ascertainment_ratio: float = 1.0,
) -> dict[str, Any]:
    """Run the three Analyze phases in one process and write the v2 report.

    Args:
        pedigree_full_path: Full, pre-ascertainment recorded pedigree parquet.
        params_path: Scenario parameters YAML.
        trait_full_path: Full pre-ascertainment phenotyped rows parquet.
        trait_path: Post-ascertainment (analysis-sample) trait parquet.
        pedigree_path: Post-ascertainment (analysis) pedigree parquet.
        report_output: Output path for the curated ``report.yaml``.
        plot_payload_output: Output path for the dense ``plot_payload.yaml``.
        samples_output: Output path for ``plotting_sample.parquet``.
        folder: Folder name recorded in the report's replicate block.
        scenario: Scenario name recorded in the report's replicate block.
        rep: Replicate number recorded in the report's replicate block.
        seed: Random seed for stats sampling / correlations.
        censor_age: Administrative censoring age.
        gen_censoring: Optional per-generation censoring windows.
        max_degree: Maximum kinship degree for stats pair extraction.
        case_ascertainment_ratio: Configured case-ascertainment ratio.

    Returns:
        The assembled v2 report dict, for in-process callers and tests.
    """
    params = load_yaml(params_path)
    scope_counts: dict[str, Any] = {}

    # --- Phase 1: Validate (full, pre-ascertainment recorded pedigree) ---
    logger.info("Analyze phase 1/3: validating %s", pedigree_full_path)
    df_full = pd.read_parquet(pedigree_full_path)
    validation_report = build_validation_report(df_full, params)
    scope_counts["recorded_pedigree"] = {
        "source": "pedigree.full.parquet",
        "n_individuals": len(df_full),
        "n_generations": _n_generations(df_full),
    }
    del df_full
    gc.collect()

    # --- Phase 2: Phenotyped population (full pre-ascertainment trait rows) ---
    logger.info("Analyze phase 2/3: phenotyped-population summaries on %s", trait_full_path)
    df_trait_full = pd.read_parquet(trait_full_path)
    df_trait_full_ped = pd.read_parquet(pedigree_full_path, columns=["id", "generation"])
    df_trait_full_hydrated = hydrate_trait(df_trait_full, df_trait_full_ped, kind="censored", columns=["generation"])
    prevalence_phenotyped = compute_prevalence(df_trait_full_hydrated)
    scope_counts["phenotyped_population"] = {
        "source": "trait.full.parquet",
        "n_individuals": len(df_trait_full),
        "n_generations": _n_generations(df_trait_full),
    }
    del df_trait_full, df_trait_full_ped, df_trait_full_hydrated
    gc.collect()

    # --- Phase 3: Analysis sample (post-ascertainment subsample) ---
    logger.info("Analyze phase 3/3: stats on %s", trait_path)
    df_trait = pd.read_parquet(trait_path)
    df_ped = pd.read_parquet(pedigree_path, columns=PEDIGREE_REPORT_COLUMNS)
    df = hydrate_trait(df_trait, df_ped, kind="censored", columns=PEDIGREE_REPORT_COLUMNS)
    stats_report = build_stats_report(
        df,
        censor_age,
        seed=seed,
        gen_censoring=gen_censoring,
        df_ped=df_ped,
        max_degree=max_degree,
        case_ascertainment_ratio=case_ascertainment_ratio,
    )
    metadata = stats_report.get("metadata", {})
    sample_n = metadata.get("n_individuals", len(df))
    scope_counts["analysis_sample"] = {
        "source": "trait.parquet",
        "n_individuals": sample_n,
        "n_generations": metadata.get("n_generations", _n_generations(df)),
    }
    pedigree_full = (stats_report.get("pedigree") or {}).get("full") or {}
    pedigree_n = pedigree_full.get("n_individuals", len(df_ped))
    scope_counts["analysis_pedigree"] = {
        "source": "pedigree.parquet",
        "n_individuals": pedigree_n,
        "n_generations": pedigree_full.get("n_generations", _n_generations(df_ped)),
        "ancestor_closure_ratio": (pedigree_n / sample_n) if sample_n else None,
    }
    del df_trait, df_ped

    report, plot_payload = assemble_report(
        replicate={"folder": folder, "scenario": scenario, "rep": rep, "seed": seed},
        params=params,
        case_ascertainment_ratio=case_ascertainment_ratio,
        validation_report=validation_report,
        stats_report=stats_report,
        prevalence_phenotyped=prevalence_phenotyped,
        scope_counts=scope_counts,
    )
    dump_yaml(report, report_output)
    logger.info("Curated report written to %s", report_output)
    dump_yaml(plot_payload, plot_payload_output)
    logger.info("Plot payload written to %s", plot_payload_output)

    sample_df = create_sample(df, seed=seed)
    save_parquet(sample_df, samples_output)
    logger.info("Plotting sample (%d rows) written to %s", len(sample_df), samples_output)

    return report


def cli() -> None:
    """Command-line interface for the combined Analyze stage (debug parity)."""
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging

    parser = argparse.ArgumentParser(description="Run combined Validate + Stats analysis")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--pedigree-full", required=True, help="Full pre-ascertainment pedigree parquet")
    parser.add_argument("--params", required=True, help="Scenario params YAML")
    parser.add_argument("--trait-full", required=True, help="Full pre-ascertainment trait parquet")
    parser.add_argument("--trait", required=True, help="Post-ascertainment trait parquet")
    parser.add_argument("--pedigree", required=True, help="Post-ascertainment pedigree parquet")
    parser.add_argument("--folder", default="", help="Folder name (replicate identity)")
    parser.add_argument("--scenario", default="", help="Scenario name (replicate identity)")
    parser.add_argument("--rep", type=int, default=1, help="Replicate number")
    parser.add_argument("--report-output", required=True, help="Output curated report YAML")
    parser.add_argument("--plot-payload-output", required=True, help="Output dense plot payload YAML")
    parser.add_argument("--samples-output", required=True, help="Output plotting sample parquet")
    parser.add_argument("--censor-age", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-censoring", default=None, help="Per-generation censoring windows as JSON dict")
    parser.add_argument("--max-degree", dest="max_degree", type=int, default=DEFAULT_MAX_DEGREE)
    parser.add_argument("--case-ascertainment-ratio", dest="case_ascertainment_ratio", type=float, default=1.0)

    args = parser.parse_args()
    init_logging(args)

    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    run_analysis(
        pedigree_full_path=args.pedigree_full,
        params_path=args.params,
        trait_full_path=args.trait_full,
        trait_path=args.trait,
        pedigree_path=args.pedigree,
        report_output=args.report_output,
        plot_payload_output=args.plot_payload_output,
        samples_output=args.samples_output,
        folder=args.folder,
        scenario=args.scenario,
        rep=args.rep,
        seed=args.seed,
        censor_age=args.censor_age,
        gen_censoring=gen_censoring,
        max_degree=args.max_degree,
        case_ascertainment_ratio=args.case_ascertainment_ratio,
    )
