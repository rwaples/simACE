"""Combined Analyze stage: produce the curated v2 ``report.yaml`` in one job.

Runs three phases sequentially within a single process (ADR 0008), each
freeing its large frame before the next so peak memory is the max of the three
phases rather than their sum (ADR 0008):

1. **Validate** — ground-truth checks on the recorded pedigree
   (``pedigree.parquet`` + ``params.yaml``).
2. **Phenotyped population** — the pre-ascertainment prevalence summaries the
   ``cohort`` stage wrote to ``phenotyped_population.yaml``, used to quantify
   ascertainment distortion.
3. **Analysis sample** — descriptive statistics on the analysis sample and
   analysis pedigree, rebuilt from ``pedigree.parquet`` + ``cohort.parquet``
   by :func:`~simace.core.cohort.selected_views` (ADR 0021), plus
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
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core.cohort import read_cohort, read_pedigree, selected_views
from simace.core.parquet import save_parquet
from simace.core.relationships import DEFAULT_MAX_DEGREE
from simace.core.trait_schema import hydrate_trait
from simace.core.yaml_io import dump_yaml, load_yaml

from .report import assemble_report
from .stats.runner import (
    LIABILITY_COMPONENT_COLUMNS,
    PEDIGREE_REPORT_COLUMNS,
    build_stats_report,
    create_sample,
)
from .validate import build_validation_report

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

logger = logging.getLogger(__name__)


def _n_generations(df: pd.DataFrame | pl.DataFrame) -> int:
    return len(np.unique(df["generation"].to_numpy())) if "generation" in df.columns else 1


def run_analysis(
    *,
    pedigree_path: str,
    params_path: str,
    cohort_path: str,
    phenotyped_population_path: str,
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
        pedigree_path: Recorded pedigree parquet (``pedigree.parquet``).
        params_path: Scenario parameters YAML.
        cohort_path: The rep's ``cohort.parquet``.
        phenotyped_population_path: The ``cohort`` stage's
            ``phenotyped_population.yaml``.
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
    # Record the value this Analyze invocation used. Reports regenerated from
    # older params.yaml files then remain self-describing.
    params["max_degree"] = max_degree
    scope_counts: dict[str, Any] = {}

    # --- Phase 1: Validate (recorded pedigree) ---
    logger.info("Analyze phase 1/3: validating %s", pedigree_path)
    df_full = read_pedigree(pedigree_path)
    validation_report = build_validation_report(df_full, params)
    scope_counts["recorded_pedigree"] = {
        "source": "pedigree.parquet",
        "n_individuals": len(df_full),
        "n_generations": _n_generations(df_full),
    }
    del df_full
    gc.collect()

    # --- Phase 2: Phenotyped population (summary written by the cohort stage) ---
    logger.info("Analyze phase 2/3: phenotyped-population summaries from %s", phenotyped_population_path)
    phenotyped_population = load_yaml(phenotyped_population_path)
    prevalence_phenotyped = phenotyped_population["prevalence"]
    scope_counts["phenotyped_population"] = {
        "source": "phenotyped_population.yaml",
        "n_individuals": phenotyped_population["n_individuals"],
        "n_generations": phenotyped_population["n_generations"],
    }

    # --- Phase 3: Analysis sample (post-ascertainment subsample) ---
    logger.info("Analyze phase 3/3: stats on %s", cohort_path)
    views = selected_views(read_pedigree(pedigree_path, columns=PEDIGREE_REPORT_COLUMNS), read_cohort(cohort_path))
    df_trait, df_ped = views.trait, views.pedigree
    del views
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
        "source": "cohort.parquet (affected1 not null)",
        "n_individuals": sample_n,
        "n_generations": metadata.get("n_generations", _n_generations(df)),
    }
    pedigree_full = (stats_report.get("pedigree") or {}).get("full") or {}
    pedigree_n = pedigree_full.get("n_individuals", len(df_ped))
    scope_counts["analysis_pedigree"] = {
        "source": "pedigree.parquet filtered to cohort.parquet",
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
    # The A/C/E component figures (joint component grid + components-by-generation)
    # need the per-trait liability components, which live in the pedigree rather
    # than the outcomes-only cohort. Hydrate them onto the plotting sample
    # only; the stats `df` above is deliberately left lean.
    components = read_pedigree(pedigree_path, columns=["id", *LIABILITY_COMPONENT_COLUMNS])
    sample_df = sample_df.join(components, on="id", how="left", maintain_order="left")
    save_parquet(sample_df, samples_output)
    logger.info("Plotting sample (%d rows) written to %s", len(sample_df), samples_output)

    return report


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line interface for the combined Analyze stage."""
    from simace.core.cli_base import add_logging_args, add_version_arg, generation_map, init_logging
    from simace.core.publish import publish

    parser = argparse.ArgumentParser(prog=prog, description="Run combined Validate + Stats analysis")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--pedigree", required=True, help="Recorded pedigree parquet (pedigree.parquet)")
    parser.add_argument("--params", required=True, help="Scenario params YAML")
    parser.add_argument("--cohort", required=True, help="The rep's cohort.parquet")
    parser.add_argument("--phenotyped-population", required=True, help="The cohort stage's phenotyped_population.yaml")
    parser.add_argument("--folder", default="", help="Folder name (replicate identity)")
    parser.add_argument("--scenario", default="", help="Scenario name (replicate identity)")
    parser.add_argument("--rep", type=int, default=1, help="Replicate number")
    parser.add_argument("--report-output", required=True, help="Output curated report YAML")
    parser.add_argument("--plot-payload-output", required=True, help="Output dense plot payload YAML")
    parser.add_argument("--samples-output", required=True, help="Output plotting sample parquet")
    parser.add_argument("--censor-age", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gen-censoring", type=generation_map, default=None, help="Per-generation censoring windows as JSON dict"
    )
    parser.add_argument("--max-degree", dest="max_degree", type=int, default=DEFAULT_MAX_DEGREE)
    parser.add_argument("--case-ascertainment-ratio", dest="case_ascertainment_ratio", type=float, default=1.0)

    args = parser.parse_args(argv)
    init_logging(args)

    with publish(args.report_output, args.plot_payload_output, args.samples_output) as (report, payload, samples):
        run_analysis(
            pedigree_path=args.pedigree,
            params_path=args.params,
            cohort_path=args.cohort,
            phenotyped_population_path=args.phenotyped_population,
            report_output=str(report),
            plot_payload_output=str(payload),
            samples_output=str(samples),
            folder=args.folder,
            scenario=args.scenario,
            rep=args.rep,
            seed=args.seed,
            censor_age=args.censor_age,
            gen_censoring=args.gen_censoring or None,
            max_degree=args.max_degree,
            case_ascertainment_ratio=args.case_ascertainment_ratio,
        )
