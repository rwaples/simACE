"""Validation orchestration: build report, run from disk, CLI entry point."""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

from pedigree_graph import PedigreeGraph

from simace.core.frames import pedigree_graph_input
from simace.core.parquet import load_parquet
from simace.core.pedigree_arrays import PedigreeArrays
from simace.core.yaml_io import dump_yaml, load_yaml

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

from .am_equilibrium import validate_am_equilibrium
from .assortative_mating import validate_assortative_mating
from .consanguinity import validate_consanguineous_matings
from .half_sibs import validate_half_sibs
from .heritability import validate_heritability
from .population import (
    compute_family_size_distribution,
    compute_per_generation_stats,
    validate_population,
)
from .statistical import validate_statistical
from .structural import validate_structural
from .twins import validate_twins

logger = logging.getLogger(__name__)


def build_validation_report(df: pd.DataFrame | pl.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Run all validation checks on an in-memory pedigree and return results.

    Runs structural, twin, half-sibling, statistical, heritability, and
    population checks. The id-addressable arrays and the sibling-pair arrays
    are derived from ``df`` here.

    Args:
        df: Pedigree DataFrame (full, pre-ascertainment).
        params: Scenario parameters.

    Returns:
        Nested dict with keys ``"structural"``, ``"twins"``, ``"half_sibs"``,
        ``"statistical"``, ``"heritability"``, ``"population"``,
        ``"per_generation"``, ``"summary"``, ``"family_size_distribution"``,
        and ``"parameters"``. The ``"summary"`` sub-dict contains
        ``passed`` (bool), ``checks_passed``, ``checks_failed``, and
        ``checks_total`` counts.
    """
    ped = PedigreeArrays.from_frame(df)

    # Validation only needs sibling categories (FS/MHS/PHS); avoid full
    # degree-2 extraction, which also materializes GP/Av pairs.
    full_sib, mat_hs, pat_hs = PedigreeGraph(pedigree_graph_input(df)).sibling_pairs()
    sibling_pairs = {"FS": full_sib, "MHS": mat_hs, "PHS": pat_hs}

    results = {
        "structural": validate_structural(df, params, ped),
        "twins": validate_twins(df, params, ped),
        "half_sibs": validate_half_sibs(df, params, ped, sibling_pairs),
        "statistical": validate_statistical(df, params),
        "heritability": validate_heritability(df, params, ped, sibling_pairs),
        "population": validate_population(df, params),
        "per_generation": compute_per_generation_stats(df, params),
        "assortative_mating": validate_assortative_mating(df, params, ped),
        "am_equilibrium": validate_am_equilibrium(df, params),
        "consanguineous_matings": validate_consanguineous_matings(df, params, ped),
    }

    checks_passed = 0
    checks_failed = 0

    for category, checks in results.items():
        if category == "per_generation":
            continue
        for check_name, check_result in checks.items():
            if "passed" in check_result:  # ty: ignore[unsupported-operator]
                if check_result["passed"]:  # ty: ignore[not-subscriptable]
                    checks_passed += 1
                else:
                    checks_failed += 1
                    logger.warning(
                        "FAILED %s.%s: %s",
                        category,
                        check_name,
                        check_result.get("details", ""),  # ty: ignore[unresolved-attribute]
                    )

    results["summary"] = {
        "passed": checks_failed == 0,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "checks_total": checks_passed + checks_failed,
    }

    results["family_size_distribution"] = compute_family_size_distribution(df, params)
    results["parameters"] = params

    logger.info(
        "Validation complete: %d/%d checks passed",
        checks_passed,
        checks_passed + checks_failed,
    )

    return results


def run_validation(pedigree_path: str, params_path: str) -> dict[str, Any]:
    """Load a pedigree + params from disk and run all validation checks.

    Thin wrapper around :func:`build_validation_report` that reads the inputs.

    Args:
        pedigree_path: Path to the pedigree parquet file.
        params_path: Path to the scenario parameters YAML file.

    Returns:
        The validation report dict (see :func:`build_validation_report`).
    """
    logger.info("Validating pedigree: %s", pedigree_path)
    df = load_parquet(pedigree_path)
    params = load_yaml(params_path)
    return build_validation_report(df, params)


def cli() -> None:
    """Command-line interface for running validation."""
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging

    parser = argparse.ArgumentParser(description="Validate ACE simulation output")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--pedigree", required=True, help="Pedigree parquet path")
    parser.add_argument("--params", required=True, help="Params YAML path")
    parser.add_argument("--output", required=True, help="Output validation YAML path")
    args = parser.parse_args()

    init_logging(args)

    results = run_validation(args.pedigree, args.params)
    dump_yaml(results, args.output)
