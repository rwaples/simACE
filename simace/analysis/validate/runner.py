"""Validation orchestration: build report, run from disk, CLI entry point."""

import argparse
import logging
from typing import Any

import pandas as pd
from pedigree_graph import PedigreeGraph

from simace.core.yaml_io import dump_yaml, load_yaml

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


def build_validation_report(
    df: pd.DataFrame,
    params: dict[str, Any],
    *,
    df_indexed: pd.DataFrame | None = None,
    sibling_pairs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all validation checks on an in-memory pedigree and return results.

    Runs structural, twin, half-sibling, statistical, heritability, and
    population checks. ``df_indexed`` and ``sibling_pairs`` are derived from
    ``df`` when not supplied; callers that already hold them (e.g. the combined
    Analyze stage) can pass them in to avoid recomputation.

    Args:
        df: Pedigree DataFrame (full, pre-ascertainment).
        params: Scenario parameters.
        df_indexed: ``df`` indexed by ``id``. Defaults to ``df.set_index("id")``.
        sibling_pairs: ``{"FS", "MHS", "PHS"}`` pair arrays. Defaults to
            extracting them from ``df`` via ``PedigreeGraph``.

    Returns:
        Nested dict with keys ``"structural"``, ``"twins"``, ``"half_sibs"``,
        ``"statistical"``, ``"heritability"``, ``"population"``,
        ``"per_generation"``, ``"summary"``, ``"family_size_distribution"``,
        and ``"parameters"``. The ``"summary"`` sub-dict contains
        ``passed`` (bool), ``checks_passed``, ``checks_failed``, and
        ``checks_total`` counts.
    """
    if df_indexed is None:
        df_indexed = df.set_index("id")

    if sibling_pairs is None:
        # Validation only needs sibling categories (FS/MHS/PHS); avoid full
        # degree-2 extraction, which also materializes GP/Av pairs.
        full_sib, mat_hs, pat_hs = PedigreeGraph(df).sibling_pairs()
        sibling_pairs = {"FS": full_sib, "MHS": mat_hs, "PHS": pat_hs}

    results = {
        "structural": validate_structural(df, params),
        "twins": validate_twins(df, params, df_indexed),
        "half_sibs": validate_half_sibs(df, params, df_indexed, sibling_pairs),
        "statistical": validate_statistical(df, params, df_indexed),
        "heritability": validate_heritability(df, params, df_indexed, sibling_pairs),
        "population": validate_population(df, params),
        "per_generation": compute_per_generation_stats(df, params),
        "assortative_mating": validate_assortative_mating(df, params, df_indexed),
        "am_equilibrium": validate_am_equilibrium(df, params, df_indexed),
        "consanguineous_matings": validate_consanguineous_matings(df, params),
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
    df = pd.read_parquet(pedigree_path)
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
