"""Quality-check orchestration: run every check family Analyze runs on a recorded pedigree."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from simace.analysis.mate_correlation import mate_pedigree
from simace.core.pedigree_arrays import PedigreeArrays

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import PedigreeGraph

from ._common import mate_correlations, sibling_moments
from .am_equilibrium import validate_am_equilibrium
from .assortative_mating import validate_assortative_mating
from .c_transmission import validate_c_transmission
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


def build_quality_report(
    df: pd.DataFrame | pl.DataFrame, params: dict[str, Any], graph: PedigreeGraph | None = None
) -> dict[str, Any]:
    """Run every quality-check family on an in-memory pedigree and return results.

    Runs every quality-check family, then the per-generation and family-size
    summaries. The id-addressable arrays, the sibling relationship
    moments and the mate correlations are derived from ``df`` here.

    Args:
        df: Pedigree DataFrame (full, pre-ascertainment).
        params: Scenario parameters.
        graph: ``df``'s pedigree graph, when the caller already built one;
            built here otherwise.

    Returns:
        Nested dict with one key per check family (``"structural"``,
        ``"twins"``, ``"half_sibs"``, ``"statistical"``, ``"c_transmission"``,
        ``"heritability"``, ``"population"``, ``"assortative_mating"``,
        ``"am_equilibrium"``, ``"consanguineous_matings"``), plus
        ``"per_generation"``, ``"summary"``, ``"family_size_distribution"``
        and ``"parameters"``. ``"summary"`` holds ``passed`` (bool),
        ``checks_passed``, ``checks_failed`` and ``checks_total`` over every
        family.
    """
    ped = PedigreeArrays.from_frame(df)
    siblings = sibling_moments(df, ped, graph)
    pedigree = mate_pedigree(df)
    mates = mate_correlations(df, pedigree)

    results = {
        "structural": validate_structural(df, params, ped),
        "twins": validate_twins(df, params, ped),
        "half_sibs": validate_half_sibs(df, params, siblings, mates),
        "statistical": validate_statistical(df, params),
        "c_transmission": validate_c_transmission(df, params, ped),
        "heritability": validate_heritability(df, params, ped, pedigree, siblings, mates),
        "population": validate_population(df, params),
        "per_generation": compute_per_generation_stats(df, params),
        "assortative_mating": validate_assortative_mating(params, mates, df),
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
        "Quality checks complete: %d/%d passed",
        checks_passed,
        checks_passed + checks_failed,
    )

    return results
