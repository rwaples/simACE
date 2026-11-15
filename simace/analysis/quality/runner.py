"""Quality-check orchestration: run every check family Analyze runs on a recorded pedigree."""

from __future__ import annotations

__all__ = ["FAMILIES", "QualityReport", "build_quality_report"]

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simace.analysis.mate_correlation import mate_pedigree
from simace.core.pedigree_arrays import PedigreeArrays

from ._common import mate_correlations, sibling_moments
from .am_equilibrium import validate_am_equilibrium
from .assortative_mating import validate_assortative_mating
from .c_transmission import validate_c_transmission
from .check import check_rows
from .consanguinity import validate_consanguineous_matings
from .half_sibs import validate_half_sibs
from .heritability import validate_heritability
from .population import compute_family_size_distribution, compute_per_generation_stats, validate_population
from .statistical import validate_statistical
from .structural import validate_structural
from .twins import validate_twins

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import PedigreeGraph

    from simace.simulation.params import SimulationParams

    from .check import FamilyResult
    from .consanguinity import ConsanguinityMeasurements
    from .population import PopulationMeasurements
    from .statistical import StatisticalMeasurements
    from .structural import StructuralMeasurements
    from .twins import TwinMeasurements

logger = logging.getLogger(__name__)

#: The check families in report order: ``quality_checks.checks`` lists each family's checks in turn.
FAMILIES = (
    "structural",
    "twins",
    "half_sibs",
    "statistical",
    "c_transmission",
    "heritability",
    "population",
    "assortative_mating",
    "am_equilibrium",
    "consanguineous_matings",
)

#: Families that still return a ``{check name: result dict}`` mapping (#89 converts them next).
_LEGACY = ("half_sibs", "c_transmission", "heritability", "assortative_mating", "am_equilibrium")


def _single(result: dict[str, Any], prefix: str) -> Any:
    """The value of the one key starting with ``prefix`` (or None); raises if there are two."""
    hits = [value for key, value in result.items() if key.startswith(prefix)]
    if len(hits) > 1:
        matched = sorted(k for k in result if k.startswith(prefix))
        raise ValueError(f"Ambiguous {prefix!r} fields in check result: {matched}")
    return hits[0] if hits else None


def _legacy_rows(family: str, results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{family}.{name}",
            "scope": "recorded_pedigree",
            "severity": "error",
            "status": "pass" if result["passed"] else "fail",
            "observed": _single(result, "observed"),
            "expected": _single(result, "expected"),
            "tolerance": result.get("tolerance"),
            "message": result.get("details"),
        }
        for name, result in results.items()
        if "passed" in result
    ]


@dataclass(frozen=True)
class QualityReport:
    """Every check family's result on one recorded pedigree, plus its per-generation and family-size summaries.

    Attributes:
        structural: Pedigree structural integrity.
        twins: MZ twins.
        statistical: Founder variances and cross-trait correlations.
        population: Generation sizes.
        consanguineous_matings: Consanguineous matings and grandparent links.
        legacy: The families not yet typed, by name.
        per_generation: Per-generation liability and component moments, by ``generation_{g}``.
        family_size_distribution: Offspring counts per mother and per father.
    """

    structural: FamilyResult[StructuralMeasurements]
    twins: FamilyResult[TwinMeasurements]
    statistical: FamilyResult[StatisticalMeasurements]
    population: FamilyResult[PopulationMeasurements]
    consanguineous_matings: FamilyResult[ConsanguinityMeasurements]
    legacy: dict[str, dict[str, Any]]
    per_generation: dict[str, dict[str, Any]]
    family_size_distribution: dict[str, Any]

    def rows(self) -> list[dict[str, Any]]:
        """The ``quality_checks.checks`` rows, family by family in ``FAMILIES`` order."""
        rows: list[dict[str, Any]] = []
        for family in FAMILIES:
            if family in _LEGACY:
                rows += _legacy_rows(family, self.legacy[family])
            else:
                result: FamilyResult[Any] = getattr(self, family)
                rows += check_rows(family, result.checks)
        return rows


def build_quality_report(
    df: pd.DataFrame | pl.DataFrame, sim: SimulationParams, graph: PedigreeGraph | None = None
) -> QualityReport:
    """Run every quality-check family on an in-memory pedigree.

    The id-addressable arrays, the sibling relationship moments and the mate
    correlations are derived from ``df`` here. Each failed check is logged.

    Args:
        df: Pedigree DataFrame (full, pre-ascertainment).
        sim: The rep's parameters.
        graph: ``df``'s pedigree graph, when the caller already built one;
            built here otherwise.
    """
    ped = PedigreeArrays.from_frame(df)
    siblings = sibling_moments(df, ped, graph)
    pedigree = mate_pedigree(df)
    mates = mate_correlations(df, pedigree)

    report = QualityReport(
        structural=validate_structural(df, sim, ped),
        twins=validate_twins(df, sim, ped),
        statistical=validate_statistical(df, sim),
        population=validate_population(df, sim),
        consanguineous_matings=validate_consanguineous_matings(df, ped),
        legacy={
            "half_sibs": validate_half_sibs(df, sim, siblings, mates),
            "c_transmission": validate_c_transmission(df, sim.c_transmission, ped),
            "heritability": validate_heritability(df, sim, ped, pedigree, siblings, mates),
            "assortative_mating": validate_assortative_mating(sim, mates),
            "am_equilibrium": validate_am_equilibrium(df, sim),
        },
        per_generation=compute_per_generation_stats(df, sim),
        family_size_distribution=compute_family_size_distribution(df),
    )

    rows = report.rows()
    for row in rows:
        if row["status"] == "fail":
            logger.warning("FAILED %s: %s", row["id"], row["message"])
    n_passed = sum(row["status"] == "pass" for row in rows)
    logger.info("Quality checks complete: %d/%d passed", n_passed, len(rows))
    return report
