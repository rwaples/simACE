"""Quality checks: the check families Analyze runs on every rep's recorded pedigree.

A quality check fails only when the rep is wrong: an invariant, or a
gross-error floor a correct simulation essentially never crosses (ADR 0028).
The same families also compute the realized quantities the report's ``truth``
and ``estimators`` groups carry. Calibrated tests live in
:mod:`simace.analysis.validate` instead.

Public API is re-exported from focused sub-modules (mirrors
:mod:`simace.analysis.stats`):

- :mod:`.structural` — pedigree structural integrity
- :mod:`.twins` — MZ-twin checks
- :mod:`.half_sibs` — half-sib structure + variance-component correlations
- :mod:`.c_transmission` — vertical C-transmission coefficients (ADR 0025)
- :mod:`.consanguinity` — consanguineous-mating reconciliation
- :mod:`.statistical` — founder variances and cross-trait correlations
- :mod:`.heritability` — MZ/DZ, Falconer, parent-offspring
- :mod:`.population` — generation sizes, per-gen stats, family-size dist
- :mod:`.assortative_mating` — mate correlation
- :mod:`.am_equilibrium` — AM additive-variance (Bulmer) equilibrium
- :mod:`.am_relatedness` — AM-aware expected relative correlations and mate correlations
- :mod:`.runner` — ``build_quality_report``
"""

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
from .runner import build_quality_report
from .statistical import validate_statistical
from .structural import validate_structural
from .twins import validate_twins

__all__ = [
    "build_quality_report",
    "compute_family_size_distribution",
    "compute_per_generation_stats",
    "validate_am_equilibrium",
    "validate_assortative_mating",
    "validate_c_transmission",
    "validate_consanguineous_matings",
    "validate_half_sibs",
    "validate_heritability",
    "validate_population",
    "validate_statistical",
    "validate_structural",
    "validate_twins",
]
