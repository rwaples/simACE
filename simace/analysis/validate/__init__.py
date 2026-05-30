"""ACE simulation validation.

Validates simulation outputs for structural integrity, statistical properties,
and heritability estimates.

Public API is re-exported from focused sub-modules (mirrors
:mod:`simace.analysis.stats`):

- :mod:`.structural` — pedigree structural integrity
- :mod:`.twins` — MZ-twin checks
- :mod:`.half_sibs` — half-sib structure + variance-component correlations
- :mod:`.consanguinity` — consanguineous-mating reconciliation
- :mod:`.statistical` — founder variances and cross-trait correlations
- :mod:`.heritability` — MZ/DZ, Falconer, parent-offspring
- :mod:`.population` — generation sizes, per-gen stats, family-size dist
- :mod:`.assortative_mating` — mate correlation
- :mod:`.effective_size` — Ne observed-vs-expected
- :mod:`.runner` — ``build_validation_report``, ``run_validation``, ``cli``
"""

from .assortative_mating import validate_assortative_mating
from .consanguinity import validate_consanguineous_matings
from .effective_size import validate_effective_size
from .half_sibs import validate_half_sibs
from .heritability import validate_heritability
from .population import (
    compute_family_size_distribution,
    compute_per_generation_stats,
    validate_population,
)
from .runner import build_validation_report, run_validation
from .runner import cli as cli  # re-export: entry point + tests import it
from .statistical import validate_statistical
from .structural import validate_structural
from .twins import validate_twins

__all__ = [
    "build_validation_report",
    "compute_family_size_distribution",
    "compute_per_generation_stats",
    "run_validation",
    "validate_assortative_mating",
    "validate_consanguineous_matings",
    "validate_effective_size",
    "validate_half_sibs",
    "validate_heritability",
    "validate_population",
    "validate_statistical",
    "validate_structural",
    "validate_twins",
]

# ``cli`` is re-exported (entry point + tests import it) but kept out of
# ``__all__`` to match the original module's public surface.
