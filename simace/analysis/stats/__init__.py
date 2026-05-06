"""Compute per-rep phenotype statistics for downstream plotting.

Reads a single phenotype.parquet and produces:
  - phenotype_stats.yaml: scalar and array statistics
  - phenotype_samples.parquet: downsampled rows for scatter/histogram plots

Public API is re-exported from focused sub-modules:

- :mod:`.tetrachoric` — tetrachoric primitives
- :mod:`.correlations` — pairwise relationship correlations, parent-offspring
  regressions, observed h² estimators, mate correlation
- :mod:`.incidence` — prevalence, mortality, cumulative incidence, regression,
  joint affection
- :mod:`.censoring` — censoring windows, confusion, cascade, person-years
- :mod:`.pedigree` — family size, parent presence
- :mod:`.sampling` — per-rep downsampling for plots
- :mod:`.runner` — ``main`` and ``cli`` entry point
"""

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
from .effective_size import compute_effective_size, theoretical_expectations
from .incidence import (
    compute_cumulative_incidence,
    compute_cumulative_incidence_by_sex,
    compute_cumulative_incidence_by_sex_generation,
    compute_joint_affection,
    compute_mortality,
    compute_prevalence,
    compute_regression,
)
from .pedigree import compute_mean_family_size, compute_parent_status
from .runner import cli, main
from .sampling import create_sample
from .tetrachoric import tetrachoric_corr, tetrachoric_corr_se

__all__ = [
    "cli",
    "compute_affected_correlations",
    "compute_censoring_cascade",
    "compute_censoring_confusion",
    "compute_censoring_windows",
    "compute_cross_trait_tetrachoric",
    "compute_cumulative_incidence",
    "compute_cumulative_incidence_by_sex",
    "compute_cumulative_incidence_by_sex_generation",
    "compute_effective_size",
    "compute_joint_affection",
    "compute_liability_correlations",
    "compute_mate_correlation",
    "compute_mean_family_size",
    "compute_mortality",
    "compute_observed_h2_estimators",
    "compute_parent_offspring_affected_corr",
    "compute_parent_offspring_corr",
    "compute_parent_offspring_corr_by_sex",
    "compute_parent_status",
    "compute_person_years",
    "compute_prevalence",
    "compute_regression",
    "compute_tetrachoric",
    "compute_tetrachoric_by_generation",
    "compute_tetrachoric_by_sex",
    "create_sample",
    "main",
    "tetrachoric_corr",
    "tetrachoric_corr_se",
    "theoretical_expectations",
]
