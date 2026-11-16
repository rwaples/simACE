"""The analysis sample's stats, for Analyze's ``observed`` and ``estimators`` groups.

:func:`build_stats_report` computes them from the hydrated analysis-sample
frame and its analysis pedigree (Analyze rebuilds both from
``pedigree.parquet`` and ``cohort.parquet``) and returns a
:class:`~simace.analysis.sample_stats.SampleStats`.
"""

from __future__ import annotations

import logging
import operator
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pedigree_graph import MAX_DEGREE, RELATIONSHIPS, PedigreeGraph

from simace.analysis.mate_correlation import mate_pedigree
from simace.analysis.prevalence import compute_prevalence
from simace.analysis.sample_stats import AnalysisPedigreeStats, AnalysisSampleStats, SampleStats, ScopeSize
from simace.core.relationships import DEFAULT_MAX_DEGREE, RELATIONSHIP_TYPES

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
from .incidence import (
    compute_cumulative_incidence,
    compute_cumulative_incidence_aj,
    compute_cumulative_incidence_aj_by_sex,
    compute_cumulative_incidence_aj_by_sex_generation,
    compute_cumulative_incidence_by_sex,
    compute_cumulative_incidence_by_sex_generation,
    compute_joint_affection,
    compute_mortality,
    compute_regression,
)
from .moments import Stratum, relationship_moments_for
from .pedigree import compute_mean_family_size, compute_parent_status
from .sampling import create_sample

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments

    type _Frame = pd.DataFrame | pl.DataFrame

__all__ = [
    "LIABILITY_COMPONENT_COLUMNS",
    "PEDIGREE_REPORT_COLUMNS",
    "RelationshipContext",
    "build_stats_report",
    "create_sample",
]

logger = logging.getLogger(__name__)

PEDIGREE_REPORT_COLUMNS = ["id", "mother", "father", "twin", "sex", "generation", "liability1", "liability2"]
# Per-trait additive (A), common-environment (C), and unique-environment (E)
# liability components. Stored in the pedigree (the trait file is outcomes-only),
# hydrated onto the plotting sample so the A/C/E component figures render.
LIABILITY_COMPONENT_COLUMNS = ["A1", "A2", "C1", "C2", "E1", "E2"]


@dataclass(frozen=True)
class RelationshipContext:
    """Relationship moments for the correlations and per-relation pair counts.

    ``moments`` covers only the ``RELATIONSHIP_TYPES`` codes the correlations
    consume. ``counts`` (analysis sample) come from the same moments pass, and
    ``full_counts`` (analysis pedigree) from that pass too when the sample is
    the whole pedigree, else from the count-only path. Both carry all 23
    registry codes; a code outside ``max_degree`` is ``None`` (not computed),
    never ``0``.
    """

    moments: RelationshipMoments
    counts: dict[str, int | None]
    full_counts: dict[str, int | None]


def _log_elapsed(label: str, start: float) -> None:
    logger.info("%s completed in %.1fs", label, time.perf_counter() - start)


def _same_ordered_ids(left: _Frame, right: _Frame) -> bool:
    """Return True when two frames contain the same ordered ``id`` column."""
    if len(left) != len(right):
        return False
    return bool(np.array_equal(left["id"].to_numpy(), right["id"].to_numpy()))


def _build_relationship_context(df: _Frame, df_ped: _Frame, max_degree: int) -> RelationshipContext:
    if isinstance(max_degree, bool):
        raise TypeError("max_degree must be an integer, not bool")
    if not 0 <= operator.index(max_degree) <= MAX_DEGREE:
        raise ValueError(f"max_degree must be in [0, {MAX_DEGREE}], got {max_degree!r}")
    t0 = time.perf_counter()
    graph = PedigreeGraph.from_frame(df_ped)
    _log_elapsed("Analysis pedigree graph build", t0)
    # Fast path when the sample is the whole analysis pedigree (no
    # ascertainment and no unphenotyped ancestors, G_ped == G_pheno): the two
    # tables are the same ordered individuals, so a view mask/remap would be
    # pure overhead.
    source = graph if _same_ordered_ids(df_ped, df) else graph.view(ids=df["id"].to_numpy())
    counted = [code for code, category in RELATIONSHIPS.items() if category.degree <= max_degree]
    t0 = time.perf_counter()
    # One engine pass serves both the correlation moments and the pair counts.
    every = relationship_moments_for(df, source, categories=[*RELATIONSHIP_TYPES, *counted])
    stratum = Stratum(every)
    n_pairs = dict(zip(stratum.categories, stratum.n_pairs.tolist(), strict=True))
    moments = every.select(category=[code for code in every.categories if code in RELATIONSHIP_TYPES])
    counts = {code: n_pairs[code] if code in counted else None for code in RELATIONSHIPS}
    logger.info(
        "Relationship moments computed in %.1fs: %s",
        time.perf_counter() - t0,
        ", ".join(f"{code}: {n_pairs[code]}" for code in RELATIONSHIP_TYPES),
    )
    if source is graph:
        full_counts = counts
    else:
        t0 = time.perf_counter()
        full_counts = dict(graph.relationship_counts(max_degree=max_degree))
        _log_elapsed("Full-pedigree pair counts", t0)
    return RelationshipContext(moments=moments, counts=counts, full_counts=full_counts)


def build_stats_report(
    df: _Frame,
    df_ped: _Frame,
    censor_age: float,
    *,
    gen_censoring: dict[int, list[float]] | None = None,
    max_degree: int = DEFAULT_MAX_DEGREE,
) -> SampleStats:
    """The stats of an analysis sample ``df`` and its analysis pedigree ``df_ped``.

    Args:
        df: The analysis sample, trait rows hydrated with ``PEDIGREE_REPORT_COLUMNS``.
        df_ped: The analysis pedigree: every cohort member (ancestor closure).
        censor_age: Administrative censoring age.
        gen_censoring: Per-generation censoring windows; the censoring
            windows, confusion and cascade tables are computed only with them.
        max_degree: Deepest relationship degree in the pair-count tables.
    """
    t0 = time.perf_counter()
    incidence = {
        "prevalence": compute_prevalence(df),
        "mortality": compute_mortality(df, censor_age),
        "regression": compute_regression(df),
        "cumulative_incidence": compute_cumulative_incidence(df, censor_age),
        "cumulative_incidence_by_sex": compute_cumulative_incidence_by_sex(df, censor_age),
        "cumulative_incidence_by_sex_generation": compute_cumulative_incidence_by_sex_generation(df, censor_age),
        "cumulative_incidence_aj": compute_cumulative_incidence_aj(df, censor_age, gen_censoring=gen_censoring),
        "cumulative_incidence_aj_by_sex": compute_cumulative_incidence_aj_by_sex(
            df, censor_age, gen_censoring=gen_censoring
        ),
        "cumulative_incidence_aj_by_sex_generation": compute_cumulative_incidence_aj_by_sex_generation(
            df, censor_age, gen_censoring=gen_censoring
        ),
    }
    _log_elapsed("Incidence stats", t0)

    t0 = time.perf_counter()
    person_years = compute_person_years(df, censor_age, gen_censoring)
    if gen_censoring is not None:
        windows = compute_censoring_windows(df, censor_age, gen_censoring)
        confusion = compute_censoring_confusion(df, censor_age, gen_censoring)
        cascade = compute_censoring_cascade(df, censor_age, gen_censoring)
    else:
        windows = confusion = cascade = None
    _log_elapsed("Censoring stats", t0)

    t0 = time.perf_counter()
    relationships = _build_relationship_context(df, df_ped, max_degree)
    _log_elapsed("Relationship context", t0)
    moments = relationships.moments
    logger.info(
        "Analysis pedigree pair counts: %s",
        ", ".join(f"{k}: {v}" for k, v in relationships.full_counts.items() if v is not None),
    )

    t0 = time.perf_counter()
    po_pedigree = mate_pedigree(df)
    affected_correlations = compute_affected_correlations(moments=moments)
    parent_offspring_affected_corr = compute_parent_offspring_affected_corr(df, po_pedigree)
    fast = {
        "liability_correlations": compute_liability_correlations(moments=moments),
        "parent_offspring_corr": compute_parent_offspring_corr(df, po_pedigree),
        "parent_offspring_corr_by_sex": compute_parent_offspring_corr_by_sex(df, po_pedigree),
        "joint_affection": compute_joint_affection(df),
    }
    logger.info("Computing mate liability correlations...")
    mate_correlation = compute_mate_correlation(df_ped)
    _log_elapsed("Fast correlation stats", t0)

    t_mle = time.perf_counter()
    tetrachoric = {
        "tetrachoric": compute_tetrachoric(moments=moments),
        "tetrachoric_by_generation": compute_tetrachoric_by_generation(df, moments=moments),
        "cross_trait_tetrachoric": compute_cross_trait_tetrachoric(df, moments=moments),
        "tetrachoric_by_sex": compute_tetrachoric_by_sex(moments=moments),
    }
    _log_elapsed("Tetrachoric correlation stats", t_mle)

    return SampleStats(
        sample_size=ScopeSize.of(df),
        pedigree_size=ScopeSize.of(df_ped),
        analysis_sample=AnalysisSampleStats(
            **incidence,
            person_years=person_years,
            censoring_windows=windows,
            censoring_confusion=confusion,
            censoring_cascade=cascade,
            family_size=compute_mean_family_size(df),
            relationship_pair_counts=relationships.counts,
            affected_correlations=affected_correlations,
            parent_offspring_affected_corr=parent_offspring_affected_corr,
            **fast,
            **tetrachoric,
        ),
        analysis_pedigree=AnalysisPedigreeStats(
            relationship_pair_counts=relationships.full_counts,
            parent_status=compute_parent_status(df, df_ped),
            mate_correlation=mate_correlation,
        ),
        observed_h2_estimators=compute_observed_h2_estimators(affected_correlations, parent_offspring_affected_corr),
    )
