"""The analysis sample's and analysis pedigree's stats, in the report's scope shape.

:func:`simace.analysis.stats.runner.build_stats_report` returns a
:class:`SampleStats`; the report writes it under ``observed`` (and its
heritability under ``estimators``), and the plotting view reads it back with
:meth:`SampleStats.from_report`. One record serves both ends, so a stat cannot
exist on one side only (ADR 0008).

Field order is the report's key order. Each table is the dict its
``compute_*`` function returns.

``partition_dense`` drops empty tables from ``report.yaml``. An always-computed
table that is absent therefore reads back as ``{}``. The censoring tables are
computed only under per-generation censoring windows, so absent reads back as
``None``. The round trip holds on the serialized form, which cannot tell an
empty table from an absent one.
"""

from __future__ import annotations

__all__ = ["AnalysisPedigreeStats", "AnalysisSampleStats", "SampleStats", "ScopeSize"]

from dataclasses import dataclass, fields
from typing import Any

#: Tables computed only when the rep has per-generation censoring windows.
_CENSORING_FIELDS = frozenset({"censoring_windows", "censoring_confusion", "censoring_cascade"})

type Table = dict[str, Any]


def _merge(base: Table, extra: Table) -> Table:
    """Recursively merge ``extra`` (dense arrays) into a copy of ``base`` (scalars); ``extra`` wins."""
    merged = dict(base)
    for key, value in extra.items():
        existing = merged.get(key)
        merged[key] = _merge(existing, value) if isinstance(existing, dict) and isinstance(value, dict) else value
    return merged


@dataclass(frozen=True)
class ScopeSize:
    """A scope's individual and generation counts (the report's ``scopes`` entry)."""

    n_individuals: int
    n_generations: int


@dataclass(frozen=True)
class AnalysisSampleStats:
    """``observed.analysis_sample``: stats over the post-ascertainment analysis sample."""

    prevalence: Table
    mortality: Table
    regression: Table
    cumulative_incidence: Table
    cumulative_incidence_by_sex: Table
    cumulative_incidence_by_sex_generation: Table
    cumulative_incidence_aj: Table
    cumulative_incidence_aj_by_sex: Table
    cumulative_incidence_aj_by_sex_generation: Table
    person_years: Table
    censoring_windows: Table | None
    censoring_confusion: Table | None
    censoring_cascade: Table | None
    family_size: Table
    relationship_pair_counts: dict[str, int | None]
    liability_correlations: Table
    affected_correlations: Table
    parent_offspring_corr: Table
    parent_offspring_corr_by_sex: Table
    parent_offspring_affected_corr: Table
    tetrachoric: Table
    tetrachoric_by_generation: Table
    tetrachoric_by_sex: Table
    cross_trait_tetrachoric: Table
    joint_affection: Table

    def to_report(self) -> Table:
        """The report block, in key order; a censoring table that was not computed is omitted."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None and f.name in _CENSORING_FIELDS:
                continue
            out[f.name] = value
        return out

    @classmethod
    def from_report(cls, block: Table) -> AnalysisSampleStats:
        """Read a report block (dense arrays already merged back in)."""
        return cls(**{f.name: block.get(f.name, None if f.name in _CENSORING_FIELDS else {}) for f in fields(cls)})


@dataclass(frozen=True)
class AnalysisPedigreeStats:
    """``observed.analysis_pedigree``: stats over the ancestor-closure analysis pedigree."""

    relationship_pair_counts: dict[str, int | None]
    parent_status: Table
    mate_correlation: Table

    def to_report(self) -> Table:
        """The report block, in key order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_report(cls, block: Table) -> AnalysisPedigreeStats:
        """Read a report block."""
        return cls(**{f.name: block.get(f.name, {}) for f in fields(cls)})


@dataclass(frozen=True)
class SampleStats:
    """One rep's descriptive stats over its analysis sample and analysis pedigree.

    Attributes:
        sample_size: The analysis sample's size (``scopes.analysis_sample``).
        pedigree_size: The analysis pedigree's size (``scopes.analysis_pedigree``).
        analysis_sample: ``observed.analysis_sample``.
        analysis_pedigree: ``observed.analysis_pedigree``.
        observed_h2_estimators: ``estimators.heritability.observed_scale``.
    """

    sample_size: ScopeSize
    pedigree_size: ScopeSize
    analysis_sample: AnalysisSampleStats
    analysis_pedigree: AnalysisPedigreeStats
    observed_h2_estimators: Table

    @classmethod
    def from_report(cls, report: Table, plot_payload: Table | None = None) -> SampleStats:
        """Read a v2 report, merging the plot payload's dense arrays back into ``analysis_sample``.

        Without ``plot_payload`` the dense arrays (incidence curves, censoring
        windows) are absent from their tables.
        """
        observed = report["observed"]
        sample = observed.get("analysis_sample", {})
        if plot_payload is not None:
            sample = _merge(sample, plot_payload.get("analysis_sample", {}))
        scopes = report["scopes"]
        return cls(
            sample_size=ScopeSize(**_size(scopes["analysis_sample"])),
            pedigree_size=ScopeSize(**_size(scopes["analysis_pedigree"])),
            analysis_sample=AnalysisSampleStats.from_report(sample),
            analysis_pedigree=AnalysisPedigreeStats.from_report(observed.get("analysis_pedigree", {})),
            observed_h2_estimators=report["estimators"]["heritability"].get("observed_scale", {}),
        )


def _size(scope: Table) -> dict[str, int]:
    return {"n_individuals": scope["n_individuals"], "n_generations": scope["n_generations"]}
