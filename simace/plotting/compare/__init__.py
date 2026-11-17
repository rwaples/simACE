"""Cross-scenario comparison figures for the Examples docs pages.

The per-scenario atlas answers "what happened in this one run?".  The Examples
docs pages in ``docs/examples/`` ask a different question: "how does the
outcome shift when we dial a knob?".  That needs a plot that overlays two or
more scenarios on the same axes.

Every figure takes its scenarios as :class:`ScenarioReps` (a label and the
replicate directories). The figures are grouped by Examples page:
:mod:`.am`, :mod:`.increasing_e` and :mod:`.observed_scale`, plus the shared
:mod:`.trajectory`. :mod:`._draw` holds the shared drawing primitives and
:mod:`.estimates` the ``report.yaml`` readers; the estimators that read the
recorded pedigree are in :mod:`simace.analysis.stats.pedigree_estimates`.
"""

from __future__ import annotations

__all__ = ["PrevalenceSeries", "ScenarioReps", "group_reps"]

from simace.plotting.compare._inputs import PrevalenceSeries, ScenarioReps, group_reps
