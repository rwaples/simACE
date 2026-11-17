"""The one input type of the comparison figures: a scenario's label and its replicate directories."""

from __future__ import annotations

__all__ = ["PEDIGREE", "REPORT", "PrevalenceSeries", "ScenarioReps", "group_reps"]

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from simace.core.yaml_io import load_yaml
from simace.simulation.params import SimulationParams

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Basenames in a replicate directory (``simace.cli.layout.RepArtifact``; not imported, so
#: plotting does not depend on the CLI).
PEDIGREE = "pedigree.parquet"
REPORT = "report.yaml"


@dataclass(frozen=True)
class ScenarioReps:
    """One scenario of a comparison figure.

    Attributes:
        label: The scenario's display label.
        rep_dirs: Its replicate directories (``results/{folder}/{scenario}/rep{N}``),
            each holding the recorded pedigree and the report. At least one.
    """

    label: str
    rep_dirs: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not self.rep_dirs:
            raise ValueError(f"scenario {self.label!r} has no replicate directories")

    @property
    def pedigrees(self) -> list[Path]:
        """Each rep's recorded ``pedigree.parquet``, in rep order."""
        return [d / PEDIGREE for d in self.rep_dirs]

    @property
    def reports(self) -> list[Path]:
        """Each rep's ``report.yaml``, in rep order."""
        return [d / REPORT for d in self.rep_dirs]

    def parameters(self) -> SimulationParams:
        """The parameters of the first rep, parsed from its ``report.yaml``."""
        report = load_yaml(self.reports[0])
        return SimulationParams.from_mapping(report["inputs"]["parameters"])


@dataclass(frozen=True)
class PrevalenceSeries:
    """One line per panel of the prevalence-drift figure: a variant run over every scenario.

    Attributes:
        label: The variant's legend label (e.g. ``standardize=global``).
        scenarios: The variant's run of each panel's scenario, in panel order.
    """

    label: str
    scenarios: tuple[ScenarioReps, ...]


def group_reps(labels: Sequence[str], rep_dirs: Sequence[str | Path], counts: Sequence[int]) -> list[ScenarioReps]:
    """Split a scenario-major flat list of replicate directories into one :class:`ScenarioReps` per label.

    Args:
        labels: One display label per scenario.
        rep_dirs: Every scenario's replicate directories, scenario after scenario.
        counts: How many of *rep_dirs* belong to each scenario, in order.

    Raises:
        ValueError: The labels and counts differ in length, a count is below
            one, or the counts do not add up to the directories given.
    """
    if len(labels) != len(counts):
        raise ValueError(f"{len(labels)} labels but {len(counts)} rep counts")
    if any(n < 1 for n in counts):
        raise ValueError(f"every rep count must be at least 1, got {list(counts)}")
    if sum(counts) != len(rep_dirs):
        raise ValueError(f"rep counts add up to {sum(counts)} but {len(rep_dirs)} directories were given")
    scenarios = []
    start = 0
    for label, n in zip(labels, counts, strict=True):
        scenarios.append(ScenarioReps(label, tuple(Path(d) for d in rep_dirs[start : start + n])))
        start += n
    return scenarios
