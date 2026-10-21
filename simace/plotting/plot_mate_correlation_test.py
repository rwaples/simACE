"""Per-rep z of the ``mate_correlation`` calibrated test, one panel per cell type (ADR 0031).

Each panel is one of liability1, liability2, cross_12 and cross_21, with one
column per parents' generation that some replicate reports, uncapped (the
test's ``cell_names`` stop at generation 11 only for gather). Points, mean
bar and band are those of the calibrated-tests page. Above the panels is the
scenario's verdict from ``validation_summary.yaml``, which the page shows and
never decides.
"""

from __future__ import annotations

__all__ = ["plot_mate_correlation_test"]

from typing import TYPE_CHECKING, Any

from simace.plotting.plot_calibrated_tests import plot_z_panels
from simace.plotting.plot_onset_tau import verdict_note
from simace.plotting.plot_utils import save_placeholder_plot

if TYPE_CHECKING:
    from pathlib import Path

_NAME = "mate_correlation"
_CELL_TYPES = ("liability1", "liability2", "cross_12", "cross_21")


def _generation(cell: str) -> int:
    return int(cell.partition("_")[0].removeprefix("g"))


def plot_mate_correlation_test(
    checks: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    verdict: dict[str, Any] | None = None,
) -> None:
    """Per-rep z of each mate-correlation cell in each rep's ``checks``, one panel per cell type."""
    sections = [stats.get(_NAME) or {} for stats in checks]
    reported = {cell for section in sections for cell in (section.get("cells") or {})}
    panels = []
    for cell_type in _CELL_TYPES:
        cells = sorted((c for c in reported if c.partition("_")[2] == cell_type), key=_generation)
        if cells:
            panels.append((cell_type, cells))
    if not panels:
        reasons = {section["not_applicable"] for section in sections if "not_applicable" in section}
        message = (
            f"Mate correlation test not applicable: {'; '.join(sorted(reasons))}"
            if reasons
            else "Mate correlation test not reported: replicates not yet validated"
        )
        save_placeholder_plot(output_path, message)
        return

    plot_z_panels(
        [
            (sections, cells, [c.partition("_")[0] for c in cells], f"{_NAME}: {cell_type} by parents' generation")
            for cell_type, cells in panels
        ],
        output_path,
        verdict_note(verdict, (_NAME,)),
        scenario,
    )
