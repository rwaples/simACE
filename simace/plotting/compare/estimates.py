"""``report.yaml`` readers for the comparison figures.

The estimators that read the recorded pedigree live in
:mod:`simace.analysis.stats.pedigree_estimates`.
"""

from __future__ import annotations

__all__ = ["load_per_gen_prevalence", "load_per_generation", "load_tetrachoric"]

from typing import TYPE_CHECKING

import numpy as np

from simace.core.relationships import RELATIONSHIP_TYPES
from simace.core.yaml_io import load_yaml
from simace.plotting.stats_report import plotting_report_view, report_per_generation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def load_per_generation(
    report_paths: Sequence[Path],
    trait: int = 1,
) -> dict[int, np.ndarray]:
    """Read ``per_generation`` variance components from report.yaml files.

    Args:
        report_paths: one path per replicate (of a single scenario).
        trait: 1 or 2.

    Returns:
        Dict keyed by generation (1-indexed, matching the YAML keys) whose
        values are (n_reps, 4) arrays of columns ``[vA, vC, vE, h2]``.  ``h2``
        is ``vA / (vA + vC + vE)``.  Generations missing from one rep raise
        ``KeyError``; this is intentional since the comparison plot needs a
        consistent gen axis.
    """
    per_rep: list[dict[int, tuple[float, float, float, float]]] = []
    for path in report_paths:
        per_gen = report_per_generation(load_yaml(path))
        rep_dict: dict[int, tuple[float, float, float, float]] = {}
        for gen_key, gen_data in per_gen.items():
            gen = int(str(gen_key).removeprefix("generation_"))
            vA = float(gen_data[f"A{trait}_var"])
            vC = float(gen_data[f"C{trait}_var"])
            vE = float(gen_data[f"E{trait}_var"])
            total = vA + vC + vE
            h2 = vA / total if total > 0 else float("nan")
            rep_dict[gen] = (vA, vC, vE, h2)
        per_rep.append(rep_dict)

    gens = sorted(set().union(*(r.keys() for r in per_rep)))
    out: dict[int, np.ndarray] = {}
    for gen in gens:
        rows = [r[gen] for r in per_rep]
        out[gen] = np.asarray(rows, dtype=float)
    return out


def load_per_gen_prevalence(
    report_paths: Sequence[Path],
    trait: int = 1,
) -> dict[int, list[float]]:
    """Read ``prevalence.by_generation.{N}.trait{trait}`` across replicates.

    Returns dict ``g -> [per-rep prevalence values]``.  Generations are the
    union of those present across reps; missing reps contribute nothing
    to a given generation's list.
    """
    per_gen: dict[int, list[float]] = {}
    for path in report_paths:
        ps = plotting_report_view(load_yaml(path))
        by_gen = ps.sample.prevalence.get("by_generation") or {}
        for g_key, entry in by_gen.items():
            g = int(g_key)
            val = entry.get(f"trait{trait}")
            if val is None:
                continue
            per_gen.setdefault(g, []).append(float(val))
    return per_gen


def load_tetrachoric(report_path: Path, trait: int) -> dict[str, float]:
    """Tetrachoric r on affected status for every relationship code; NaN where the report has none."""
    ps = plotting_report_view(load_yaml(report_path))
    tet = ps.sample.tetrachoric.get(f"trait{trait}", {}) or {}
    out: dict[str, float] = {}
    for code in RELATIONSHIP_TYPES:
        r = (tet.get(code) or {}).get("r")
        out[code] = float(r) if r is not None else float("nan")
    return out
