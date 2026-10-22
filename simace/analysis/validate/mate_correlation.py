"""Per-generation mate correlations as a calibrated test (ADR 0031).

Run by ``simace validate`` with the scenario parameters from the rep's
``report.yaml``. For each recorded mating iteration
(:func:`~simace.analysis.quality.assortative_mating.recorded_matings`) it
takes the Mating Pairs of parents' generation g, the distinct (mother,
father) of the generation g+1 rows whose parents are both pedigree rows, and
scores the Pearson correlation of the mothers' and fathers' liabilities
against the iteration's nominal target. Cells ``g{g}_liability1``,
``g{g}_liability2``, and ``g{g}_cross_12``, ``g{g}_cross_21`` on iterations
where both traits assort.

The SE is the iid delta-method sandwich over the pairs' standardized
liabilities. Where both traits assort, Metropolis swaps pin the correlation
over all M couples to the target (``simace/simulation/simulate.py:603``, stop
rule at ``:357``), and the recorded pairs are the n couples
``allocate_offspring`` (``:760``) gave a child: every n-subset equally
likely, so the SE takes the finite-population factor √(1 − n/M̂). Where one
trait assorts, the copula's correlation is itself random (``:700``) and the
iid SE stands.

Not applicable, each by design:

- the section, when no recorded iteration assorts on either trait: the
  simulator then shuffles (``:862``) and every target is 0;
- a diagonal cell whose trait does not assort while the other does: its
  correlation is **induced** through ρ_w, and AM moves ρ_w away from the
  configured value its target is built from (ADR 0031 Evidence);
- a cell reading a trait with A, C and E all configured 0, whose liability
  is then constant;
- 1 to ``MIN_PAIRS − 1`` Mating Pairs, too few for a calibrated SE;
- a both-trait cell whose SE at the target, (1 − ρ²)/√n · √(1 − p̂), is
  under 4 × ``METROPOLIS_TOL``: the stopping error (``:357``) does not shrink
  with N and is not in the SE. The gate reads the target, not the rep's own
  SE, so which reps are scored does not depend on their r.
"""

from __future__ import annotations

__all__ = ["CELL_NAMES", "MIN_PAIRS", "n_couples", "run_mate_correlation"]

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq

from simace.analysis.quality.assortative_mating import recorded_matings
from simace.core.numerics import unique_sorted
from simace.simulation.assortment import METROPOLIS_TOL

from .calibrated import information_cell, not_applicable_cell, pearson_influence, scored_cell, section
from .inheritance import parent_rows

if TYPE_CHECKING:
    from collections.abc import Mapping

    from simace.analysis.quality.assortative_mating import RecordedMating

    from .calibrated import RepInputs

logger = logging.getLogger(__name__)

SCOPE = "mating_pairs_by_parents_generation"

#: Fewer Mating Pairs than this leaves the SE uncalibrated (SD z 1.06 at ~97 pairs, 1.03 at ~246; ADR 0031).
MIN_PAIRS = 250

#: Generations with a gather column; later ones are scored and judged without one.
_COLUMN_GENERATIONS = 12

#: Each cell's label and its (mother trait, father trait) index.
_CELLS = (("liability1", (0, 0)), ("liability2", (1, 1)), ("cross_12", (0, 1)), ("cross_21", (1, 0)))

CELL_NAMES = tuple(f"g{g}_{label}" for g in range(_COLUMN_GENERATIONS) for label, _ in _CELLS)


def n_couples(n_pairs: int, n_children: int) -> float:
    """M̂ with M̂·(1 − (1 − 1/M̂)^{n_children}) = n_pairs: the couples before those with no child drop out.

    The left side is the expected number of couples with a child when
    ``n_children`` children fall multinomially on M̂ couples; it rises from
    1 at M̂ = 1 toward ``n_children``. Zero pairs gives 0, one gives 1, and
    ``n_pairs >= n_children`` (every child its own couple) has no finite
    root and gives ``inf``.
    """
    if n_pairs <= 0:
        return 0.0
    if n_pairs >= n_children:
        return math.inf
    if n_pairs == 1:
        return 1.0

    def occupied(m: float) -> float:
        return -m * math.expm1(n_children * math.log1p(-1 / m))

    lo, hi = float(n_pairs), 2.0 * n_pairs
    while occupied(hi) < n_pairs:
        lo, hi = hi, 2 * hi
    return float(brentq(lambda m: occupied(m) - n_pairs, lo, hi, rtol=1e-12))


def _mating_pairs(
    mother: np.ndarray, father: np.ndarray, generation: np.ndarray, g: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Row indices of the mothers and fathers of generation ``g``'s Mating Pairs, and the children's count.

    A Mating Pair is a distinct (mother, father) of the generation ``g + 1``
    rows, both parents pedigree rows (``mother``/``father`` from
    :func:`~.inheritance.parent_rows`, -1 otherwise).
    """
    children = generation == g + 1
    m, f = mother[children], father[children]
    known = (m >= 0) & (f >= 0)
    n_rows = len(mother)
    keys = unique_sorted(m[known] * n_rows + f[known])
    return keys // n_rows, keys % n_rows, int(children.sum())


def _configured_zero(params: Mapping[str, Any], mating: RecordedMating, t: int) -> bool:
    """Whether trait ``t``'s A, C and E are all configured 0 for the parents of ``mating``."""
    c, e = mating.ce[t - 1], mating.ce[t + 1]
    return float(params.get(f"A{t}", 0)) == 0 and c == 0 and e == 0


def _cell(x: np.ndarray, y: np.ndarray, expected: float, couples: tuple[float, float] | None) -> dict[str, Any]:
    """Score the Pearson r of mothers' ``x`` with fathers' ``y`` over one generation's Mating Pairs.

    ``couples`` is ``(M̂, p̂)`` on an iteration where both traits assort, else None.
    """
    n = len(x)
    stats: dict[str, Any] = {"n_pairs": n}
    if n == 0:
        return information_cell("no Mating Pairs where the simulator recorded matings", expected=expected, **stats)
    if n < MIN_PAIRS:
        return not_applicable_cell(f"{n} Mating Pairs < {MIN_PAIRS}: too few for a calibrated SE", **stats)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return information_cell("a liability is not finite", expected=expected, **stats)
    sx, sy = float(x.std()), float(y.std())
    if not (sx > 0 and sy > 0):
        return information_cell("one side's liability is constant", expected=expected, **stats)
    u, v = (x - x.mean()) / sx, (y - y.mean()) / sy
    r = float(np.mean(u * v))
    se = math.sqrt(float(np.sum(pearson_influence(u, v, r) ** 2))) / n
    if couples is not None:
        m_hat, p_hat = couples
        stats.update(M_hat=m_hat, p_hat=p_hat, se_iid=se)
        se *= math.sqrt(1 - p_hat)
    if not se > 0 or not math.isfinite(se):
        return information_cell("SE is not positive", observed=r, expected=expected, **stats)
    if couples is not None:
        # Gate on the SE at the target, not this rep's, so the choice of scored reps does not depend on their r.
        nominal = (1 - expected**2) / math.sqrt(n) * math.sqrt(1 - couples[1])
        if nominal / 4 < METROPOLIS_TOL:
            return not_applicable_cell(
                f"Metropolis tolerance {METROPOLIS_TOL:g} > SE/4 ({nominal / 4:.2g}): its stopping error is not in the SE",
                observed=r,
                expected=expected,
                se=se,
                **stats,
            )
    return scored_cell(r, expected, se, **stats)


def run_mate_correlation(inputs: RepInputs) -> dict[str, Any]:
    """Score each recorded mating iteration's mate correlations against its nominal target.

    Returns:
        ``{"scope", "cells", "summary"}`` with cells ``g{g}_{liability1,
        liability2, cross_12, cross_21}``, or no cells and ``not_applicable``.
    """
    params = inputs.params
    matings = recorded_matings(dict(params))
    if not any(m.assort != (0, 0) for m in matings):
        return section(SCOPE, {}, "no assortative mating in a recorded mating iteration")
    pedigree = inputs.pedigree
    mother, father = parent_rows(pedigree)
    generation = pedigree["generation"].to_numpy()
    liability = np.column_stack([pedigree[f"liability{t}"].to_numpy().astype(np.float64) for t in (1, 2)])
    cells = {}
    for m in matings:
        g = m.parents_generation
        mothers, fathers, n_children = _mating_pairs(mother, father, generation, g)
        couples = None
        if m.cross_scored and len(mothers):
            m_hat = n_couples(len(mothers), n_children)
            couples = (m_hat, len(mothers) / m_hat)
        for label, (i, j) in _CELLS:
            if i != j and not m.cross_scored:
                continue
            name, expected = f"g{g}_{label}", float(m.expected[i, j])
            x, y = liability[mothers, i], liability[fathers, j]
            if i == j and m.assort[i] == 0 and m.assort[1 - i] != 0:
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else math.nan
                cells[name] = not_applicable_cell(
                    f"induced by assortment on trait {2 - i}",
                    observed=r if math.isfinite(r) else None,
                    expected=expected,
                    n_pairs=len(x),
                )
            elif _configured_zero(params, m, i + 1) or _configured_zero(params, m, j + 1):
                cells[name] = not_applicable_cell("A, C and E configured 0", n_pairs=len(x))
            else:
                cells[name] = _cell(x, y, expected, couples)
    if late := [m.parents_generation for m in matings if m.parents_generation >= _COLUMN_GENERATIONS]:
        logger.info(
            "mate_correlation: parents' generations %d-%d are scored without a gather column", late[0], late[-1]
        )
    return section(SCOPE, cells)
