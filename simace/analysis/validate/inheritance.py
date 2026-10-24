"""Sibling inheritance calibrated tests: residuals of the simulator's inheritance (ADR 0029).

Run by ``simace validate`` on the recorded pedigree, with the scenario
parameters from the rep's ``report.yaml``:

- :func:`run_mendelian_residual`, per trait with A > 0: the Pearson
  correlation of the **Mendelian residual** A − midparent over FS, MHS and
  PHS pairs (expected 0), and its variance over the non-founders with one
  member of each MZ pair (expected A/2, the configured founder variance).
  When the pedigree carries the simulation's inbreeding coefficients ``F``
  (ADR 0033), the variance is of the residual divided by √(1 − F̄), F̄ the
  parents' mean F, which restores the expectation A/2;
- :func:`run_household_draw`, per trait with C > 0: the Pearson correlation
  of the **household draw** C − b_m·C_mother − b_f·C_father over PHS pairs
  (expected 0), with the share of those pairs whose two households share
  more than one father, a dependence the SE does not see (ADR 0029 §3).

A residual is defined where both parents are pedigree rows. The
correlations take pg-phenotype's MRCA-or-shared-member sandwich SE; the
variance takes the asymptotic iid SE √((m₄ − s⁴)/n). A test whose
component is configured 0 for both traits records ``not_applicable`` for the
whole section and drops out of the verdict. A cell whose residual
has zero variance to within float32 rounding is not applicable; one with
fewer than :data:`MIN_PAIRS` pairs (residuals for the variance), or no SE,
is information.
"""

from __future__ import annotations

__all__ = [
    "HOUSEHOLD_DRAW_CELLS",
    "MENDELIAN_RESIDUAL_CELLS",
    "linked_household_share",
    "parent_rows",
    "run_household_draw",
    "run_mendelian_residual",
]

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .calibrated import information_cell, not_applicable_cell, scored_cell, section

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .calibrated import RepInputs

SCOPE = "recorded_pedigree"

#: Fewer pairs (residuals for the variance cell) than this leaves the normal approximation unreliable.
MIN_PAIRS = 100

#: A residual whose SD is below this share of its component's SD is zero to within float32 rounding.
ZERO_SD_SHARE = 1e-5

_SIBLINGS = ("FS", "MHS", "PHS")

MENDELIAN_RESIDUAL_CELLS = tuple(f"trait{t}_{cell}" for t in (1, 2) for cell in (*_SIBLINGS, "variance"))
HOUSEHOLD_DRAW_CELLS = ("trait1_PHS", "trait2_PHS")


def parent_rows(pedigree: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each row's mother and father row, -1 where the parent is unknown or not a row."""
    ids = pedigree["id"].to_numpy().astype(np.int64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    def lookup(parent: np.ndarray) -> np.ndarray:
        parent = parent.astype(np.int64)
        at = np.clip(np.searchsorted(sorted_ids, parent), 0, len(ids) - 1)
        return np.where((parent >= 0) & (sorted_ids[at] == parent), order[at], -1)

    return lookup(pedigree["mother"].to_numpy()), lookup(pedigree["father"].to_numpy())


def linked_household_share(pedigree: pl.DataFrame) -> float:
    """Share of PHS pairs whose two households have children by more than one common father."""
    per_father = pedigree.filter(pl.col("father") >= 0).group_by("father", "household_id").agg(k=pl.len())
    pairs = (
        per_father.join(per_father, on="father")
        .filter(pl.col("household_id") < pl.col("household_id_right"))
        .with_columns(pairs=pl.col("k") * pl.col("k_right"))
        .group_by("household_id", "household_id_right")
        .agg(fathers=pl.len(), pairs=pl.col("pairs").sum())
    )
    total = int(pairs["pairs"].sum())
    linked = int(pairs.filter(pl.col("fathers") > 1)["pairs"].sum())
    return linked / total if total else 0.0


def _residual(
    pedigree: pl.DataFrame, column: str, mother: np.ndarray, father: np.ndarray, b_m: float, b_f: float
) -> np.ndarray:
    """``x − b_m·x_mother − b_f·x_father`` in float64, NaN where a parent is not a row."""
    x = pedigree[column].to_numpy().astype(np.float64)
    both = (mother >= 0) & (father >= 0)
    out = np.full(len(x), np.nan)
    out[both] = x[both] - b_m * x[mother[both]] - b_f * x[father[both]]
    return out


def _degenerate(residual: np.ndarray, component: np.ndarray) -> str | None:
    """The not-applicable reason when the residual is zero to within rounding of its component."""
    defined = residual[~np.isnan(residual)]
    if len(defined) < 2:
        return None
    if float(defined.std()) <= ZERO_SD_SHARE * float(component.std()):
        return "residual has zero variance (to within float32 rounding)"
    return None


def _correlation_cells(
    pedigree: pl.DataFrame, residual: np.ndarray, categories: Sequence[str], not_applicable: str | None
) -> dict[str, dict[str, Any]]:
    """Each category's Pearson correlation of the residual against 0, by category code."""
    if not_applicable is not None:
        return {code: not_applicable_cell(not_applicable) for code in categories}
    from pg_phenotype import Trait
    from pg_phenotype.relative_pairs import correlation

    found = correlation(pedigree, Trait(residual, kind="continuous"), categories=list(categories), se=True)
    cells = {}
    for cell in found.cells:
        est = cell.crude[0]
        stats = {"n_pairs": cell.n}
        if cell.n < MIN_PAIRS:
            reason = f"{cell.n} pairs < {MIN_PAIRS}"
        elif est.value is None:
            reason = est.reason or "not computed"
        elif est.se is None or not est.se > 0:
            reason = est.se_reason or "no SE"
        else:
            cells[cell.category] = scored_cell(float(est.value), 0.0, float(est.se), **stats)
            continue
        cells[cell.category] = information_cell(reason, observed=est.value, expected=0.0, se=est.se, **stats)
    return cells


def _variance_cell(residual: np.ndarray, keep: np.ndarray, expected: float) -> dict[str, Any]:
    """The residual's variance over ``keep`` against ``expected``, with the asymptotic iid SE √((m₄ − s⁴)/n)."""
    x = residual[keep & ~np.isnan(residual)]
    n = len(x)
    if n < MIN_PAIRS:
        return information_cell(f"{n} residuals < {MIN_PAIRS}", n_residuals=n, expected=expected)
    s2 = float(x.var(ddof=1))
    se = math.sqrt(max(float(np.mean((x - x.mean()) ** 4)) - s2**2, 0.0) / n)
    if not se > 0:
        return information_cell("SE is zero", n_residuals=n, observed=s2, expected=expected)
    return scored_cell(s2, expected, se, n_residuals=n)


def _section(cells: Mapping[str, dict[str, Any]], component: str) -> dict[str, Any]:
    """The rep's section; with no trait carrying ``component``, the test does not apply to the scenario."""
    return section(SCOPE, cells, None if cells else f"{component} is configured 0 for both traits")


def run_mendelian_residual(inputs: RepInputs) -> dict[str, Any]:
    """Score the Mendelian residual's sibling correlations and variance per trait with A > 0.

    Returns:
        ``{"scope", "cells", "summary"}`` with cells ``trait{t}_FS``,
        ``trait{t}_MHS``, ``trait{t}_PHS`` and ``trait{t}_variance``.
    """
    pedigree, params = inputs.pedigree, inputs.params
    mother, father = parent_rows(pedigree)
    ids, twin = pedigree["id"].to_numpy(), pedigree["twin"].to_numpy()
    one_per_mz = (twin < 0) | (ids < twin)
    scale = _mendelian_scale(pedigree, mother, father)
    cells: dict[str, dict[str, Any]] = {}
    for t in (1, 2):
        a = float(params[f"A{t}"])
        if a <= 0:
            continue
        residual = _residual(pedigree, f"A{t}", mother, father, 0.5, 0.5)
        not_applicable = _degenerate(residual, pedigree[f"A{t}"].to_numpy().astype(np.float64))
        for code, cell in _correlation_cells(pedigree, residual, _SIBLINGS, not_applicable).items():
            cells[f"trait{t}_{code}"] = cell
        cells[f"trait{t}_variance"] = (
            not_applicable_cell(not_applicable)
            if not_applicable is not None
            else _variance_cell(residual / scale, one_per_mz, a / 2)
        )
    return _section(cells, "A")


def _mendelian_scale(pedigree: pl.DataFrame, mother: np.ndarray, father: np.ndarray) -> np.ndarray | float:
    """Each row's √(1 − F̄) from the parents' ``F`` (ADR 0033), or 1 for a pedigree without the column (mode off)."""
    if "F" not in pedigree.columns:
        return 1.0
    F = pedigree["F"].to_numpy().astype(np.float64)
    both = (mother >= 0) & (father >= 0)
    scale = np.ones(len(F))
    scale[both] = np.sqrt(1.0 - (F[mother[both]] + F[father[both]]) / 2)
    return scale


def run_household_draw(inputs: RepInputs) -> dict[str, Any]:
    """Score the household draw's PHS correlation per trait with C > 0.

    Returns:
        ``{"scope", "cells", "summary"}`` with cells ``trait{t}_PHS``, each
        carrying ``linked_household_share``.
    """
    pedigree, params = inputs.pedigree, inputs.params
    mother, father = parent_rows(pedigree)
    cells: dict[str, dict[str, Any]] = {}
    linked = None
    for t in (1, 2):
        if float(params[f"C{t}"]) <= 0:
            continue
        b_m = float(params.get(f"Ctm_maternal{t}", 0.0))
        b_f = float(params.get(f"Ctm_paternal{t}", 0.0))
        draw = _residual(pedigree, f"C{t}", mother, father, b_m, b_f)
        not_applicable = _degenerate(draw, pedigree[f"C{t}"].to_numpy().astype(np.float64))
        if linked is None:
            linked = linked_household_share(pedigree)
        cell = _correlation_cells(pedigree, draw, ("PHS",), not_applicable)["PHS"]
        cells[f"trait{t}_PHS"] = {**cell, "linked_household_share": linked}
    return _section(cells, "C")
