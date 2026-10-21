"""Recorded generation 0's calibrated tests: total variance and cross-trait rC (ADR 0030).

Run by ``simace validate`` on the recorded pedigree's generation 0 (the
offspring of iteration ``burnin = G_sim − G_ped``), with the scenario
parameters from the rep's ``report.yaml``:

- :func:`run_total_variance`, per trait: the ddof-0 variance m₂ of
  A + C + E against V₀ = A + C + E[burnin], scored on the relative scale,
  z = (m₂/V₀ − 1) / (SE/m₂);
- :func:`run_cross_trait_rc`: the Pearson r(C1, C2) over every row against rC.

Both SEs are sandwiches clustered on ``household_id``: householdmates share C
and their mother's A contribution. A test's section is ``not_applicable``
under any C transmission; total variance's also under assortative mating up
to burn-in on either trait, and rC's when C1 or C2 is configured 0 or
|rC| = 1. A cell
with fewer than :data:`MIN_HOUSEHOLDS` households, a non-finite estimate, or
no positive SE is information.
"""

from __future__ import annotations

__all__ = [
    "CROSS_TRAIT_RC_CELLS",
    "TOTAL_VARIANCE_CELLS",
    "run_cross_trait_rc",
    "run_total_variance",
]

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from simace.analysis.quality._common import cluster_codes
from simace.core.relationships import transmission_active

from .calibrated import information_cell, not_applicable_cell, scored_cell, summarize

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .calibrated import RepInputs

SCOPE = "recorded_generation_0"

#: Fewer households than this leaves the cluster sandwich unreliable.
MIN_HOUSEHOLDS = 100

TOTAL_VARIANCE_CELLS = ("trait1_total", "trait2_total")
CROSS_TRAIT_RC_CELLS = ("rC",)


def _burnin(params: Mapping[str, Any]) -> tuple[int, int]:
    """``(G_sim, burnin)``: recorded generation 0 is the offspring of iteration ``burnin``."""
    G_sim = int(params.get("G_sim") or params["G_ped"])
    return G_sim, G_sim - int(params["G_ped"])


def _per_gen(params: Mapping[str, Any], name: str, G_sim: int) -> list[float]:
    from simace.simulation.simulate import resolve_per_gen_param

    return resolve_per_gen_param(params.get(name, 0.0), G_sim, name=name, allow_negative=True)


def _assortment(params: Mapping[str, Any]) -> str | None:
    """Why total variance does not apply: assortment in iterations 0..burnin, or None.

    It shifts generation 0's A, and rA carries either trait's into the other.
    Wright-Fisher ignores the assortment parameters. Without C transmission
    the household C draw does not depend on the parents, so rC is unaffected
    (ADR 0030 §4).
    """
    if params.get("mating_model", "standard") != "standard":
        return None
    G_sim, burnin = _burnin(params)
    matrix = params.get("assort_matrix")
    if matrix is not None:
        assort = [[float(matrix[0][0])] * G_sim, [float(matrix[1][1])] * G_sim]
    else:
        assort = [_per_gen(params, f"assort{t}", G_sim) for t in (1, 2)]
    for i in range(burnin + 1):
        if assort[0][i] != 0 or assort[1][i] != 0:
            return f"assortative mating in iteration {i}, up to burn-in ({burnin})"
    return None


def _transmission(params: Mapping[str, Any]) -> str | None:
    """Why neither test applies: C transmission sizes the fresh draw against the transmitted part (ADR 0030 §4)."""
    return "C transmission" if transmission_active(params) else None


def _generation0(pedigree: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, int]:
    """Recorded generation 0, each row's household index, and the number of households."""
    rows = pedigree.filter(pl.col("generation") == 0)
    households = cluster_codes(rows["household_id"].to_numpy())
    return rows, households, int(households.max()) + 1 if len(households) else 0


def _cluster_se(psi: np.ndarray, households: np.ndarray) -> float:
    """``√(Σ_h (Σ_{i∈h} ψ_i)²) / n`` for influence values ``ψ`` summing to 0."""
    return math.sqrt(float(np.sum(np.bincount(households, weights=psi) ** 2))) / len(psi)


def _total_cell(x: np.ndarray, households: np.ndarray, n_households: int, expected: float) -> dict[str, Any]:
    stats = {"n_rows": len(x), "n_households": n_households}
    if expected <= 0:
        return not_applicable_cell("A, C and E configured 0", **stats)
    if n_households < MIN_HOUSEHOLDS:
        return information_cell(f"{n_households} households < {MIN_HOUSEHOLDS}", expected=expected, **stats)
    m2 = float(x.var())
    if not math.isfinite(m2):
        return information_cell("variance is not finite", expected=expected, **stats)
    if m2 <= 0:
        return information_cell("observed variance is zero", observed=m2, expected=expected, **stats)
    se_obs = _cluster_se((x - x.mean()) ** 2 - m2, households)
    # The relative scale: z = (m₂/V₀ − 1) / (SE_obs/m₂), so the studentizing SE does not grow with m₂.
    se = se_obs * expected / m2
    if not se > 0 or not math.isfinite(se):
        return information_cell("SE is not positive", observed=m2, expected=expected, **stats)
    return scored_cell(m2, expected, se, se_observed=se_obs, **stats)


def _rc_cell(
    c1: np.ndarray, c2: np.ndarray, households: np.ndarray, n_households: int, expected: float
) -> dict[str, Any]:
    stats = {"n_rows": len(c1), "n_households": n_households}
    if n_households < MIN_HOUSEHOLDS:
        return information_cell(f"{n_households} households < {MIN_HOUSEHOLDS}", expected=expected, **stats)
    s1, s2 = float(c1.std()), float(c2.std())
    if not (s1 > 0 and s2 > 0):
        return information_cell("C1 or C2 has zero variance", expected=expected, **stats)
    z1, z2 = (c1 - c1.mean()) / s1, (c2 - c2.mean()) / s2
    r = float(np.mean(z1 * z2))
    if not math.isfinite(r):
        return information_cell("correlation is not finite", expected=expected, **stats)
    se = _cluster_se(z1 * z2 - r / 2 * (z1**2 + z2**2), households)
    if not se > 0 or not math.isfinite(se):
        return information_cell("SE is not positive", observed=r, expected=expected, **stats)
    return scored_cell(r, expected, se, **stats)


def _section(cells: Mapping[str, dict[str, Any]], not_applicable: str | None) -> dict[str, Any]:
    section: dict[str, Any] = {"scope": SCOPE, "cells": dict(cells), "summary": summarize(cells)}
    if not_applicable is not None:
        section["not_applicable"] = not_applicable
    return section


def run_total_variance(inputs: RepInputs) -> dict[str, Any]:
    """Score Var(A + C + E) over recorded generation 0 per trait against A + C + E[burnin].

    Returns:
        ``{"scope", "cells", "summary"}`` with cells ``trait1_total`` and
        ``trait2_total``, or no cells and ``not_applicable``.
    """
    params = inputs.params
    if (reason := _transmission(params) or _assortment(params)) is not None:
        return _section({}, reason)
    G_sim, burnin = _burnin(params)
    rows, households, n_households = _generation0(inputs.pedigree)
    cells = {}
    for t in (1, 2):
        expected = float(params[f"A{t}"]) + float(params[f"C{t}"]) + _per_gen(params, f"E{t}", G_sim)[burnin]
        a, c, e = (rows[f"{k}{t}"].to_numpy().astype(np.float64) for k in "ACE")
        x = a + c + e
        cells[f"trait{t}_total"] = _total_cell(x, households, n_households, expected)
    return _section(cells, None)


def run_cross_trait_rc(inputs: RepInputs) -> dict[str, Any]:
    """Score r(C1, C2) over recorded generation 0 against rC.

    Returns:
        ``{"scope", "cells", "summary"}`` with cell ``rC``, or no cells and
        ``not_applicable``.
    """
    params = inputs.params
    reason = _transmission(params)
    rc = float(params.get("rC", 0.0))
    if reason is None and (float(params["C1"]) == 0 or float(params["C2"]) == 0):
        reason = "C1 or C2 configured 0"
    if reason is None and abs(rc) == 1:
        reason = "|rC| = 1"
    if reason is not None:
        return _section({}, reason)
    rows, households, n_households = _generation0(inputs.pedigree)
    c1, c2 = (rows[f"C{t}"].to_numpy().astype(np.float64) for t in (1, 2))
    return _section({"rC": _rc_cell(c1, c2, households, n_households, rc)}, None)
