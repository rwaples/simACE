"""Realized per-relationship truth of the recorded pedigree under C transmission (ADR 0025).

With C transmission, relatives in different households share part of C and
assortative mating gives A and C a covariance, so the analytic
``2 * kinship * A + C_shared * C`` is exact only for single-path types under
random mating. This module records what the recorded pedigree actually
carries, before ascertainment: per relationship type and trait, the pair
covariances of liability, A and C, the A-C cross covariances, and the
liability correlation, plus the midparent-offspring liability slope. Plots
draw these where the analytic line does not apply.

Pairs are windowed like the report's correlation tables: by the first
member's generation (the offspring for MO and FO), over the last three
generations, pooled and per generation.
"""

from __future__ import annotations

__all__ = ["AM_LIMITATION", "recorded_pedigree_truth"]

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from pedigree_graph import RELATIONSHIPS, PedigreeGraph

from simace.analysis.stats.correlations import report_generations
from simace.analysis.validate.am_relatedness import am_relatedness_mode
from simace.core.pedigree_arrays import PedigreeArrays
from simace.core.relationships import RELATIONSHIP_TYPES, transmission_active

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments

logger = logging.getLogger(__name__)

AM_LIMITATION = (
    "Assortative mating with C transmission correlates mates' C and gives offspring an A-C covariance. "
    "cov_AC and cov_CA record it here; the AM equilibrium check and the rho_w cross-trait term leave it "
    "out, so both are informational for this scenario (ADR 0025)."
)

_COMPONENTS = ("liability", "A", "C")


def _products() -> list[tuple[str, str]]:
    diag = [(f"first.{c}{t}", f"second.{c}{t}") for t in (1, 2) for c in _COMPONENTS]
    cross = [pair for t in (1, 2) for pair in ((f"first.A{t}", f"second.C{t}"), (f"first.C{t}", f"second.A{t}"))]
    return diag + cross


def _cell_entry(cell: RelationshipMoments, t: int) -> dict[str, Any]:
    """``{n_pairs, r, cov_L, cov_A, cov_C, cov_AC, cov_CA}`` for one 0-d cell, None statistics when empty."""
    n = int(cell.counts)
    entry: dict[str, Any] = {"n_pairs": n}
    names = {
        "cov_L": (f"first.liability{t}", f"second.liability{t}"),
        "cov_A": (f"first.A{t}", f"second.A{t}"),
        "cov_C": (f"first.C{t}", f"second.C{t}"),
        "cov_AC": (f"first.A{t}", f"second.C{t}"),
        "cov_CA": (f"first.C{t}", f"second.A{t}"),
    }
    comoment = cell.comoment
    for key, product in names.items():
        entry[key] = float(comoment[cell.products.index(product)]) / n if n else None
    r = float(cell.pearson(*names["cov_L"])) if n else float("nan")
    entry["r"] = None if np.isnan(r) else r
    return entry


def _relationship_truth(moments: RelationshipMoments, window: list[int], t: int) -> dict[str, Any]:
    levels = set(moments.axis("first_generation").levels)
    out: dict[str, Any] = {}
    for code in moments.categories:
        by_category = moments.select(category=[code])
        in_window = by_category.select(first_generation=[g for g in window if g in levels])
        entry = _cell_entry(in_window.sum("category", "first_generation"), t)
        entry["by_generation"] = {
            g: _cell_entry(by_category.select(first_generation=[g]).sum("category", "first_generation"), t)
            for g in window
            if g in levels
        }
        out[code] = entry
    return out


def _slope(child: np.ndarray, midparent: np.ndarray) -> dict[str, Any]:
    n = len(child)
    if n < 3 or np.var(midparent) == 0.0:
        return {"slope": None, "n": n}
    return {"slope": float(np.cov(child, midparent)[0, 1] / np.var(midparent, ddof=1)), "n": n}


def _midparent_truth(df: pd.DataFrame | pl.DataFrame, window: list[int], t: int) -> dict[str, Any]:
    """Cov(L_child, L_midparent) / Var(L_midparent) over children with both parents recorded."""
    ped = PedigreeArrays.from_frame(df)
    mother, father = df["mother"].to_numpy(), df["father"].to_numpy()
    gen = df["generation"].to_numpy()
    liability = df[f"liability{t}"].to_numpy().astype(np.float64)
    kids = np.flatnonzero(ped.contains(mother) & ped.contains(father))
    midparent = (liability[ped.positions(mother[kids])] + liability[ped.positions(father[kids])]) / 2.0
    child, kid_gen = liability[kids], gen[kids]
    in_window = np.isin(kid_gen, window)
    out = _slope(child[in_window], midparent[in_window])
    out["by_generation"] = {g: _slope(child[kid_gen == g], midparent[kid_gen == g]) for g in window}
    return out


def recorded_pedigree_truth(
    df: pd.DataFrame | pl.DataFrame,
    params: dict[str, Any],
    max_degree: int,
    clips: list[dict[str, Any]] | None = None,
    graph: PedigreeGraph | None = None,
) -> dict[str, Any] | None:
    """Per-trait realized relationship moments and midparent slope, or None without C transmission.

    One ``relationship_moments`` pass over the recorded pedigree covers the
    seven correlation types and every type of degree at most ``max_degree``.
    ``clips`` are the simulation's clip records (the pedigree's
    :func:`~simace.core.cohort.read_c_transmission_clips`); None when the
    pedigree carries none. ``graph`` is the recorded pedigree's graph when the
    caller already built one (Analyze phase 1 shares validation's).

    Returns:
        ``{"traits": {"trait{t}": {"relationships": {code: entry},
        "midparent_slope": {...}}}, "limitations": [...],
        "c_transmission_clips": [...]}``, each relationship entry ``{n_pairs,
        r, cov_L, cov_A, cov_C, cov_AC, cov_CA, by_generation}``; the clip
        list only when ``clips`` is not None.
    """
    if not transmission_active(params):
        return None
    t0 = time.perf_counter()
    codes = sorted(set(RELATIONSHIP_TYPES) | {c for c, rel in RELATIONSHIPS.items() if rel.degree <= max_degree})
    values = {f"{c}{t}": df[f"{c}{t}"].to_numpy().astype(np.float64) for t in (1, 2) for c in _COMPONENTS}
    generation = df["generation"].to_numpy()
    moments = (graph or PedigreeGraph.from_frame(df)).relationship_moments(
        categories=codes,
        first={"generation": generation.astype(np.int64)},
        second={},
        values=values,
        products=_products(),
    )
    window = report_generations(df)
    traits = {
        f"trait{t}": {
            "relationships": _relationship_truth(moments, window, t),
            "midparent_slope": _midparent_truth(df, window, t),
        }
        for t in (1, 2)
    }
    am = any(am_relatedness_mode(params, t) != "none" for t in (1, 2))
    logger.info("Recorded-pedigree truth over %d categories in %.1fs", len(codes), time.perf_counter() - t0)
    limitations = [AM_LIMITATION] if am else []
    out: dict[str, Any] = {"traits": traits, "limitations": limitations}
    if clips is not None:
        out["c_transmission_clips"] = clips
        if clips:
            limitations.append(_clip_limitation(clips))
    return out


def _clip_limitation(clips: list[dict[str, Any]]) -> str:
    generations = sorted({clip["generation"] for clip in clips})
    largest = max(clip["excess"] for clip in clips)
    return (
        f"The transmitted C covariance exceeded the configured C in {len(generations)} generation(s) "
        f"({', '.join(map(str, generations))}; negative is burn-in), by at most {largest:.4g} in variance. The "
        "fresh draw was clipped to 0 there, so realized Var(C) sits above the configured total (ADR 0025)."
    )
