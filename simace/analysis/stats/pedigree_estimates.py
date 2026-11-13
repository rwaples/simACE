"""Estimates from one replicate's recorded pedigree: liability correlations, midparent slope, realized h²."""

from __future__ import annotations

__all__ = [
    "POOLED_RELATIONSHIP_CLASSES",
    "load_pedigree_estimates",
    "load_pedigree_estimates_per_generation",
    "load_pooled_liability_correlations",
]

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pg_phenotype import Trait

from simace.analysis.mate_correlation import mate_pedigree
from simace.analysis.midparent import midparent_offspring, stratum_line
from simace.core.numerics import safe_corrcoef
from simace.core.parquet import load_parquet
from simace.core.relationships import DEFAULT_MAX_DEGREE, RELATIONSHIP_TYPES

if TYPE_CHECKING:
    from pathlib import Path

# Display-ordered relationship classes, pooled from the raw classes in
# report.yaml. Expected liability correlation under random mating is
# ``k * A + c * C`` where k is the relatedness 2*kinship; the middle column
# below is k so callers can draw reference bars.  k is sourced from
# ``RELATIONSHIPS[code].nominal_kinship`` (members of a pooled class share one
# kinship value, so the first member is representative) — see ADR 0009.  MZ twins are deliberately
# omitted — their liability correlation is pinned at ``A + C`` regardless of
# AM, so including them washes out the visual story that this plot tells.
POOLED_RELATIONSHIP_CLASSES: tuple[tuple[str, float, tuple[str, ...]], ...] = tuple(
    (name, 2.0 * RELATIONSHIPS[members[0]].nominal_kinship, members)
    for name, members in (
        ("FS", ("FS",)),
        ("PO", ("MO", "FO")),
        ("HS", ("MHS", "PHS")),
        ("1C", ("1C",)),
    )
)


def load_pedigree_estimates(
    pedigree_path: Path,
    trait: int = 1,
    min_generation: int | None = None,
) -> dict[str, float]:
    """Per-rep liability correlations, midparent-PO slope, and realized h².

    Everything is computed on a single replicate's recorded ``pedigree.parquet`` —
    not on pre-aggregated YAML — so that a generation filter can be applied
    cleanly.  When ``min_generation`` is set, all liability correlations
    and the realized h² are computed on the subset of individuals in
    ``generation >= min_generation``.  Pair extraction uses the *full*
    pedigree so that cousins of filtered-subset individuals can still be
    found via their grandparents in earlier generations.  The
    midparent-offspring regression uses offspring in the filtered subset
    but reads parent liabilities from the full pedigree (parents may be in
    earlier generations).

    Args:
        pedigree_path: one recorded ``pedigree.parquet`` path (one replicate).
        trait: 1 or 2.
        min_generation: keep individuals with ``generation >= min_generation``
            (inclusive).  ``None`` keeps every generation.

    Returns:
        Flat dict with keys for each of the seven raw relationship classes
        (``MZ``, ``FS``, ``MO``, ``FO``, ``MHS``, ``PHS``, ``1C``) holding
        liability correlations, plus ``po_slope`` (midparent-offspring
        regression slope), ``realized_h2`` (in the filtered subset), and
        the raw variance components ``vA``, ``vC``, ``vE`` used to compute
        it.  Values are ``NaN`` where there aren't enough pairs.
    """
    cols = [
        "id",
        "sex",
        "mother",
        "father",
        "twin",
        "generation",
        f"A{trait}",
        f"C{trait}",
        f"E{trait}",
        f"liability{trait}",
    ]
    df_full = load_parquet(pedigree_path, columns=cols)
    df = df_full.filter(pl.col("generation") >= min_generation) if min_generation is not None else df_full

    # Examples-page pedigree estimates extract at the analysis default depth.
    # NB: this is not plumbed from analysis.max_degree config — a scenario that
    # raises max_degree above the default still extracts at DEFAULT_MAX_DEGREE here.
    pairs = (
        PedigreeGraph.from_frame(df_full)
        .view(ids=df["id"].to_numpy())
        .relationship_pairs(max_degree=DEFAULT_MAX_DEGREE)
    )
    liab = df[f"liability{trait}"].to_numpy()

    corrs: dict[str, float] = {}
    for ptype in RELATIONSHIP_TYPES:
        idx1, idx2 = pairs[ptype]
        if len(idx1) < 10:
            corrs[ptype] = float("nan")
        else:
            corrs[ptype] = safe_corrcoef(liab[idx1], liab[idx2])

    # Midparent-offspring regression: offspring from the filtered subset (stratum 1),
    # parent liabilities from the full pedigree (parents may be pre-filter).
    in_subset = np.where(np.isin(df_full["id"].to_numpy(), df["id"].to_numpy()), 1.0, np.nan)
    fits = midparent_offspring(
        mate_pedigree(df_full), Trait(df_full[f"liability{trait}"].to_numpy(), "continuous"), in_subset
    )
    fit = stratum_line(fits, 1)
    po_slope = fit.slope if fit is not None and fit.n >= 10 and fit.slope is not None else float("nan")

    vA = float(np.var(df[f"A{trait}"].to_numpy(), ddof=1, dtype=np.float64))
    vC = float(np.var(df[f"C{trait}"].to_numpy(), ddof=1, dtype=np.float64))
    vE = float(np.var(df[f"E{trait}"].to_numpy(), ddof=1, dtype=np.float64))
    total = vA + vC + vE
    realized_h2 = vA / total if total > 0 else float("nan")

    return {**corrs, "po_slope": po_slope, "realized_h2": realized_h2, "vA": vA, "vC": vC, "vE": vE}


def load_pooled_liability_correlations(
    pedigree_paths: list[Path],
    trait: int = 1,
    min_generation: int | None = None,
) -> dict[str, list[float]]:
    """Read per-rep liability correlations pooled to FS/PO/HS/1C.

    Thin aggregator over :func:`load_pedigree_estimates`: pools MO+FO → PO
    and MHS+PHS → HS (equal-weight average) so the chart has four x-axis
    categories instead of seven.

    Args:
        pedigree_paths: one recorded ``pedigree.parquet`` path per replicate of a
            single scenario.
        trait: 1 or 2.
        min_generation: passed through to :func:`load_pedigree_estimates`.

    Returns:
        Dict keyed by pooled class name whose values are lists of per-rep
        correlations.  ``NaN`` entries are skipped rather than propagated.
    """
    out: dict[str, list[float]] = {cls: [] for cls, _, _ in POOLED_RELATIONSHIP_CLASSES}
    for path in pedigree_paths:
        est = load_pedigree_estimates(path, trait=trait, min_generation=min_generation)
        for pooled_name, _kinship, members in POOLED_RELATIONSHIP_CLASSES:
            vals = [est.get(m, float("nan")) for m in members]
            finite = [float(v) for v in vals if np.isfinite(v)]
            if finite:
                out[pooled_name].append(float(np.mean(finite)))
    return out


def load_pedigree_estimates_per_generation(
    pedigree_path: Path,
    trait: int = 1,
    gens: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Per-generation cohort liability correlations and realized h².

    Pairs are extracted once over the whole pedigree and then masked to each
    generation ``g``, so MZ and FS pairs are the cohort's own.  ``FS`` and
    ``MZ`` are separate registry categories and a pair belongs to exactly one
    of them, so the FS block already excludes twins.

    Args:
        pedigree_path: one recorded ``pedigree.parquet`` path (one replicate).
        trait: 1 or 2.
        gens: which generations to compute for; defaults to every generation
            present in the pedigree, founders included.

    Returns:
        Dict ``g -> {MZ, FS, n_MZ, n_FS, realized_h2, vA, vC, vE}``.  Values
        are ``NaN`` (or ``0`` for counts) where there aren't enough pairs.
    """
    cols = [
        "id",
        "sex",
        "mother",
        "father",
        "twin",
        "generation",
        f"A{trait}",
        f"C{trait}",
        f"E{trait}",
        f"liability{trait}",
    ]
    df_full = load_parquet(pedigree_path, columns=cols)
    if gens is None:
        gens = sorted(int(g) for g in np.unique(df_full["generation"].to_numpy()))

    pairs_full = PedigreeGraph.from_frame(df_full).relationship_pairs(max_degree=1)
    gen_arr = df_full["generation"].to_numpy()
    liab_full = df_full[f"liability{trait}"].to_numpy()
    # float64 to match pandas Series.var(ddof=1) precision used previously.
    A_full = df_full[f"A{trait}"].to_numpy().astype(np.float64)
    C_full = df_full[f"C{trait}"].to_numpy().astype(np.float64)
    E_full = df_full[f"E{trait}"].to_numpy().astype(np.float64)

    out: dict[int, dict[str, float]] = {}
    for g in gens:
        gen_mask = gen_arr == g
        if not gen_mask.any():
            out[g] = {
                "MZ": float("nan"),
                "FS": float("nan"),
                "n_MZ": 0,
                "n_FS": 0,
                "realized_h2": float("nan"),
                "vA": float("nan"),
                "vC": float("nan"),
                "vE": float("nan"),
            }
            continue

        cohort_corrs: dict[str, tuple[float, int]] = {}
        # MZ + FS are within-generation by construction; gen_mask[idx1] alone is sufficient.
        for code in ("MZ", "FS"):
            idx1, idx2 = pairs_full[code]
            pair_mask = gen_mask[idx1]
            idx1_g, idx2_g = idx1[pair_mask], idx2[pair_mask]
            n_pairs = len(idx1_g)
            if n_pairs < 10:
                cohort_corrs[code] = (float("nan"), n_pairs)
            else:
                cohort_corrs[code] = (safe_corrcoef(liab_full[idx1_g], liab_full[idx2_g]), n_pairs)
        r_mz, n_mz = cohort_corrs["MZ"]
        r_fs, n_fs = cohort_corrs["FS"]

        vA = float(np.var(A_full[gen_mask], ddof=1))
        vC = float(np.var(C_full[gen_mask], ddof=1))
        vE = float(np.var(E_full[gen_mask], ddof=1))
        total = vA + vC + vE
        realized_h2 = vA / total if total > 0 else float("nan")

        out[g] = {
            "MZ": r_mz,
            "FS": r_fs,
            "n_MZ": n_mz,
            "n_FS": n_fs,
            "realized_h2": realized_h2,
            "vA": vA,
            "vC": vC,
            "vE": vE,
        }
    return out
