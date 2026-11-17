"""Estimates from one replicate's recorded pedigree: liability correlations, midparent slope, realized h²."""

from __future__ import annotations

__all__ = [
    "POOLED_RELATIONSHIP_CLASSES",
    "CohortEstimates",
    "H2Reps",
    "PedigreeEstimates",
    "load_naive_estimator_h2",
    "load_observed_vs_liability_h2",
    "load_pedigree_estimates",
    "load_pedigree_estimates_per_generation",
    "load_pooled_liability_correlations",
    "load_sib_pair_liabilities",
]

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pg_phenotype import Trait

from simace.analysis.mate_correlation import mate_pedigree
from simace.analysis.midparent import midparent_offspring, stratum_line
from simace.analysis.stats.naive_h2 import NaiveH2, falconer_h2, naive_h2, pooled_mean
from simace.core.numerics import safe_corrcoef
from simace.core.parquet import load_parquet
from simace.core.relationships import DEFAULT_MAX_DEGREE, RELATIONSHIP_TYPES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
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


@dataclass(frozen=True)
class PedigreeEstimates:
    """One replicate's liability correlations, midparent slope and realized h².

    Attributes:
        r: Liability correlation per relationship code (every code in
            ``RELATIONSHIP_TYPES``); NaN below ten pairs.
        po_slope: Midparent-offspring regression slope; NaN below ten children.
        realized_h2: ``vA / (vA + vC + vE)`` in the same individuals; NaN at
            zero total variance.
    """

    r: Mapping[str, float]
    po_slope: float
    realized_h2: float


@dataclass(frozen=True)
class CohortEstimates:
    """One generation's MZ and FS liability correlations (NaN below ten pairs) and realized h²."""

    r_mz: float
    r_fs: float
    realized_h2: float


@dataclass(frozen=True)
class H2Reps:
    """Per-rep h² estimates beside the realized h² of the same reps.

    Attributes:
        estimates: Estimator name to its per-rep values, in rep order.
        realized: Per-rep realized h², in the same order.
    """

    estimates: Mapping[str, np.ndarray]
    realized: np.ndarray


def load_pedigree_estimates(
    pedigree_path: Path,
    trait: int = 1,
    min_generation: int | None = None,
) -> PedigreeEstimates:
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
        The replicate's estimates; ``realized_h2`` is in the filtered subset.
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

    return PedigreeEstimates(r=corrs, po_slope=po_slope, realized_h2=realized_h2)


def load_pooled_liability_correlations(
    pedigree_paths: list[Path],
    trait: int = 1,
    min_generation: int | None = None,
) -> dict[str, list[float]]:
    """Read per-rep liability correlations pooled to FS/PO/HS/1C.

    Thin aggregator over :func:`load_pedigree_estimates`: pools MO+FO → PO
    and MHS+PHS → HS (:func:`~simace.analysis.stats.naive_h2.pooled_mean`)
    so the chart has four x-axis categories instead of seven.

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
            pooled = pooled_mean(est.r[m] for m in members)
            if math.isfinite(pooled):
                out[pooled_name].append(pooled)
    return out


def load_naive_estimator_h2(
    pedigree_paths: Sequence[Path],
    trait: int = 1,
    min_generation: int | None = None,
) -> H2Reps:
    """Per-rep naive h² estimates (:class:`~simace.analysis.stats.naive_h2.NaiveH2`) and realized h².

    Every estimate and the realized h² come from :func:`load_pedigree_estimates`
    on the same individuals, so the generation filter applies to all of them.

    Args:
        pedigree_paths: one recorded ``pedigree.parquet`` path per replicate.
        trait: 1 or 2.
        min_generation: passed through to :func:`load_pedigree_estimates`.

    Returns:
        Estimates keyed by :class:`NaiveH2` field name; NaN where a rep had
        too few pairs.
    """
    reps = [load_pedigree_estimates(path, trait=trait, min_generation=min_generation) for path in pedigree_paths]
    per_rep = [naive_h2(est.r, est.po_slope) for est in reps]
    return H2Reps(
        estimates={f.name: np.asarray([getattr(est, f.name) for est in per_rep], dtype=float) for f in fields(NaiveH2)},
        realized=np.asarray([est.realized_h2 for est in reps], dtype=float),
    )


def load_observed_vs_liability_h2(
    pedigree_paths: Sequence[Path],
    tetrachoric: Sequence[Mapping[str, float]],
    trait: int = 1,
    min_generation: int | None = None,
) -> H2Reps:
    """Per-rep Falconer h² from liability and from tetrachoric correlations, and realized h².

    Args:
        pedigree_paths: one recorded ``pedigree.parquet`` path per replicate.
        tetrachoric: per replicate, in the same order, the tetrachoric
            correlations on affected status keyed by relationship code (at
            least ``MZ`` and ``FS``; NaN where missing). They are
            pre-aggregated over phenotyped generations and not re-filtered.
        trait: 1 or 2.
        min_generation: forwarded to :func:`load_pedigree_estimates` for the
            liability correlations and realized h².

    Returns:
        Estimates keyed ``liability_falconer`` and ``tetrachoric_falconer``.
    """
    reps = [load_pedigree_estimates(path, trait=trait, min_generation=min_generation) for path in pedigree_paths]
    pairs = list(zip(reps, tetrachoric, strict=True))
    return H2Reps(
        estimates={
            "liability_falconer": np.asarray([falconer_h2(est.r["MZ"], est.r["FS"]) for est, _ in pairs], dtype=float),
            "tetrachoric_falconer": np.asarray([falconer_h2(tet["MZ"], tet["FS"]) for _, tet in pairs], dtype=float),
        },
        realized=np.asarray([est.realized_h2 for est in reps], dtype=float),
    )


def load_sib_pair_liabilities(
    pedigree_paths: Sequence[Path],
    trait: int = 1,
    min_generation: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of paired sibling liabilities across all reps.

    Picks two full siblings per mating — the first two by ``id`` within each
    ``(mother, father)`` group.  Founders (mother == -1) and MZ twins
    (``twin != -1``) are excluded, so the returned pairs have an expected
    kinship of exactly 0.5 under random mating.

    Args:
        pedigree_paths: one recorded ``pedigree.parquet`` path per replicate.
        trait: 1 or 2.
        min_generation: if set, restrict to individuals in generations
            ``>= min_generation``; useful for waiting out AM's burn-in period.

    Returns:
        ``(liab_a, liab_b)`` 1-D arrays of equal length, one element per family.
    """
    liab_col = f"liability{trait}"
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for path in pedigree_paths:
        df = load_parquet(
            path,
            columns=["id", "mother", "father", "twin", "generation", liab_col],
        )
        mothers = df["mother"].to_numpy()
        keep = (mothers >= 0) & (df["twin"].to_numpy() == -1)
        if min_generation is not None:
            keep &= df["generation"].to_numpy() >= min_generation
        mothers = df["mother"].to_numpy()[keep].astype(np.int64)
        fathers = df["father"].to_numpy()[keep].astype(np.int64)
        ids = df["id"].to_numpy()[keep]
        liab = df[liab_col].to_numpy()[keep].astype(float)
        if mothers.size == 0:
            continue
        # First two children per (mother, father) mating, in ascending id
        # order — the sort/groupby-cumcount/merge idiom this replaced.
        # Fathers shift by +1 so a severed father (-1) keys injectively.
        base = np.int64(max(int(mothers.max()), int(fathers.max())) + 2)
        pair_key = mothers * base + (fathers + 1)
        order = np.lexsort((ids, pair_key))
        sorted_keys = pair_key[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        sizes = np.diff(np.r_[starts, len(sorted_keys)])
        multi = starts[sizes >= 2]
        liab_sorted = liab[order]
        a_list.append(liab_sorted[multi])
        b_list.append(liab_sorted[multi + 1])
    if not a_list:
        # Every pedigree was skipped (no qualifying pairs anywhere, or no paths
        # given) — np.concatenate rejects an empty list.
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return np.concatenate(a_list), np.concatenate(b_list)


def load_pedigree_estimates_per_generation(
    pedigree_path: Path,
    trait: int = 1,
    gens: list[int] | None = None,
) -> dict[int, CohortEstimates]:
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
        Dict ``g -> CohortEstimates``; a generation absent from the pedigree is all NaN.
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

    out: dict[int, CohortEstimates] = {}
    for g in gens:
        gen_mask = gen_arr == g
        if not gen_mask.any():
            out[g] = CohortEstimates(r_mz=math.nan, r_fs=math.nan, realized_h2=math.nan)
            continue

        cohort_corrs: dict[str, float] = {}
        # MZ + FS are within-generation by construction; gen_mask[idx1] alone is sufficient.
        for code in ("MZ", "FS"):
            idx1, idx2 = pairs_full[code]
            pair_mask = gen_mask[idx1]
            idx1_g, idx2_g = idx1[pair_mask], idx2[pair_mask]
            if len(idx1_g) < 10:
                cohort_corrs[code] = math.nan
            else:
                cohort_corrs[code] = safe_corrcoef(liab_full[idx1_g], liab_full[idx2_g])

        vA = float(np.var(A_full[gen_mask], ddof=1))
        vC = float(np.var(C_full[gen_mask], ddof=1))
        vE = float(np.var(E_full[gen_mask], ddof=1))
        total = vA + vC + vE
        realized_h2 = vA / total if total > 0 else float("nan")

        out[g] = CohortEstimates(r_mz=cohort_corrs["MZ"], r_fs=cohort_corrs["FS"], realized_h2=realized_h2)
    return out
