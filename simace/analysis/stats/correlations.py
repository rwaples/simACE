"""Pairwise relationship correlations, parent-offspring regressions, and h² estimators.

Covers liability/affected pair correlations, tetrachoric correlations across pair
types (overall, by generation, by sex, cross-trait), midparent-offspring
regressions (overall, by sex, on affected status), the closed-form observed-scale
h² estimators derived from those correlations, and the mate-pair correlation
matrix.

The pair statistics read a :class:`~pedigree_graph.RelationshipMoments` table
from :func:`~simace.analysis.stats.moments.relationship_moments_for` (ADR
0020): exact pair counts, 2×2 affection tables and liability moments over
every pair, with no pair list. A generation stratum is the first member's
generation, which is the offspring for ``MO``/``FO`` and the lower receiver
row for the symmetric codes; same-sex strata read sex from both members.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core.numerics import fast_linregress, safe_corrcoef
from simace.core.pedigree_arrays import PedigreeArrays
from simace.core.relationships import SEX_LEVELS

from .moments import Stratum, select_levels
from .tetrachoric import tetrachoric_corr_se, tetrachoric_from_table

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pedigree_graph import RelationshipMoments

    type _Frame = pd.DataFrame | pl.DataFrame

_MIN_PAIRS = 10


def _report_generations(df: _Frame) -> list[int]:
    """The last three generations of *df*, the ones the report stratifies by."""
    max_gen = int(df["generation"].to_numpy().max())
    return list(range(max(1, max_gen - 2), max_gen + 1))


def _finite_or_none(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _tetrachoric_entry(table: np.ndarray, n_pairs: int, liability_r: float | None = None) -> dict[str, Any]:
    """``{r, se, n_pairs[, liability_r]}`` for one category's 2×2 *table*; ``None`` values below ten pairs."""
    entry: dict[str, Any] = {"r": None, "se": None, "n_pairs": int(n_pairs)}
    if n_pairs >= _MIN_PAIRS:
        r, se = tetrachoric_from_table(int(table[1, 1]), int(table[1, 0]), int(table[0, 1]), int(table[0, 0]))
        entry["r"], entry["se"] = _finite_or_none(r), _finite_or_none(se)
    if liability_r is not None:
        entry["liability_r"] = float(liability_r) if n_pairs >= _MIN_PAIRS else None
    return entry


def _stratified_tetrachoric(moments: RelationshipMoments) -> dict[str, Any]:
    """Per-trait, per-category ``{r, se, n_pairs, liability_r}`` of one selection."""
    stratum = Stratum(moments)
    n_pairs = stratum.n_pairs
    result: dict[str, Any] = {}
    for trait_num in (1, 2):
        table = stratum.affection_table(trait_num)
        liability_r = stratum.liability_r(trait_num)
        result[f"trait{trait_num}"] = {
            ptype: _tetrachoric_entry(table[i], n_pairs[i], liability_r[i]) for ptype, i in stratum.report_positions()
        }
    return result


def compute_liability_correlations(*, moments: RelationshipMoments) -> dict[str, Any]:
    """Compute Pearson liability correlations per pair type and trait.

    Args:
        moments: Relationship moments of the sample.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to correlation
        or None below ten pairs.
    """
    stratum = Stratum(moments)
    n_pairs = stratum.n_pairs
    result = {}
    for trait_num in (1, 2):
        r = stratum.liability_r(trait_num)
        result[f"trait{trait_num}"] = {
            ptype: float(r[i]) if n_pairs[i] >= _MIN_PAIRS else None for ptype, i in stratum.report_positions()
        }
    return result


def _phi(table: np.ndarray) -> float | None:
    """Pearson r on the {0, 1} affection indicators from their 2×2 *table*; None when a side is constant."""
    n11, n10, n01, n00 = (int(table[1, 1]), int(table[1, 0]), int(table[0, 1]), int(table[0, 0]))
    first_affected, first_not = n11 + n10, n01 + n00
    second_affected, second_not = n11 + n01, n10 + n00
    denominator = first_affected * first_not * second_affected * second_not
    if denominator == 0:
        return None
    return (n11 * n00 - n10 * n01) / math.sqrt(denominator)


def compute_affected_correlations(*, moments: RelationshipMoments) -> dict[str, Any]:
    """Compute Pearson correlations on binary affected status per pair type and trait.

    This is the phi coefficient — Pearson r on {0, 1} data — and is the input
    to observed-scale Falconer-style h² estimators (e.g. ``2·(r_MZ − r_FS)``).

    Args:
        moments: Relationship moments of the sample.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to phi r or
        None (if fewer than 10 pairs, or either side is constant).
    """
    stratum = Stratum(moments)
    n_pairs = stratum.n_pairs
    result = {}
    for trait_num in (1, 2):
        table = stratum.affection_table(trait_num)
        result[f"trait{trait_num}"] = {
            ptype: _phi(table[i]) if n_pairs[i] >= _MIN_PAIRS else None for ptype, i in stratum.report_positions()
        }
    return result


def compute_tetrachoric(*, moments: RelationshipMoments) -> dict[str, Any]:
    """Compute tetrachoric correlations per pair type and trait.

    Args:
        moments: Relationship moments of the sample.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to
        ``{r, se, n_pairs}``.
    """
    stratum = Stratum(moments)
    n_pairs = stratum.n_pairs
    result = {}
    for trait_num in (1, 2):
        table = stratum.affection_table(trait_num)
        result[f"trait{trait_num}"] = {
            ptype: _tetrachoric_entry(table[i], n_pairs[i]) for ptype, i in stratum.report_positions()
        }
    return result


def compute_tetrachoric_by_generation(df: _Frame, *, moments: RelationshipMoments) -> dict[str, Any]:
    """Compute tetrachoric correlations stratified by generation.

    A pair belongs to the generation of its first member. The moments are
    role-oriented, so for the parent-offspring types (``MO``, ``FO``) that is
    the offspring, and for the symmetric types both members share a generation
    except across a skipped-generation pedigree. Only the last three
    generations are reported.

    Args:
        df: Phenotype DataFrame with a ``generation`` column; without one the
            result is empty.
        moments: Relationship moments of the sample.

    Returns:
        Dict keyed by ``gen{N}``, each containing per-trait per-pair-type
        ``{r, se, n_pairs, liability_r}``.
    """
    if "generation" not in df.columns:
        return {}
    return {
        f"gen{gen}": _stratified_tetrachoric(select_levels(moments, first_generation=gen))
        for gen in _report_generations(df)
    }


def compute_cross_trait_tetrachoric(df: _Frame, *, moments: RelationshipMoments) -> dict[str, Any]:
    """Compute cross-trait tetrachoric correlations (trait 1 vs trait 2).

    Includes same-person, same-person-by-generation, and cross-person
    (across relationship pair types) correlations.

    Args:
        df: Phenotype DataFrame with binary affection columns for both traits.
        moments: Relationship moments of the sample.

    Returns:
        Dict with keys ``same_person``, ``same_person_by_generation``,
        and ``cross_person``.
    """
    a1 = df["affected1"].to_numpy().astype(bool)
    a2 = df["affected2"].to_numpy().astype(bool)
    r_sp, se_sp = tetrachoric_corr_se(a1, a2)
    result: dict[str, Any] = {"same_person": {"r": _finite_or_none(r_sp), "se": _finite_or_none(se_sp), "n": len(df)}}
    by_gen: dict[str, Any] = {}
    if "generation" in df.columns:
        gen_arr = df["generation"].to_numpy()
        for gen in _report_generations(df):
            mask = gen_arr == gen
            n_g = int(mask.sum())
            if n_g < 50:
                by_gen[f"gen{gen}"] = {"r": None, "se": None, "n": n_g}
                continue
            r_g, se_g = tetrachoric_corr_se(a1[mask], a2[mask])
            by_gen[f"gen{gen}"] = {"r": _finite_or_none(r_g), "se": _finite_or_none(se_g), "n": n_g}
    result["same_person_by_generation"] = by_gen
    stratum = Stratum(moments)
    n_pairs = stratum.n_pairs
    table = stratum.affection_table(1, 2)
    result["cross_person"] = {
        ptype: _tetrachoric_entry(table[i], n_pairs[i]) for ptype, i in stratum.report_positions()
    }
    return result


def compute_tetrachoric_by_sex(*, moments: RelationshipMoments) -> dict[str, Any]:
    """Compute tetrachoric correlations for same-sex pairs only (FF and MM).

    Returns dict keyed by "female"/"male", each containing per-trait
    per-pair-type {r, se, n_pairs, liability_r}.
    """
    return {
        sex_label: _stratified_tetrachoric(select_levels(moments, first_sex=sex_val, second_sex=sex_val))
        for sex_val, sex_label in SEX_LEVELS
    }


def _po_regression(gen_idx: np.ndarray, liability: np.ndarray, id_to_row: np.ndarray, df: _Frame) -> dict:
    """Midparent-offspring regression for a given set of offspring indices."""
    mother_ids = df["mother"].to_numpy()[gen_idx]
    father_ids = df["father"].to_numpy()[gen_idx]
    has_m = (mother_ids >= 0) & (mother_ids < len(id_to_row))
    has_f = (father_ids >= 0) & (father_ids < len(id_to_row))
    m_rows = np.full(len(gen_idx), -1, dtype=np.int32)
    f_rows = np.full(len(gen_idx), -1, dtype=np.int32)
    m_rows[has_m] = id_to_row[mother_ids[has_m]]
    f_rows[has_f] = id_to_row[father_ids[has_f]]
    valid = (m_rows >= 0) & (f_rows >= 0)
    n_pairs = int(valid.sum())
    null = {"r": None, "r2": None, "slope": None, "intercept": None, "stderr": None, "pvalue": None, "n_pairs": n_pairs}
    if n_pairs < 10:
        return null
    offspring = liability[gen_idx[valid]]
    midparent = (liability[m_rows[valid]] + liability[f_rows[valid]]) / 2.0
    slope, intercept, r, stderr, pvalue = fast_linregress(midparent, offspring)
    return {
        "r": r,
        "r2": r**2,
        "slope": slope,
        "intercept": intercept,
        "stderr": stderr,
        "pvalue": pvalue,
        "n_pairs": n_pairs,
    }


def compute_parent_offspring_corr(df: _Frame) -> dict[str, Any]:
    """Compute midparent-offspring liability regression per generation and trait.

    Args:
        df: Phenotype DataFrame with liability, generation, and parent columns.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each containing per-generation
        regression stats (slope, r, r2, intercept, stderr, pvalue, n_pairs).
    """
    if "generation" not in df.columns:
        return {}
    max_gen = int(df["generation"].to_numpy().max())
    ids_arr = df["id"].to_numpy()
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)
    gen_arr = df["generation"].to_numpy()
    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].to_numpy()
        trait_result = {}
        for gen in range(1, max_gen + 1):
            gen_idx = np.where(gen_arr == gen)[0]
            trait_result[f"gen{gen}"] = _po_regression(gen_idx, liability, id_to_row, df)
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_parent_offspring_affected_corr(df: _Frame) -> dict[str, Any]:
    """Compute pooled midparent-offspring regression on binary affected status.

    Regresses ``offspring.affected`` (0/1) on midparent affected status
    ``(mother.affected + father.affected) / 2`` (values in {0, 0.5, 1}),
    pooled across every non-founder individual whose parents are both in the
    DataFrame.  The regression slope is the observed-scale PO heritability
    estimator; under LTM it can be back-transformed to liability via
    Dempster-Lerner.

    Args:
        df: Phenotype DataFrame with ``id``, ``mother``, ``father``, and
            ``affected{1,2}`` columns.

    Returns:
        Dict keyed ``trait1``/``trait2``, each with
        ``{slope, r, r2, intercept, stderr, pvalue, n_pairs}``.  Values are
        None when fewer than 10 valid trios or midparent has zero variance.
    """
    if "id" not in df.columns or "mother" not in df.columns or "father" not in df.columns:
        return {}

    ids_arr = df["id"].to_numpy()
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)

    non_founder_idx = np.where(df["mother"].to_numpy() >= 0)[0]

    null = {
        "r": None,
        "r2": None,
        "slope": None,
        "intercept": None,
        "stderr": None,
        "pvalue": None,
        "n_pairs": 0,
    }

    # Precompute valid trios once (parents present, looked up via id_to_row).
    mother_ids_arr = df["mother"].to_numpy()[non_founder_idx]
    father_ids_arr = df["father"].to_numpy()[non_founder_idx]
    has_m = (mother_ids_arr >= 0) & (mother_ids_arr < len(id_to_row))
    has_f = (father_ids_arr >= 0) & (father_ids_arr < len(id_to_row))
    m_rows = np.full(len(non_founder_idx), -1, dtype=np.int32)
    f_rows = np.full(len(non_founder_idx), -1, dtype=np.int32)
    m_rows[has_m] = id_to_row[mother_ids_arr[has_m]]
    f_rows[has_f] = id_to_row[father_ids_arr[has_f]]
    valid = (m_rows >= 0) & (f_rows >= 0)
    n_pairs = int(valid.sum())

    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        aff_col = f"affected{trait_num}"
        if aff_col not in df.columns:
            result[f"trait{trait_num}"] = {**null}
            continue
        affected = df[aff_col].to_numpy().astype(np.float64)
        if n_pairs < 10:
            result[f"trait{trait_num}"] = {**null, "n_pairs": n_pairs}
            continue
        midparent = (affected[m_rows[valid]] + affected[f_rows[valid]]) / 2.0
        # Zero-variance midparent (all parents concordant) gives an undefined
        # regression; surface as None rather than 0/0.
        if float(np.var(midparent)) < 1e-12:
            result[f"trait{trait_num}"] = {**null, "n_pairs": n_pairs}
            continue
        entry = _po_regression(non_founder_idx, affected, id_to_row, df)
        slope = entry.get("slope")
        if slope is not None and not np.isfinite(slope):
            entry = {**null, "n_pairs": entry.get("n_pairs", 0)}
        result[f"trait{trait_num}"] = entry
    return result


def compute_parent_offspring_corr_by_sex(df: _Frame) -> dict[str, Any]:
    """Compute midparent-offspring regression partitioned by offspring sex.

    Returns dict keyed by "female"/"male", each containing per-trait
    per-generation {slope, r, r2, intercept, stderr, pvalue, n_pairs}.
    """
    if "generation" not in df.columns:
        return {}
    max_gen = int(df["generation"].to_numpy().max())
    ids_arr = df["id"].to_numpy()
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)
    gen_arr = df["generation"].to_numpy()
    sex_arr = df["sex"].to_numpy()
    result: dict[str, Any] = {}
    for sex_val, sex_label in SEX_LEVELS:
        sex_result: dict[str, Any] = {}
        for trait_num in [1, 2]:
            liability = df[f"liability{trait_num}"].to_numpy()
            trait_result: dict[str, Any] = {}
            for gen in range(1, max_gen + 1):
                gen_idx = np.where((gen_arr == gen) & (sex_arr == sex_val))[0]
                trait_result[f"gen{gen}"] = _po_regression(gen_idx, liability, id_to_row, df)
            sex_result[f"trait{trait_num}"] = trait_result
        result[sex_label] = sex_result
    return result


def compute_observed_h2_estimators(
    affected_correlations: dict[str, Any],
    parent_offspring_affected_corr: dict[str, Any],
) -> dict[str, Any]:
    """Derive five naive observed-scale h² estimators from precomputed correlations.

    Reads from affected-status correlations (phi r per relationship type) and
    parent-offspring affected-status regression slopes.
    Each estimator is a closed-form combination that, under a liability-threshold
    model, is an unbiased estimator of ``h²_liab · z(K)²/(K(1−K))`` — i.e. the
    observed-scale h² — where K is the affected-status prevalence.

    Args:
        affected_correlations: Per-trait affected-status correlations by
            relationship type.
        parent_offspring_affected_corr: Per-trait parent-offspring regression
            on binary affected status.

    Returns:
        Dict keyed ``trait1``/``trait2``, each mapping estimator name to a
        float or None: ``{falconer, sibs, po, hs, cousins}``.
    """
    aff = affected_correlations or {}
    po_all = parent_offspring_affected_corr or {}

    def _two_diff(r_a: Any, r_b: Any) -> float | None:
        if r_a is None or r_b is None:
            return None
        return 2.0 * (float(r_a) - float(r_b))

    def _scale(r: Any, factor: float) -> float | None:
        if r is None:
            return None
        return factor * float(r)

    def _mean_hs(r_mhs: Any, r_phs: Any) -> float | None:
        vals = [float(v) for v in (r_mhs, r_phs) if v is not None]
        if not vals:
            return None
        return 4.0 * (sum(vals) / len(vals))

    result: dict[str, Any] = {}
    for trait_num in [1, 2]:
        key = f"trait{trait_num}"
        rs = aff.get(key, {}) or {}
        po_entry = po_all.get(key, {}) or {}
        po_slope = po_entry.get("slope")
        result[key] = {
            "falconer": _two_diff(rs.get("MZ"), rs.get("FS")),
            "sibs": _scale(rs.get("FS"), 2.0),
            "po": float(po_slope) if po_slope is not None else None,
            "hs": _mean_hs(rs.get("MHS"), rs.get("PHS")),
            "cousins": _scale(rs.get("1C"), 8.0),
        }
    return result


def compute_mate_correlation(df: _Frame) -> dict:
    """Compute 2x2 Pearson correlation matrix between mated pairs' liabilities.

    Each unique (mother, father) pair is counted once (not weighted by offspring).
    Only non-founders are considered.
    """
    ped = PedigreeArrays.from_frame(df)
    mothers_all = df["mother"].to_numpy()
    fathers_all = df["father"].to_numpy()
    child_mask = (mothers_all != -1) & (fathers_all != -1)
    m_child = mothers_all[child_mask].astype(np.int64)
    f_child = fathers_all[child_mask].astype(np.int64)
    if m_child.size == 0:
        return {"matrix": [[float("nan")] * 2] * 2, "n_pairs": 0}

    # Unique (mother, father) matings via an int64 pair key (ids are int32).
    base = np.int64(max(int(m_child.max()), int(f_child.max())) + 1)
    uniq_pairs = np.unique(m_child * base + f_child)
    mothers = uniq_pairs // base
    fathers = uniq_pairs % base
    parents_present = ped.contains(mothers) & ped.contains(fathers)
    mothers, fathers = mothers[parents_present], fathers[parents_present]
    if len(mothers) < 2:
        # A correlation needs two points, so the matrix is NaN -- but n_pairs
        # denotes the distinct valid-mating count everywhere else, so report
        # the real count rather than hard-coding zero for the singleton case.
        return {"matrix": [[float("nan")] * 2] * 2, "n_pairs": len(mothers)}

    f_liab = np.column_stack([ped.gather(f"liability{t}", mothers) for t in (1, 2)])  # (N, 2)
    m_liab = np.column_stack([ped.gather(f"liability{t}", fathers) for t in (1, 2)])  # (N, 2)

    matrix = [[float(safe_corrcoef(f_liab[:, i], m_liab[:, j])) for j in range(2)] for i in range(2)]
    return {"matrix": matrix, "n_pairs": len(mothers)}
