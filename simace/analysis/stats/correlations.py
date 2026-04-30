"""Pairwise relationship correlations, parent-offspring regressions, and h² estimators.

Covers liability/affected pair correlations, tetrachoric correlations across pair
types (overall, by generation, by sex, cross-trait), midparent-offspring
regressions (overall, by sex, on affected status), the closed-form observed-scale
h² estimators derived from those correlations, and the mate-pair correlation
matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core._numba_utils import _pearsonr_core
from simace.core.numerics import fast_linregress, safe_corrcoef
from simace.core.relationships import PAIR_TYPES, SEX_LEVELS

from .tetrachoric import _tetrachoric_for_pairs, tetrachoric_corr_se

if TYPE_CHECKING:
    import pandas as pd


def compute_liability_correlations(
    df: pd.DataFrame,
    seed: int = 42,
    *,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Compute Pearson liability correlations per pair type and trait.

    Args:
        df: Phenotype DataFrame with liability columns.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to correlation.
    """
    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].values
        trait_result: dict[str, float | None] = {}
        for ptype in PAIR_TYPES:
            idx1, idx2 = pairs[ptype]
            trait_result[ptype] = float(_pearsonr_core(liability[idx1], liability[idx2])) if len(idx1) >= 10 else None
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_affected_correlations(
    df: pd.DataFrame,
    seed: int = 42,
    *,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Compute Pearson correlations on binary affected status per pair type and trait.

    This is the phi coefficient — Pearson r on {0, 1} data — and is the input
    to observed-scale Falconer-style h² estimators (e.g. ``2·(r_MZ − r_FS)``).

    Args:
        df: Phenotype DataFrame with ``affected{1,2}`` columns.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to phi r or
        None (if fewer than 10 pairs, or either side is constant).
    """
    result = {}
    for trait_num in [1, 2]:
        affected = df[f"affected{trait_num}"].values.astype(np.float64)
        trait_result: dict[str, float | None] = {}
        for ptype in PAIR_TYPES:
            idx1, idx2 = pairs[ptype]
            if len(idx1) < 10:
                trait_result[ptype] = None
                continue
            r = safe_corrcoef(affected[idx1], affected[idx2])
            trait_result[ptype] = None if np.isnan(r) else r
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_tetrachoric(
    df: pd.DataFrame,
    seed: int = 42,
    *,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Compute tetrachoric correlations per pair type and trait.

    Args:
        df: Phenotype DataFrame with binary affection columns.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each mapping pair type to
        ``{r, se, n_pairs}``.
    """
    result = {}
    for trait_num in [1, 2]:
        affected = df[f"affected{trait_num}"].values.astype(bool)
        trait_result = {}
        for ptype in PAIR_TYPES:
            idx1, idx2 = pairs[ptype]
            trait_result[ptype] = _tetrachoric_for_pairs(idx1, idx2, affected)
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_tetrachoric_by_generation(
    df: pd.DataFrame,
    seed: int = 42,
    *,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Compute tetrachoric correlations stratified by generation.

    Args:
        df: Phenotype DataFrame with generation and affection columns.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs.

    Returns:
        Dict keyed by ``gen{N}``, each containing per-trait per-pair-type
        ``{r, se, n_pairs, liability_r}``.
    """
    if "generation" not in df.columns:
        return {}
    gen_arr = df["generation"].values
    max_gen = int(gen_arr.max())
    plot_gens = list(range(max(1, max_gen - 2), max_gen + 1))
    result = {}
    for gen in plot_gens:
        gen_result = {}
        for trait_num in [1, 2]:
            affected = df[f"affected{trait_num}"].values.astype(bool)
            liability = df[f"liability{trait_num}"].values
            trait_result = {}
            for ptype in PAIR_TYPES:
                idx1, idx2 = pairs[ptype]
                mask = gen_arr[idx1] == gen
                trait_result[ptype] = _tetrachoric_for_pairs(idx1[mask], idx2[mask], affected, liability)
            gen_result[f"trait{trait_num}"] = trait_result
        result[f"gen{gen}"] = gen_result
    return result


def compute_cross_trait_tetrachoric(
    df: pd.DataFrame,
    seed: int = 42,
    *,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Compute cross-trait tetrachoric correlations (trait 1 vs trait 2).

    Includes same-person, same-person-by-generation, and cross-person
    (across relationship pair types) correlations.

    Args:
        df: Phenotype DataFrame with binary affection columns for both traits.
        seed: Random seed (unused, kept for API consistency).
        pairs: Pre-extracted relationship pairs.

    Returns:
        Dict with keys ``same_person``, ``same_person_by_generation``,
        and ``cross_person``.
    """
    a1 = df["affected1"].values.astype(bool)
    a2 = df["affected2"].values.astype(bool)
    r_sp, se_sp = tetrachoric_corr_se(a1, a2)
    result: dict[str, Any] = {
        "same_person": {
            "r": float(r_sp) if not np.isnan(r_sp) else None,
            "se": float(se_sp) if not np.isnan(se_sp) else None,
            "n": len(df),
        }
    }
    by_gen: dict[str, Any] = {}
    if "generation" in df.columns:
        gen_arr = df["generation"].values
        max_gen = int(gen_arr.max())
        plot_gens = list(range(max(1, max_gen - 2), max_gen + 1))
        for gen in plot_gens:
            mask = gen_arr == gen
            n_g = int(mask.sum())
            if n_g < 50:
                by_gen[f"gen{gen}"] = {"r": None, "se": None, "n": n_g}
                continue
            r_g, se_g = tetrachoric_corr_se(a1[mask], a2[mask])
            by_gen[f"gen{gen}"] = {
                "r": float(r_g) if not np.isnan(r_g) else None,
                "se": float(se_g) if not np.isnan(se_g) else None,
                "n": n_g,
            }
    result["same_person_by_generation"] = by_gen
    cross: dict[str, Any] = {}
    for ptype in PAIR_TYPES:
        idx1, idx2 = pairs[ptype]
        n_p = len(idx1)
        if n_p < 10:
            cross[ptype] = {"r": None, "se": None, "n_pairs": int(n_p)}
            continue
        r_cp, se_cp = tetrachoric_corr_se(a1[idx1], a2[idx2])
        cross[ptype] = {
            "r": float(r_cp) if not np.isnan(r_cp) else None,
            "se": float(se_cp) if not np.isnan(se_cp) else None,
            "n_pairs": int(n_p),
        }
    result["cross_person"] = cross
    return result


def compute_tetrachoric_by_sex(
    df: pd.DataFrame,
    seed: int = 42,
    *,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Compute tetrachoric correlations for same-sex pairs only (FF and MM).

    Returns dict keyed by "female"/"male", each containing per-trait
    per-pair-type {r, se, n_pairs, liability_r}.
    """
    sex_arr = df["sex"].values
    result: dict[str, Any] = {}
    for sex_val, sex_label in SEX_LEVELS:
        sex_result: dict[str, Any] = {}
        for trait_num in [1, 2]:
            affected = df[f"affected{trait_num}"].values.astype(bool)
            liability = df[f"liability{trait_num}"].values
            trait_result: dict[str, Any] = {}
            for ptype in PAIR_TYPES:
                idx1, idx2 = pairs[ptype]
                sex_mask = (sex_arr[idx1] == sex_val) & (sex_arr[idx2] == sex_val)
                trait_result[ptype] = _tetrachoric_for_pairs(idx1[sex_mask], idx2[sex_mask], affected, liability)
            sex_result[f"trait{trait_num}"] = trait_result
        result[sex_label] = sex_result
    return result


def _po_regression(gen_idx: np.ndarray, liability: np.ndarray, id_to_row: np.ndarray, df: pd.DataFrame) -> dict:
    """Midparent-offspring regression for a given set of offspring indices."""
    mother_ids = df["mother"].values[gen_idx]
    father_ids = df["father"].values[gen_idx]
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


def compute_parent_offspring_corr(df: pd.DataFrame) -> dict[str, Any]:
    """Compute midparent-offspring liability regression per generation and trait.

    Args:
        df: Phenotype DataFrame with liability, generation, and parent columns.

    Returns:
        Dict keyed by ``trait1``/``trait2``, each containing per-generation
        regression stats (slope, r, r2, intercept, stderr, pvalue, n_pairs).
    """
    if "generation" not in df.columns:
        return {}
    max_gen = int(df["generation"].max())
    ids_arr = df["id"].values
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)
    gen_arr = df["generation"].values
    result = {}
    for trait_num in [1, 2]:
        liability = df[f"liability{trait_num}"].values
        trait_result = {}
        for gen in range(1, max_gen + 1):
            gen_idx = np.where(gen_arr == gen)[0]
            trait_result[f"gen{gen}"] = _po_regression(gen_idx, liability, id_to_row, df)
        result[f"trait{trait_num}"] = trait_result
    return result


def compute_parent_offspring_affected_corr(df: pd.DataFrame) -> dict[str, Any]:
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

    ids_arr = df["id"].values
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)

    non_founder_idx = np.where(df["mother"].values >= 0)[0]

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
    mother_ids_arr = df["mother"].values[non_founder_idx]
    father_ids_arr = df["father"].values[non_founder_idx]
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
        affected = df[aff_col].values.astype(np.float64)
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


def compute_parent_offspring_corr_by_sex(df: pd.DataFrame) -> dict[str, Any]:
    """Compute midparent-offspring regression partitioned by offspring sex.

    Returns dict keyed by "female"/"male", each containing per-trait
    per-generation {slope, r, r2, intercept, stderr, pvalue, n_pairs}.
    """
    if "generation" not in df.columns:
        return {}
    max_gen = int(df["generation"].max())
    ids_arr = df["id"].values
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)
    gen_arr = df["generation"].values
    sex_arr = df["sex"].values
    result: dict[str, Any] = {}
    for sex_val, sex_label in SEX_LEVELS:
        sex_result: dict[str, Any] = {}
        for trait_num in [1, 2]:
            liability = df[f"liability{trait_num}"].values
            trait_result: dict[str, Any] = {}
            for gen in range(1, max_gen + 1):
                gen_idx = np.where((gen_arr == gen) & (sex_arr == sex_val))[0]
                trait_result[f"gen{gen}"] = _po_regression(gen_idx, liability, id_to_row, df)
            sex_result[f"trait{trait_num}"] = trait_result
        result[sex_label] = sex_result
    return result


def compute_observed_h2_estimators(stats: dict[str, Any]) -> dict[str, Any]:
    """Derive five naive observed-scale h² estimators from precomputed correlations.

    Reads from ``stats["affected_correlations"]`` (phi r per pair type) and
    ``stats["parent_offspring_affected_corr"]`` (PO regression slope on binary).
    Each estimator is a closed-form combination that, under a liability-threshold
    model, is an unbiased estimator of ``h²_liab · z(K)²/(K(1−K))`` — i.e. the
    observed-scale h² — where K is the affected-status prevalence.

    Args:
        stats: The in-progress stats dict with ``affected_correlations`` and
            ``parent_offspring_affected_corr`` already populated.

    Returns:
        Dict keyed ``trait1``/``trait2``, each mapping estimator name to a
        float or None: ``{falconer, sibs, po, hs, cousins}``.
    """
    aff = stats.get("affected_correlations", {}) or {}
    po_all = stats.get("parent_offspring_affected_corr", {}) or {}

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


def compute_mate_correlation(df: pd.DataFrame) -> dict:
    """Compute 2x2 Pearson correlation matrix between mated pairs' liabilities.

    Each unique (mother, father) pair is counted once (not weighted by offspring).
    Only non-founders are considered.
    """
    nf = df[df["mother"] != -1][["mother", "father"]].drop_duplicates()
    if len(nf) < 2:
        return {"matrix": [[float("nan")] * 2] * 2, "n_pairs": 0}

    lookup = df.set_index("id")[["liability1", "liability2"]]
    f_liab = lookup.loc[nf["mother"].values].values  # (N, 2)
    m_liab = lookup.loc[nf["father"].values].values  # (N, 2)

    matrix = [[float(safe_corrcoef(f_liab[:, i], m_liab[:, j])) for j in range(2)] for i in range(2)]
    return {"matrix": matrix, "n_pairs": len(nf)}
