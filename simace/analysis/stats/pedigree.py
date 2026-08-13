"""Pedigree-structure summaries: family size and parent presence.

Library-agnostic over pandas/polars input (transitional, ADR 0015): columns
come out through ``.to_numpy()`` and all grouping runs in NumPy, so both
frame libraries produce identical results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core.pedigree_arrays import PedigreeArrays
from simace.core.relationships import SEX_LEVELS

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def _offspring_count_dist(counts: np.ndarray, n: int) -> dict[str, float]:
    """Return ``{"0", "1", "2", "3", "4+"}`` proportions of ``counts`` over ``n``."""
    out = {"0": round(int((counts == 0).sum()) / n, 4)}
    for k in (1, 2, 3):
        out[str(k)] = round(int((counts == k).sum()) / n, 4)
    out["4+"] = round(int((counts >= 4).sum()) / n, 4)
    return out


def compute_mean_family_size(df: pd.DataFrame | pl.DataFrame) -> dict[str, Any]:
    """Compute mean realised family size (offspring per mating pair).

    Uses non-founder individuals (mother != -1) grouped by (mother, father).
    """
    if "mother" not in df.columns or "father" not in df.columns:
        return {}

    mothers_all = df["mother"].to_numpy()
    fathers_all = df["father"].to_numpy()
    child_mask = (mothers_all != -1) & (fathers_all != -1)
    n_children = int(child_mask.sum())
    if n_children == 0:
        return {}

    mothers = mothers_all[child_mask]
    fathers = fathers_all[child_mask]

    # Group by (mother, father) via an int64 pair key (ids are int32, so
    # base**2 fits int64 — same bound as the canonical pair-key encoding).
    base = np.int64(max(int(mothers.max()), int(fathers.max())) + 1)
    pair_key = mothers.astype(np.int64) * base + fathers.astype(np.int64)
    uniq_pairs, pair_inverse, family_sizes = np.unique(pair_key, return_inverse=True, return_counts=True)

    # Fraction with at least one phenotyped full sibling
    has_sib = family_sizes[pair_inverse] >= 2
    frac_with_full_sib = round(float(has_sib.sum()) / n_children, 4)

    # Family size distribution per mating (1, 2, 3, 4+)
    n_fam = len(family_sizes)
    dist: dict[str, float] = {}
    for k in [1, 2, 3]:
        dist[str(k)] = round(int((family_sizes == k).sum()) / n_fam, 4)
    dist["4+"] = round(int((family_sizes >= 4).sum()) / n_fam, 4)

    # Offspring per person (including 0 for childless individuals).
    # Count via bincount on row positions: faster than groupby + Series.update/add.
    # When df is a subsample, a child's parent may not be in df["id"]; id_to_row
    # marks those as -1 and they must be masked out before bincount (which rejects
    # negatives). This matches the prior groupby+update semantics, which only
    # counted offspring against parents present in df["id"].
    ped = PedigreeArrays.from_frame(df)
    n_total = len(df)
    m_rows = ped.positions(mothers[ped.contains(mothers)])
    f_rows = ped.positions(fathers[ped.contains(fathers)])
    counts_arr = np.bincount(m_rows, minlength=n_total) + np.bincount(f_rows, minlength=n_total)
    person_dist = _offspring_count_dist(counts_arr, n_total)

    # Offspring per person by sex
    person_dist_by_sex: dict[str, dict[str, float]] = {}
    if "sex" in df.columns:
        # counts_arr is already in df row order, so selecting by sex is a
        # mask -- the id round-trip this replaced was never needed.
        sex_vals = df["sex"].to_numpy()
        for sex_val, sex_label in SEX_LEVELS:
            sex_counts = counts_arr[sex_vals == sex_val]
            if len(sex_counts) > 0:
                person_dist_by_sex[sex_label] = _offspring_count_dist(sex_counts, len(sex_counts))

    # Number of mates by sex: each unique (mother, father) key is one distinct
    # mating, so counting keys per mother (father) counts distinct mates.
    pair_mothers = uniq_pairs // base
    pair_fathers = uniq_pairs % base
    mates_female = np.unique(pair_mothers, return_counts=True)[1]
    mates_male = np.unique(pair_fathers, return_counts=True)[1]
    n_mothers = len(mates_female)
    n_fathers = len(mates_male)
    mates_by_sex: dict[str, Any] = {
        "female_mean": round(float(mates_female.mean()), 2) if n_mothers else 0,
        "male_mean": round(float(mates_male.mean()), 2) if n_fathers else 0,
        "female_1": round(int((mates_female == 1).sum()) / n_mothers, 4) if n_mothers else 0,
        "female_2+": round(int((mates_female >= 2).sum()) / n_mothers, 4) if n_mothers else 0,
        "male_1": round(int((mates_male == 1).sum()) / n_fathers, 4) if n_fathers else 0,
        "male_2+": round(int((mates_male >= 2).sum()) / n_fathers, 4) if n_fathers else 0,
    }

    return {
        "mean": round(float(family_sizes.mean()), 2),
        "median": round(float(np.median(family_sizes)), 1),
        "q1": round(float(np.quantile(family_sizes, 0.25)), 1),
        "q3": round(float(np.quantile(family_sizes, 0.75)), 1),
        "n_families": len(family_sizes),
        "frac_with_full_sib": frac_with_full_sib,
        "size_dist": dist,
        "person_offspring_dist": person_dist,
        "person_offspring_dist_by_sex": person_dist_by_sex,
        "mates_by_sex": mates_by_sex,
    }


def compute_parent_status(
    df: pd.DataFrame | pl.DataFrame,
    df_ped: pd.DataFrame | pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Count individuals by number of parents phenotyped and in pedigree.

    Returns dict with 'phenotyped' and optionally 'in_pedigree', each mapping
    0/1/2 → count of individuals with that many parents present.
    """
    if "mother" not in df.columns or "father" not in df.columns:
        return {}

    pheno_ids = df["id"].to_numpy()
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()

    m_pheno = np.isin(mothers, pheno_ids) & (mothers != -1)
    f_pheno = np.isin(fathers, pheno_ids) & (fathers != -1)
    n_parents_pheno = m_pheno.astype(int) + f_pheno.astype(int)
    result: dict[str, Any] = {
        "phenotyped": {str(k): int((n_parents_pheno == k).sum()) for k in [0, 1, 2]},
    }

    if df_ped is not None:
        ped_ids = df_ped["id"].to_numpy()
        m_ped = np.isin(mothers, ped_ids) & (mothers != -1)
        f_ped = np.isin(fathers, ped_ids) & (fathers != -1)
        n_parents_ped = m_ped.astype(int) + f_ped.astype(int)
        result["in_pedigree"] = {str(k): int((n_parents_ped == k).sum()) for k in [0, 1, 2]}

    return result
