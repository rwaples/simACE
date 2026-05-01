"""Pedigree-structure summaries: family size and parent presence."""

from typing import Any

import numpy as np
import pandas as pd

from simace.core.relationships import SEX_LEVELS


def _offspring_count_dist(counts: pd.Series, n: int) -> dict[str, float]:
    """Return ``{"0", "1", "2", "3", "4+"}`` proportions of ``counts`` over ``n``."""
    out = {"0": round(int((counts == 0).sum()) / n, 4)}
    for k in (1, 2, 3):
        out[str(k)] = round(int((counts == k).sum()) / n, 4)
    out["4+"] = round(int((counts >= 4).sum()) / n, 4)
    return out


def compute_mean_family_size(df: pd.DataFrame) -> dict[str, Any]:
    """Compute mean realised family size (offspring per mating pair).

    Uses non-founder individuals (mother != -1) grouped by (mother, father).
    """
    if "mother" not in df.columns or "father" not in df.columns:
        return {}

    children = df.loc[(df["mother"] != -1) & (df["father"] != -1)]
    if len(children) == 0:
        return {}

    family_sizes = children.groupby(["mother", "father"]).size()

    # Fraction with at least one phenotyped full sibling
    families_with_sibs = family_sizes[family_sizes >= 2].index
    has_sib = children.set_index(["mother", "father"]).index.isin(families_with_sibs)
    frac_with_full_sib = round(float(has_sib.sum()) / len(children), 4)

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
    ids_arr = df["id"].to_numpy()
    n_total = len(df)
    id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(n_total, dtype=np.int32)
    m_rows = id_to_row[children["mother"].to_numpy()]
    f_rows = id_to_row[children["father"].to_numpy()]
    m_rows = m_rows[m_rows >= 0]
    f_rows = f_rows[f_rows >= 0]
    counts_arr = np.bincount(m_rows, minlength=n_total) + np.bincount(f_rows, minlength=n_total)
    offspring_counts = pd.Series(counts_arr, index=df["id"])
    person_dist = _offspring_count_dist(offspring_counts, n_total)

    # Offspring per person by sex
    person_dist_by_sex: dict[str, dict[str, float]] = {}
    if "sex" in df.columns:
        sex_by_id = df.set_index("id")["sex"]
        for sex_val, sex_label in SEX_LEVELS:
            sex_ids = sex_by_id[sex_by_id == sex_val].index
            sex_counts = offspring_counts.reindex(sex_ids, fill_value=0)
            if len(sex_counts) > 0:
                person_dist_by_sex[sex_label] = _offspring_count_dist(sex_counts, len(sex_counts))

    # Number of mates by sex
    # Females: unique fathers per mother
    mates_female = children.groupby("mother")["father"].nunique()
    # Males: unique mothers per father
    mates_male = children.groupby("father")["mother"].nunique()
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
        "median": round(float(family_sizes.median()), 1),
        "q1": round(float(family_sizes.quantile(0.25)), 1),
        "q3": round(float(family_sizes.quantile(0.75)), 1),
        "n_families": len(family_sizes),
        "frac_with_full_sib": frac_with_full_sib,
        "size_dist": dist,
        "person_offspring_dist": person_dist,
        "person_offspring_dist_by_sex": person_dist_by_sex,
        "mates_by_sex": mates_by_sex,
    }


def compute_parent_status(
    df: pd.DataFrame,
    df_ped: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Count individuals by number of parents phenotyped and in pedigree.

    Returns dict with 'phenotyped' and optionally 'in_pedigree', each mapping
    0/1/2 → count of individuals with that many parents present.
    """
    if "mother" not in df.columns or "father" not in df.columns:
        return {}

    pheno_ids = df["id"].to_numpy()
    mothers = df["mother"].values
    fathers = df["father"].values

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
