"""Tidy data exports for external-tool consumption (R, GCTA, PLINK, sparseREML).

Three per-rep artifacts:

- :func:`export_cumulative_incidence` — long/tidy TSV with one row per
  ``(trait, sex, generation, age)`` stratum.
- :func:`export_pairwise_relatedness` — TSV of canonical relationship pairs
  ``(id1, id2, rel_code, kinship)``, filtered by a user-supplied
  ``min_kinship`` threshold.
- :func:`export_sparse_grm` — sparse GRM in ``ace_sreml`` binary CSC format,
  with founder-couple FIDs in the accompanying ``.grm.id`` file.

Reuses :func:`~sim_ace.analysis.stats.compute_cumulative_incidence_by_sex_generation`,
:class:`~sim_ace.core.pedigree_graph.PedigreeGraph`,
:data:`~sim_ace.core.pedigree_graph.PAIR_KINSHIP`, and
:func:`~sim_ace.analysis.export_grm.export_sparse_grm_binary`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from sim_ace.analysis.export_grm import export_sparse_grm_binary
from sim_ace.analysis.stats import compute_cumulative_incidence_by_sex_generation
from sim_ace.core.pedigree_graph import REL_REGISTRY, PedigreeGraph

logger = logging.getLogger(__name__)

__all__ = [
    "assign_founder_family_ids",
    "export_cumulative_incidence",
    "export_pairwise_relatedness",
    "export_sparse_grm",
]


# ---------------------------------------------------------------------------
# Cumulative incidence
# ---------------------------------------------------------------------------


def export_cumulative_incidence(
    phenotype_df: pd.DataFrame,
    censor_age: float,
    out_path: str | Path,
    n_points: int = 200,
) -> Path:
    """Write a long/tidy cumulative incidence TSV.

    Columns: ``trait`` (1|2), ``sex`` (F|M), ``generation`` (int),
    ``age`` (float), ``cum_incidence`` (float in ``[0, 1]``),
    ``n_at_risk`` (int).

    ``cum_incidence`` is the same quantity computed by
    :func:`~sim_ace.analysis.stats.compute_cumulative_incidence_by_sex_generation`
    (affected count divided by stratum size).  ``n_at_risk`` at age *a* is
    the count of individuals in the stratum whose observed time
    (event or censor) satisfies ``t_observed >= a``.
    """
    for col in ("sex", "generation"):
        if col not in phenotype_df.columns:
            raise ValueError(f"phenotype_df missing required column: {col!r}")

    nested = compute_cumulative_incidence_by_sex_generation(phenotype_df, censor_age, n_points=n_points)
    ages_grid = np.linspace(0, censor_age, n_points)

    sex_arr = phenotype_df["sex"].values
    gen_arr = phenotype_df["generation"].values

    frames: list[pd.DataFrame] = []
    for trait_key, trait_dict in nested.items():
        trait_num = int(trait_key.removeprefix("trait"))
        t_obs = phenotype_df[f"t_observed{trait_num}"].values
        for gen_key, gen_dict in trait_dict.items():
            gen_num = int(gen_key.removeprefix("gen"))
            for sex_label_long, stratum in gen_dict.items():
                sex_val = 0 if sex_label_long == "female" else 1
                mask = (gen_arr == gen_num) & (sex_arr == sex_val)
                n_stratum = int(mask.sum())
                sorted_t = np.sort(t_obs[mask])
                # at-risk(a) = #{i : t_observed_i >= a}
                n_below = np.searchsorted(sorted_t, ages_grid, side="left")
                n_at_risk = n_stratum - n_below
                frames.append(
                    pd.DataFrame(
                        {
                            "trait": trait_num,
                            "sex": "F" if sex_val == 0 else "M",
                            "generation": gen_num,
                            "age": np.asarray(stratum["ages"], dtype=np.float64),
                            "cum_incidence": np.asarray(stratum["values"], dtype=np.float64),
                            "n_at_risk": n_at_risk.astype(np.int64),
                        }
                    )
                )

    columns = ["trait", "sex", "generation", "age", "cum_incidence", "n_at_risk"]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    df = df[columns]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    logger.info("export_cumulative_incidence: %d rows -> %s", len(df), out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# Pairwise relatedness
# ---------------------------------------------------------------------------


def _min_max_degree_for_kinship(min_kinship: float) -> int:
    """Smallest ``max_degree`` covering every code with nominal kinship >= threshold.

    Returns 0 if no code qualifies.
    """
    degrees = [rt.degree for rt in REL_REGISTRY.values() if rt.kinship >= min_kinship]
    return max(degrees) if degrees else 0


def export_pairwise_relatedness(
    pedigree_df: pd.DataFrame,
    out_path: str | Path,
    min_kinship: float = 0.0625,
) -> Path:
    """Write a TSV of canonical relationship pairs.

    Columns: ``id1`` (lo), ``id2`` (hi), ``rel_code``
    (e.g. ``MZ``/``FS``/``PO``/``MHS``/``Av``/``1C``), ``kinship``.

    Only pairs with kinship >= ``min_kinship`` are emitted.  Internally we
    ask :meth:`~sim_ace.core.pedigree_graph.PedigreeGraph.extract_pairs`
    only for the smallest ``max_degree`` that still covers every code whose
    nominal kinship meets the threshold, so we do not enumerate pairs we
    would immediately discard.  A final per-row check enforces the
    threshold exactly even when inbreeding perturbs pairwise kinship away
    from its nominal lookup value.

    ``kinship`` values come from
    :meth:`~sim_ace.core.pedigree_graph.PedigreeGraph.compute_pair_kinship`,
    which equals ``PAIR_KINSHIP[rel_code]`` in non-inbred pedigrees and
    reads from the sparse kinship matrix otherwise.
    """
    max_degree = _min_max_degree_for_kinship(min_kinship)
    columns = ["id1", "id2", "rel_code", "kinship"]

    if max_degree == 0 or len(pedigree_df) == 0:
        df = pd.DataFrame(columns=columns)
    else:
        pg = PedigreeGraph(pedigree_df)
        pairs = pg.extract_pairs(max_degree=max_degree)
        kinship_by_code = pg.compute_pair_kinship(pairs)

        ids = pedigree_df["id"].values
        frames: list[pd.DataFrame] = []
        for code, (idx1, idx2) in pairs.items():
            if len(idx1) == 0:
                continue
            kin = kinship_by_code[code]
            keep = kin >= min_kinship
            if not np.any(keep):
                continue
            i1 = ids[idx1[keep]]
            i2 = ids[idx2[keep]]
            lo = np.minimum(i1, i2)
            hi = np.maximum(i1, i2)
            frames.append(
                pd.DataFrame(
                    {
                        "id1": lo.astype(np.int64),
                        "id2": hi.astype(np.int64),
                        "rel_code": code,
                        "kinship": kin[keep].astype(np.float64),
                    }
                )
            )

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        df = df[columns].sort_values(["rel_code", "id1", "id2"], kind="mergesort").reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(
        "export_pairwise_relatedness: %d rows -> %s (min_kinship=%g, max_degree=%d)",
        len(df),
        out_path.name,
        min_kinship,
        max_degree,
    )
    return out_path


# ---------------------------------------------------------------------------
# Founder-couple family IDs
# ---------------------------------------------------------------------------


def assign_founder_family_ids(pedigree_df: pd.DataFrame) -> pd.Series:
    """Assign a family id (FID) to every individual in the pedigree.

    Two individuals share an FID iff they belong to the same connected
    component of the parent-child graph — equivalently, they descend from
    (or are) the same founder couple, possibly linked through marriages
    between founder lineages.  The FID value is the smallest ``id`` inside
    each connected component, giving a deterministic, human-readable label.

    Returns:
        An ``int64`` :class:`pandas.Series` aligned to ``pedigree_df.index``.
    """
    n = len(pedigree_df)
    if n == 0:
        return pd.Series([], dtype=np.int64, name="FID")

    ids = pedigree_df["id"].values
    id_to_row = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    id_to_row[ids] = np.arange(n, dtype=np.int64)

    mother = pedigree_df["mother"].values
    father = pedigree_df["father"].values

    row_idx = np.arange(n, dtype=np.int64)
    m_mask = mother >= 0
    f_mask = father >= 0
    src = np.concatenate([row_idx[m_mask], row_idx[f_mask]])
    dst = np.concatenate([id_to_row[mother[m_mask]], id_to_row[father[f_mask]]])

    data = np.ones(len(src), dtype=np.int8)
    graph = sp.coo_matrix((data, (src, dst)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    fids = pd.Series(ids, index=pedigree_df.index).groupby(labels).transform("min")
    return fids.astype(np.int64).rename("FID")


# ---------------------------------------------------------------------------
# Sparse GRM
# ---------------------------------------------------------------------------


def export_sparse_grm(
    pedigree_df: pd.DataFrame,
    prefix: str | Path,
    threshold: float = 0.05,
) -> tuple[Path, Path]:
    r"""Export a sparse GRM in ``ace_sreml`` binary format with founder FIDs.

    Wraps :func:`~sim_ace.analysis.export_grm.export_sparse_grm_binary`.
    After the binary writer produces its default ``FID=IID`` id file, this
    function overwrites the id file with ``<founder_fid>\t<iid>`` lines so
    downstream tools (GCTA / PLINK / sparseREML) see family structure.

    ``threshold`` is on the GRM scale (kinship * 2), matching the writer's
    convention; ``0.05`` is sparseREML's default ``GRM_range[0]``.
    """
    pg = PedigreeGraph(pedigree_df)
    pg.compute_inbreeding()  # populates pg._kinship_matrix
    K = pg._kinship_matrix

    ids = np.asarray(pedigree_df["id"].values)
    fids = assign_founder_family_ids(pedigree_df).to_numpy()

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path, id_path = export_sparse_grm_binary(K, iids=ids, prefix=prefix, to_grm=True, threshold=threshold)

    with id_path.open("w") as fh:
        for fid, iid in zip(fids, ids, strict=True):
            fh.write(f"{int(fid)}\t{int(iid)}\n")
    logger.info(
        "export_sparse_grm: rewrote %s with %d founder-couple FIDs",
        id_path.name,
        len(np.unique(fids)),
    )
    return bin_path, id_path
