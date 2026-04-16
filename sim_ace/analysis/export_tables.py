"""Tidy data exports for external-tool consumption (R, GCTA, PLINK, sparseREML).

Four per-rep artifacts:

- :func:`export_cumulative_incidence` — long/tidy TSV with one row per
  ``(trait, sex, generation, age)`` stratum.
- :func:`export_pairwise_relatedness` — TSV of canonical relationship pairs
  ``(id1, id2, rel_code, kinship)``, filtered by a user-supplied
  ``min_kinship`` threshold.
- :func:`export_sparse_grm` — sparse GRM in ``ace_sreml`` binary CSC format,
  with founder-couple FIDs in the accompanying ``.grm.id`` file.
- :func:`export_pgs` — per-individual proxy polygenic score with a
  user-specified accuracy ``r²`` per trait, plus a JSON metadata sidecar.

Reuses :func:`~sim_ace.analysis.stats.compute_cumulative_incidence_by_sex_generation`,
:class:`~sim_ace.core.pedigree_graph.PedigreeGraph`,
:data:`~sim_ace.core.pedigree_graph.PAIR_KINSHIP`,
:func:`~sim_ace.analysis.export_grm.export_sparse_grm_binary`, and
:func:`~sim_ace.simulation.simulate.generate_correlated_components`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from sim_ace.analysis.export_grm import export_sparse_grm_binary, require_cols
from sim_ace.analysis.stats import compute_cumulative_incidence_by_sex_generation
from sim_ace.core.pedigree_graph import REL_REGISTRY, PedigreeGraph
from sim_ace.core.utils import save_parquet
from sim_ace.simulation.simulate import generate_correlated_components

logger = logging.getLogger(__name__)

__all__ = [
    "assign_founder_family_ids",
    "export_cumulative_incidence",
    "export_pairwise_relatedness",
    "export_pgs",
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
    (``MZ``/``FS``/``PO``/``MHS``/``Av``/``1C``/…), ``kinship``.  Only
    pairs with kinship >= ``min_kinship`` are emitted.  The per-row check
    uses the kinship returned by ``PedigreeGraph.compute_pair_kinship``,
    which may exceed the nominal lookup under inbreeding.
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


def assign_founder_family_ids(
    pedigree_df: pd.DataFrame,
    graph: PedigreeGraph | None = None,
) -> pd.Series:
    """Assign a family id (FID) to every individual in the pedigree.

    Two individuals share an FID iff they belong to the same connected
    component of the parent-child graph — equivalently, they descend from
    (or are) the same founder couple, possibly linked through marriages
    between founder lineages.  The FID value is the smallest ``id`` inside
    each connected component, giving a deterministic, human-readable label.

    Pass *graph* when a :class:`PedigreeGraph` has already been built for
    the same DataFrame; its pre-remapped parent matrices are reused to
    avoid the redundant id→row translation.

    Returns:
        An ``int64`` :class:`pandas.Series` aligned to ``pedigree_df.index``.
    """
    n = len(pedigree_df)
    if n == 0:
        return pd.Series([], dtype=np.int64, name="FID")

    pg = graph if graph is not None else PedigreeGraph(pedigree_df)
    _, labels = connected_components(pg._Am + pg._Af, directed=False)

    ids = pedigree_df["id"].to_numpy()
    comp_min = np.full(labels.max() + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(comp_min, labels, ids)
    return pd.Series(comp_min[labels], index=pedigree_df.index, name="FID", dtype=np.int64)


# ---------------------------------------------------------------------------
# Sparse GRM
# ---------------------------------------------------------------------------


def export_sparse_grm(
    pedigree_df: pd.DataFrame,
    prefix: str | Path,
    threshold: float = 0.05,
) -> tuple[Path, Path]:
    """Export a sparse GRM in ``ace_sreml`` binary format with founder FIDs.

    ``threshold`` is on the GRM scale (kinship × 2), matching the writer's
    convention; ``0.05`` is sparseREML's default ``GRM_range[0]``.
    """
    pg = PedigreeGraph(pedigree_df)
    pg.compute_inbreeding()
    ids = pedigree_df["id"].to_numpy()
    fids = assign_founder_family_ids(pedigree_df, graph=pg).to_numpy()
    logger.info("export_sparse_grm: %d founder-couple FIDs", len(np.unique(fids)))
    return export_sparse_grm_binary(
        pg._kinship_matrix,
        iids=ids,
        prefix=prefix,
        to_grm=True,
        threshold=threshold,
        fids=fids,
    )


# ---------------------------------------------------------------------------
# Proxy polygenic score
# ---------------------------------------------------------------------------


def export_pgs(
    pedigree_df: pd.DataFrame,
    r2: tuple[float, float] | list[float],
    rA: float,
    var_A: tuple[float, float] | list[float],
    sub_seed: int,
    out_path: str | Path,
    meta_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a per-individual proxy polygenic score parquet plus JSON sidecar.

    The simulator has no SNPs, so this writes a *proxy* PGS that mimics what
    a real trained predictor would deliver, with a user-specified accuracy::

        PGS_{i,t} = sqrt(r²_t) · A_{i,t}
                    + sqrt(Var(A_t) · (1 − r²_t)) · e_{i,t}

    where ``(e_{i,1}, e_{i,2}) ~ N(0, Σ_e)`` with unit variances and
    correlation ``rA``, independent of ``A`` and i.i.d. across individuals.
    ``Var(A_t)`` is the *nominal* value from config (not rescaled to the
    realized sample variance), so ``E[Cor(PGS_t, A_t)²] = r²_t`` with
    finite-sample fluctuation around it.

    The noise correlation is tied to ``rA``, which means realized
    ``Cor(PGS_1, PGS_2) = rA · [sqrt(r1·r2) + sqrt((1-r1)(1-r2))]``.
    This collapses to ``rA`` only when ``r1 == r2``; for unequal accuracies
    the cross-PGS correlation is attenuated.

    Output parquet columns: ``id, sex, generation, A1, A2, PGS1, PGS2``
    (one row per individual in *pedigree_df*). The JSON sidecar records
    ``pgs_r2, rA, var_A, sub_seed, n`` plus the realized
    ``cor(PGS_t, A_t)`` and ``cor(PGS_1, PGS_2)`` so downstream tools can
    read the accuracy without parsing YAML config.

    Args:
        pedigree_df: must contain ``id, sex, generation, A1, A2``.
        r2: per-trait expected squared correlation ``(r²_1, r²_2)``;
            each in ``[0, 1]``.
        rA: genetic correlation between traits. Used both as the target
            correlation of the (A_1, A_2) signal (already set by the
            simulator) and as the correlation of the noise draws.
        var_A: nominal ``(Var(A_1), Var(A_2))`` from config.
        sub_seed: deterministic integer seed for the noise draw.
        out_path: output parquet path; parents are created if missing.
        meta_path: sidecar JSON path; defaults to *out_path* with its
            extension replaced by ``.meta.json``.

    Returns:
        ``(parquet_path, meta_path)``.
    """
    require_cols(pedigree_df, ["id", "sex", "generation", "A1", "A2"])
    r2_1, r2_2 = float(r2[0]), float(r2[1])
    var_1, var_2 = float(var_A[0]), float(var_A[1])
    for idx, r in enumerate((r2_1, r2_2)):
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"r2[{idx}] must be in [0, 1], got {r}")
    if not -1.0 <= rA <= 1.0:
        raise ValueError(f"rA must be in [-1, 1], got {rA}")
    if var_1 < 0 or var_2 < 0:
        raise ValueError(f"var_A entries must be non-negative, got {var_1}, {var_2}")

    n = len(pedigree_df)
    A1 = pedigree_df["A1"].to_numpy(dtype=np.float64)
    A2 = pedigree_df["A2"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(int(sub_seed))
    e1, e2 = generate_correlated_components(rng, n, 1.0, 1.0, float(rA))

    pgs1 = np.sqrt(r2_1) * A1 + np.sqrt(var_1 * (1.0 - r2_1)) * e1
    pgs2 = np.sqrt(r2_2) * A2 + np.sqrt(var_2 * (1.0 - r2_2)) * e2

    out_df = pd.DataFrame(
        {
            "id": pedigree_df["id"].to_numpy(),
            "sex": pedigree_df["sex"].to_numpy(),
            "generation": pedigree_df["generation"].to_numpy(),
            "A1": A1.astype(np.float32),
            "A2": A2.astype(np.float32),
            "PGS1": pgs1.astype(np.float32),
            "PGS2": pgs2.astype(np.float32),
        }
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(out_df, out_path)

    meta_path = Path(meta_path) if meta_path is not None else out_path.with_name(out_path.stem + ".meta.json")
    cor_pgs_a = (
        [float(np.corrcoef(pgs1, A1)[0, 1]), float(np.corrcoef(pgs2, A2)[0, 1])] if n > 1 else [float("nan")] * 2
    )
    cor_pgs_pgs = float(np.corrcoef(pgs1, pgs2)[0, 1]) if n > 1 else float("nan")
    meta = {
        "pgs_r2": [r2_1, r2_2],
        "rA": float(rA),
        "var_A": [var_1, var_2],
        "sub_seed": int(sub_seed),
        "n": int(n),
        "realized_cor_pgs_a": cor_pgs_a,
        "realized_cor_pgs1_pgs2": cor_pgs_pgs,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(
        "export_pgs: n=%d, r²=(%.3f, %.3f), realized cor(PGS,A)=(%.3f, %.3f) -> %s",
        n,
        r2_1,
        r2_2,
        cor_pgs_a[0],
        cor_pgs_a[1],
        out_path.name,
    )
    return out_path, meta_path
