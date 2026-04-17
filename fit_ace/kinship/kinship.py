"""Sparse kinship / expected A-matrix (2φ) builders for pedigrees.

Two construction paths:

- ``build_sparse_kinship``: generation-by-generation dynamic programming
  (mirrors ``kinship2::kinship`` in R).  Correct for arbitrary pedigrees
  including consanguinity.
- ``build_kinship_from_pairs``: enumerates relationship pairs via
  ``PedigreeGraph`` and assigns known coefficients from ``PAIR_KINSHIP``.
  Exact for non-consanguineous pedigrees and dramatically faster.

Both return ``scipy.sparse.csc_matrix`` in 2φ convention (self-kinship on
diagonal = 0.5 for non-inbred founders; MZ twin off-diagonal = 0.5).
"""

from __future__ import annotations

__all__ = [
    "build_kinship_from_pairs",
    "build_sparse_kinship",
]

import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sim_ace.core.pedigree_graph import PAIR_KINSHIP

logger = logging.getLogger(__name__)


def build_sparse_kinship(
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
    twins: np.ndarray | None = None,
    min_kinship: float = 1e-10,
) -> sp.csc_matrix:
    """Build a sparse kinship matrix from pedigree arrays.

    Uses the generation-by-generation dynamic programming algorithm
    (same as ``kinship2::kinship`` in R).  Founders (parent == -1) get
    self-kinship 0.5 and zero kinship with everyone else.

    Parameters
    ----------
    ids, mothers, fathers : int arrays (n,)
        Individual, mother, and father IDs.  -1 = missing/founder parent.
    twins : int array (n,) or None
        MZ twin partner ID (-1 if not a twin).
    min_kinship : float, default 1e-10
        During the DP propagation, off-diagonal kinships below this
        threshold are dropped rather than stored.  Raising this (e.g. to
        ``2**-10 ≈ 1e-3`` excludes beyond ~4th cousins) shrinks the
        intermediate dicts and speeds up deep pedigrees; the diagonal is
        always kept.  Value is on the **kinship scale** (max 0.5 for
        non-inbred individuals), not the GRM scale (2φ, max 1.0).

    Returns:
    -------
    scipy.sparse.csc_matrix (n, n), symmetric.
    """
    t0 = time.perf_counter()
    n = len(ids)

    # Map IDs → contiguous 0-based indices
    max_id = int(ids.max())
    id_to_idx = np.full(max_id + 2, -1, dtype=np.int32)  # +2 for safety
    id_to_idx[ids] = np.arange(n, dtype=np.int32)

    def _remap(arr):
        out = np.full(len(arr), -1, dtype=np.int32)
        valid = (arr >= 0) & (arr <= max_id)
        out[valid] = id_to_idx[arr[valid]]
        return out

    m_idx = _remap(mothers)
    f_idx = _remap(fathers)
    tw_idx = _remap(twins) if twins is not None else np.full(n, -1, dtype=np.int32)

    # Compute generation depth from pedigree structure
    depth = _compute_depth(m_idx, f_idx, n)
    max_depth = int(depth.max())

    # Build kinship using dict-of-dicts: kin[i] = {j: value}
    # Full symmetric storage (kin[i][j] == kin[j][i]) for correct iteration
    kin: list[dict[int, float]] = [{i: 0.5} for i in range(n)]

    t_dp = time.perf_counter()
    for d in range(1, max_depth + 1):
        t_gen = time.perf_counter()
        gen_indices = np.where(depth == d)[0]
        for j in gen_indices:
            m, f = int(m_idx[j]), int(f_idx[j])

            # Self-kinship: (1 + kinship(m, f)) / 2
            km_f = 0.0
            if m >= 0 and f >= 0:
                km_f = kin[m].get(f, 0.0)
            kin[j][j] = (1.0 + km_f) / 2.0

            # Kinship(j, k) = (kinship(m, k) + kinship(f, k)) / 2
            # kin[parent] contains ALL of that parent's relatives (symmetric)
            m_row = kin[m] if m >= 0 else {}
            f_row = kin[f] if f >= 0 else {}

            all_k = set(m_row.keys()) | set(f_row.keys())
            all_k.discard(j)
            for k in all_k:
                mk = m_row.get(k, 0.0)
                fk = f_row.get(k, 0.0)
                val = (mk + fk) / 2.0
                if val > min_kinship:
                    kin[j][k] = val
                    kin[k][j] = val  # symmetric

        # Handle MZ twins in this generation
        for j in gen_indices:
            tw = int(tw_idx[j])
            if tw >= 0 and tw != j:
                self_kin = kin[j].get(j, 0.5)
                kin[j][tw] = self_kin
                kin[tw][j] = self_kin

        avg_rels = np.mean([len(kin[j]) for j in gen_indices]) if len(gen_indices) > 0 else 0
        logger.info(
            "  gen %d: %d individuals, avg %.0f relatives, %.2fs",
            d,
            len(gen_indices),
            avg_rels,
            time.perf_counter() - t_gen,
        )

    dp_elapsed = time.perf_counter() - t_dp
    total_entries = sum(len(row) for row in kin)
    mem_mb = total_entries * 80 / 1e6  # ~80 bytes per Python dict entry
    logger.info("  DP phase: %.2fs, %d total dict entries (~%.0f MB)", dp_elapsed, total_entries, mem_mb)

    # Convert to COO then CSC
    t_conv = time.perf_counter()
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_key, v in kin[i].items():
            if j_key >= i:  # upper triangle only (avoid duplicates)
                rows.append(i)
                cols.append(j_key)
                vals.append(v)
                if i != j_key:
                    rows.append(j_key)
                    cols.append(i)
                    vals.append(v)

    del kin  # free dict memory before allocating arrays

    kmat = sp.csc_matrix(
        (np.array(vals), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n),
    )
    conv_elapsed = time.perf_counter() - t_conv
    elapsed = time.perf_counter() - t0
    nnz = len(vals)
    csc_mb = (kmat.data.nbytes + kmat.indices.nbytes + kmat.indptr.nbytes) / 1e6
    logger.info(
        "Kinship matrix: %d individuals, %d nnz, %.2fs (DP=%.2fs, convert=%.2fs, CSC=%.0f MB)",
        n,
        nnz,
        elapsed,
        dp_elapsed,
        conv_elapsed,
        csc_mb,
    )
    return kmat


def _compute_depth(m_idx: np.ndarray, f_idx: np.ndarray, n: int) -> np.ndarray:
    """Compute generation depth: founders=0, children=max(parent_depth)+1."""
    depth = np.full(n, -1, dtype=np.int32)
    # Founders: both parents missing
    founders = (m_idx < 0) & (f_idx < 0)
    depth[founders] = 0

    changed = True
    while changed:
        changed = False
        unset = np.where(depth < 0)[0]
        for j in unset:
            m, f = int(m_idx[j]), int(f_idx[j])
            md = depth[m] if m >= 0 else 0
            fd = depth[f] if f >= 0 else 0
            if md >= 0 and fd >= 0:
                depth[j] = max(md, fd) + 1
                changed = True

    # Any remaining unset → treat as depth 0 (disconnected founders)
    depth[depth < 0] = 0
    return depth


def build_kinship_from_pairs(
    pedigree_df: pd.DataFrame,
    ndegree: int = 2,
    full_pedigree: pd.DataFrame | None = None,
) -> sp.csc_matrix:
    """Build sparse kinship matrix from extracted relationship pairs.

    Uses ``PedigreeGraph.extract_pairs()`` to enumerate all relationship
    pairs and assigns known kinship coefficients.  Exact for ACE's
    non-consanguineous pedigrees and dramatically faster than the
    generation-by-generation DP (``build_sparse_kinship``).

    Parameters
    ----------
    pedigree_df : pedigree or phenotype DataFrame (rows to score)
    ndegree : relationship degree cutoff (2 = up to 1st cousins)
    full_pedigree : full pedigree if pedigree_df is a subset

    Returns:
    -------
    scipy.sparse.csc_matrix (n, n), symmetric.
    """
    from sim_ace.core.pedigree_graph import PedigreeGraph

    t0 = time.perf_counter()
    n = len(pedigree_df)
    kin_threshold = 0.5 ** (ndegree + 1) - 1e-6

    graph = PedigreeGraph(pedigree_df)
    pairs = graph.extract_pairs(max_degree=ndegree, min_kinship=kin_threshold)

    # Collect all (row, col, kinship) triplets
    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []

    for pair_type, (idx1, idx2) in pairs.items():
        kin_coeff = PAIR_KINSHIP.get(pair_type)
        if kin_coeff is None or kin_coeff < kin_threshold:
            continue
        k = len(idx1)
        if k == 0:
            continue
        vals = np.full(k, kin_coeff)
        # Add both directions for symmetry
        all_rows.append(idx1)
        all_cols.append(idx2)
        all_vals.append(vals)
        all_rows.append(idx2)
        all_cols.append(idx1)
        all_vals.append(vals)

    # Self-kinship = 0.5 on diagonal
    diag_idx = np.arange(n, dtype=np.int32)
    all_rows.append(diag_idx)
    all_cols.append(diag_idx)
    all_vals.append(np.full(n, 0.5))

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)

    # Deduplicate: some individuals appear in multiple pair types
    # (e.g., both avuncular and parent-offspring through different paths).
    # COO sums duplicates; we want the maximum kinship per pair.
    kmat = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    kmat = kmat.tocsr()
    # CSR sums duplicates too; fix by capping off-diagonal at 0.5
    kmat.data = np.minimum(kmat.data, 0.5)
    kmat = kmat.tocsc()

    elapsed = time.perf_counter() - t0
    nnz = kmat.nnz
    csc_mb = (kmat.data.nbytes + kmat.indices.nbytes + kmat.indptr.nbytes) / 1e6
    logger.info(
        "Kinship (pair-based): %d individuals, %d nnz, %.2fs, CSC=%.0f MB",
        n,
        nnz,
        elapsed,
        csc_mb,
    )
    return kmat
