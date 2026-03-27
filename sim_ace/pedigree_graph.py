"""
Pedigree relationship extraction via sparse matrix products.

Builds parent→child CSR matrices and extracts 10 relationship categories
using sparse matrix algebra (A @ A.T for siblings, A² @ (A²).T for cousins, etc.).
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class PedigreeGraph:
    """Parent→child DAG for efficient relationship queries.

    Each individual is a vertex whose index equals its DataFrame row index.
    Sparse CSR matrices encode parent-child edges for O(nnz) relationship
    extraction via matrix products.

    Args:
        df: Pedigree DataFrame with columns id, mother, father, twin, sex, generation.
    """

    def __init__(self, df: pd.DataFrame, sample_mask: np.ndarray | None = None) -> None:
        n = len(df)
        self.n = n

        # Optional boolean mask: True = "active" individual (in the sample).
        # When set, extract_pairs only returns pairs of active individuals.
        self._active = sample_mask

        # Build ID → row index mapping
        ids_arr = df["id"].values.astype(np.int64)
        self._ids = ids_arr
        id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int64)
        id_to_row[ids_arr] = np.arange(n, dtype=np.int64)

        def _remap(col_vals: np.ndarray) -> np.ndarray:
            """Map IDs to row indices; -1 stays -1."""
            out = np.full(len(col_vals), -1, dtype=np.int64)
            valid = (col_vals >= 0) & (col_vals < len(id_to_row))
            out[valid] = id_to_row[col_vals[valid]]
            return out

        # Original pedigree parent IDs (for sibling classification —
        # valid even when the parent isn't in the sample).
        self._orig_mother = df["mother"].values.astype(np.int64)
        self._orig_father = df["father"].values.astype(np.int64)

        # Remap parent/twin IDs to row indices (for sparse matrices)
        self.mother = _remap(self._orig_mother)
        self.father = _remap(self._orig_father)
        self.twin = _remap(df["twin"].values.astype(np.int64))
        self.sex = df["sex"].values.astype(np.int32)
        self.generation = df["generation"].values.astype(np.int32)

        # Build parent→child matrices using ALL available edges.
        # Each matrix is built independently so partial-pedigree data
        # (e.g. after subsampling) still contributes edges.
        m_mask = self.mother >= 0
        m_idx = np.where(m_mask)[0]
        f_mask = self.father >= 0
        f_idx = np.where(f_mask)[0]

        self._Am = sp.csr_matrix(
            (np.ones(len(m_idx), dtype=np.float64), (m_idx, self.mother[m_idx])),
            shape=(n, n),
        )
        self._Af = sp.csr_matrix(
            (np.ones(len(f_idx), dtype=np.float64), (f_idx, self.father[f_idx])),
            shape=(n, n),
        )

    # ------------------------------------------------------------------
    # Lazy sparse products (computed on first access)
    # ------------------------------------------------------------------

    @cached_property
    def _A(self):
        """Child → both parents adjacency matrix."""
        t0 = time.perf_counter()
        result = self._Am + self._Af
        logger.debug("_A (Am + Af) computed in %.3fs", time.perf_counter() - t0)
        return result

    @cached_property
    def _S(self):
        """Shared-any-parent matrix: A @ A.T."""
        t0 = time.perf_counter()
        result = self._A @ self._A.T
        logger.debug("_S = A @ A.T computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    @cached_property
    def _A2(self):
        """2-hop parent reach (grandparents): A @ A."""
        t0 = time.perf_counter()
        result = self._A @ self._A
        logger.debug("_A2 = A @ A computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    @cached_property
    def _A2_shared(self):
        """Shared-grandparent matrix: A² @ (A²).T.

        Only needed when 2nd cousin extraction is enabled.
        """
        t0 = time.perf_counter()
        result = self._A2 @ self._A2.T
        logger.debug("_A2_shared = A2 @ A2.T computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    @cached_property
    def _A3(self):
        """3-hop parent reach (great-grandparents): A² @ A."""
        t0 = time.perf_counter()
        result = self._A2 @ self._A
        logger.debug("_A3 = A2 @ A computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    # ------------------------------------------------------------------
    # Relationship extraction
    # ------------------------------------------------------------------

    def _mz_twin_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """MZ twin pairs: twin != -1, deduplicated with id < twin_id."""
        has_twin = self.twin >= 0
        ids = np.where(has_twin)[0]
        partners = self.twin[has_twin]
        mask = ids < partners
        return ids[mask], partners[mask].astype(np.intp)

    def _parent_offspring_pairs(
        self,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """Mother-offspring and Father-offspring pairs.

        Each parent link is reported independently, so a child with only
        one parent in the sample still contributes a PO pair.
        """
        m_mask = self.mother >= 0
        m_children = np.where(m_mask)[0]

        f_mask = self.father >= 0
        f_children = np.where(f_mask)[0]

        return (m_children, self.mother[m_children].astype(np.intp)), (
            f_children,
            self.father[f_children].astype(np.intp),
        )

    def _sibling_pairs(
        self,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """Full sib, maternal half sib, and paternal half sib pairs.

        Uses numpy sort+group for direct enumeration — faster than sparse
        matmul for 1-hop relationships since it avoids materializing N×N
        shared-parent matrices.

        Groups by ORIGINAL pedigree parent IDs (not remapped row indices)
        so that siblings are correctly detected even when parents are absent
        from a subsampled dataset.

        Individuals with only one known parent can participate in half-sib
        detection through that parent (but not full-sib detection, which
        requires both parents known).

        Twin individuals are excluded entirely (matching legacy semantics).
        Returns (full_sib, maternal_hs, paternal_hs) tuples of (idx1, idx2).
        """
        empty = np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        # Non-twin individuals with at least one known parent
        has_parent = (self._orig_mother >= 0) | (self._orig_father >= 0)
        nt_mask = has_parent & (self.twin < 0)
        nt_idx = np.where(nt_mask)[0]

        if len(nt_idx) < 2:
            self._full_sib_matrix = sp.csr_matrix((self.n, self.n))
            return empty, empty, empty

        nt_mother = self._orig_mother[nt_idx]
        nt_father = self._orig_father[nt_idx]

        # --- Full sibs: same KNOWN mother AND same KNOWN father ---
        both_known = (nt_mother >= 0) & (nt_father >= 0)
        bk_idx = nt_idx[both_known]
        bk_mother = nt_mother[both_known]
        bk_father = nt_father[both_known]

        if len(bk_idx) >= 2:
            max_parent = max(int(bk_mother.max()), int(bk_father.max())) + 1
            family_key = bk_mother.astype(np.int64) * max_parent + bk_father.astype(np.int64)
            full_sib = self._pairs_from_groups(bk_idx, family_key)
        else:
            full_sib = empty

        # --- Maternal half sibs: all pairs sharing known mother, minus full-sib pairs ---
        has_mother = nt_mother >= 0
        m_idx = nt_idx[has_mother]
        m_mother = nt_mother[has_mother]
        if len(m_idx) >= 2:
            mat_all = self._pairs_from_groups(m_idx, m_mother.astype(np.int64))
            mat_hs = self._subtract_pairs(mat_all, full_sib)
        else:
            mat_hs = empty

        # --- Paternal half sibs: all pairs sharing known father, minus full-sib pairs ---
        has_father = nt_father >= 0
        f_idx = nt_idx[has_father]
        f_father = nt_father[has_father]
        if len(f_idx) >= 2:
            pat_all = self._pairs_from_groups(f_idx, f_father.astype(np.int64))
            pat_hs = self._subtract_pairs(pat_all, full_sib)
        else:
            pat_hs = empty

        # Build full-sib sparse matrix for _avuncular_pairs
        sib1, sib2 = full_sib
        if len(sib1) > 0:
            ones = np.ones(len(sib1), dtype=np.float64)
            F = sp.csr_matrix((ones, (sib1, sib2)), shape=(self.n, self.n))
            self._full_sib_matrix = F + F.T
        else:
            self._full_sib_matrix = sp.csr_matrix((self.n, self.n))

        return full_sib, mat_hs, pat_hs

    @staticmethod
    def _subtract_pairs(
        all_pairs: tuple[np.ndarray, np.ndarray],
        remove_pairs: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove pairs in *remove_pairs* from *all_pairs* using set subtraction.

        Both inputs must be canonically ordered (lo, hi).
        Encodes each pair as lo * max_id + hi for O(1) lookup.
        """
        a1, a2 = all_pairs
        r1, r2 = remove_pairs

        if len(a1) == 0:
            return all_pairs
        if len(r1) == 0:
            return all_pairs

        max_id = int(max(a1.max(), a2.max(), r1.max(), r2.max())) + 1
        remove_keys = r1.astype(np.int64) * max_id + r2.astype(np.int64)
        all_keys = a1.astype(np.int64) * max_id + a2.astype(np.int64)

        keep = ~np.isin(all_keys, remove_keys)

        return a1[keep].astype(np.intp), a2[keep].astype(np.intp)

    @staticmethod
    def _pairs_from_groups(indices: np.ndarray, group_key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate all (i < j) pairs of indices within each group.

        Uses batch-by-size triu_indices for vectorized pair generation.
        """
        sort_idx = np.argsort(group_key, kind="mergesort")
        sorted_keys = group_key[sort_idx]
        sorted_indices = indices[sort_idx]

        _, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

        multi = counts >= 2
        starts = starts[multi]
        counts = counts[multi]

        if len(starts) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        pair_i_parts = []
        pair_j_parts = []
        for size in np.unique(counts):
            gs = starts[counts == size]
            ii, jj = np.triu_indices(size, k=1)
            all_i = (gs[:, np.newaxis] + ii[np.newaxis, :]).ravel()
            all_j = (gs[:, np.newaxis] + jj[np.newaxis, :]).ravel()
            pair_i_parts.append(sorted_indices[all_i])
            pair_j_parts.append(sorted_indices[all_j])

        p1 = np.concatenate(pair_i_parts)
        p2 = np.concatenate(pair_j_parts)

        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        return lo.astype(np.intp), hi.astype(np.intp)

    @staticmethod
    def _pairs_from_groups_filtered(
        indices: np.ndarray,
        group_key: np.ndarray,
        filter_key: np.ndarray,
        keep_same: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate pairs within groups, filtered by a secondary key.

        Like _pairs_from_groups but also checks the filter_key for each pair
        and keeps only pairs where filter_key matches (keep_same=True) or
        differs (keep_same=False).
        """
        sort_idx = np.argsort(group_key, kind="mergesort")
        sorted_keys = group_key[sort_idx]
        sorted_indices = indices[sort_idx]
        sorted_filter = filter_key[sort_idx]

        _, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

        multi = counts >= 2
        starts = starts[multi]
        counts = counts[multi]

        if len(starts) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        pair_i_parts = []
        pair_j_parts = []
        for size in np.unique(counts):
            gs = starts[counts == size]
            ii, jj = np.triu_indices(size, k=1)
            all_i = (gs[:, np.newaxis] + ii[np.newaxis, :]).ravel()
            all_j = (gs[:, np.newaxis] + jj[np.newaxis, :]).ravel()

            if keep_same:
                mask = sorted_filter[all_i] == sorted_filter[all_j]
            else:
                mask = sorted_filter[all_i] != sorted_filter[all_j]

            pair_i_parts.append(sorted_indices[all_i[mask]])
            pair_j_parts.append(sorted_indices[all_j[mask]])

        if not pair_i_parts:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        p1 = np.concatenate(pair_i_parts)
        p2 = np.concatenate(pair_j_parts)

        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        return lo.astype(np.intp), hi.astype(np.intp)

    def _cousin_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """1st cousin pairs: share a grandparent but not a parent.

        Uses group-by-grandparent enumeration instead of A² @ (A²).T matmul.
        Groups grandchildren by each grandparent, enumerates within-group pairs,
        removes sibling pairs, and deduplicates.
        """
        t0 = time.perf_counter()
        gc_i, gp_j = self._A2.nonzero()
        if len(gc_i) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        # Enumerate all (i < j) pairs sharing a grandparent
        p1, p2 = self._pairs_from_groups(gc_i.astype(np.intp), gp_j.astype(np.int64))
        if len(p1) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        logger.debug(
            "Cousin group-by: %d candidate pairs from %d edges (%.3fs)",
            len(p1),
            len(gc_i),
            time.perf_counter() - t0,
        )

        # Remove sibling/half-sib pairs (those sharing a parent)
        share_mother = (self._orig_mother[p1] >= 0) & (self._orig_mother[p1] == self._orig_mother[p2])
        share_father = (self._orig_father[p1] >= 0) & (self._orig_father[p1] == self._orig_father[p2])
        is_sib = share_mother | share_father
        p1, p2 = p1[~is_sib], p2[~is_sib]

        if len(p1) == 0:
            logger.debug("Cousins: 0 pairs after sibling removal (%.3fs)", time.perf_counter() - t0)
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        # Deduplicate (two cousins may share multiple grandparents)
        max_id = max(int(p1.max()), int(p2.max())) + 1
        keys = p1.astype(np.int64) * max_id + p2.astype(np.int64)
        _, unique_idx = np.unique(keys, return_index=True)

        result = p1[unique_idx].astype(np.intp), p2[unique_idx].astype(np.intp)
        logger.debug("Cousins: %d unique pairs (%.3fs)", len(result[0]), time.perf_counter() - t0)
        return result

    def _grandparent_grandchild_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Grandparent-grandchild pairs: 2-hop ancestor links.

        A2[child, grandparent] > 0 iff grandparent is a 2-hop ancestor.
        """
        gc_i, gp_j = self._A2.nonzero()

        if len(gc_i) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        return gc_i.astype(np.intp), gp_j.astype(np.intp)

    def _avuncular_pairs(self, full_sib: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Avuncular (uncle/aunt-nephew/niece) pairs.

        An avuncular pair (child C, uncle U) exists when C's parent P is a
        full sibling of U. In matrix form: A @ S_full, then exclude
        parent-child pairs (which share the same edge structure).
        """
        # Get the full-sib symmetric matrix (cached by _sibling_pairs)
        if not hasattr(self, "_full_sib_matrix"):
            # Build it if _sibling_pairs wasn't called first
            sib1, sib2 = full_sib
            if len(sib1) == 0:
                return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
            n = self.n
            ones = np.ones(len(sib1), dtype=np.float64)
            F = sp.csr_matrix((ones, (sib1, sib2)), shape=(n, n))
            full_sib_mat = F + F.T
        else:
            full_sib_mat = self._full_sib_matrix

        avunc = self._A @ full_sib_mat
        avunc.setdiag(0)

        # Exclude parent-child pairs: A[child, parent] = 1, so A + A.T marks
        # all parent-child edges in both directions
        parent_child = (self._A + self._A.T) > 0
        avunc = avunc - avunc.multiply(parent_child)
        avunc.eliminate_zeros()

        if avunc.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        a_i, a_j = avunc.nonzero()
        # Canonical ordering: smaller index first
        lo = np.minimum(a_i, a_j)
        hi = np.maximum(a_i, a_j)

        # Dedup: pack into int64 keys, sort+diff
        max_id = int(hi.max()) + 1
        keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
        sort_idx = np.argsort(keys, kind="mergesort")
        keys_sorted = keys[sort_idx]
        mask = np.empty(len(keys_sorted), dtype=bool)
        mask[0] = True
        mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
        kept = sort_idx[mask]

        return lo[kept].astype(np.intp), hi[kept].astype(np.intp)

    def _second_cousin_matrix(self) -> sp.spmatrix:
        """Symmetric sparse matrix with nonzeros at 2nd cousin pairs.

        Shared-great-grandparent pairs minus shared-grandparent pairs.
        """
        t0 = time.perf_counter()
        D_raw = self._A3 @ self._A3.T
        logger.debug("A3 @ A3.T computed in %.3fs (nnz=%d)", time.perf_counter() - t0, D_raw.nnz)
        D_bool = (D_raw > 0).astype(np.float64)

        C_bool = (self._A2_shared > 0).astype(np.float64)

        second_cousins = D_bool - D_bool.multiply(C_bool)
        second_cousins.setdiag(0)
        second_cousins.eliminate_zeros()
        logger.debug("2nd cousin matrix: nnz=%d (%.3fs total)", second_cousins.nnz, time.perf_counter() - t0)
        return second_cousins

    def _second_cousin_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """2nd cousin pairs: share a great-grandparent but not a grandparent."""
        second_cousins = self._second_cousin_matrix()

        sc_upper = sp.triu(second_cousins, k=1)
        sc_i, sc_j = sc_upper.nonzero()

        if len(sc_i) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        return sc_i.astype(np.intp), sc_j.astype(np.intp)

    def extract_pairs(
        self,
        seed: int = 42,
        skip_2nd_cousins: bool = True,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Extract all relationship categories.

        When ``sample_mask`` was provided at construction, only pairs where
        both individuals are active are returned.

        Args:
            seed: Random seed (unused — kept for API compatibility).
            skip_2nd_cousins: When True, skip 2nd cousin extraction (avoids
                expensive A³ and A³ @ (A³).T matrix products).

        Returns:
            Dict mapping relationship name to (idx1, idx2) row-index arrays.
        """
        t_total = time.perf_counter()
        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        pairs["MZ twin"] = self._mz_twin_pairs()

        mo, fo = self._parent_offspring_pairs()
        pairs["Mother-offspring"] = mo
        pairs["Father-offspring"] = fo

        t0 = time.perf_counter()
        full_sib, mat_hs, pat_hs = self._sibling_pairs()
        pairs["Full sib"] = full_sib
        pairs["Maternal half sib"] = mat_hs
        pairs["Paternal half sib"] = pat_hs
        logger.info(
            "Siblings: %d full, %d maternal HS, %d paternal HS (%.3fs)",
            len(full_sib[0]),
            len(mat_hs[0]),
            len(pat_hs[0]),
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        pairs["1st cousin"] = self._cousin_pairs()
        logger.info("1st cousins: %d pairs (%.3fs)", len(pairs["1st cousin"][0]), time.perf_counter() - t0)

        pairs["Grandparent-grandchild"] = self._grandparent_grandchild_pairs()
        logger.info("Grandparent-grandchild: %d pairs", len(pairs["Grandparent-grandchild"][0]))

        t0 = time.perf_counter()
        pairs["Avuncular"] = self._avuncular_pairs(full_sib)
        logger.info("Avuncular: %d pairs (%.3fs)", len(pairs["Avuncular"][0]), time.perf_counter() - t0)

        empty = np.array([], dtype=np.intp)
        if skip_2nd_cousins:
            pairs["2nd cousin"] = (empty, empty)
            logger.info("2nd cousins: skipped (skip_2nd_cousins=True)")
        else:
            t0 = time.perf_counter()
            pairs["2nd cousin"] = self._second_cousin_pairs()
            logger.info("2nd cousins: %d pairs (%.3fs)", len(pairs["2nd cousin"][0]), time.perf_counter() - t0)

        # Save raw counts before sample_mask filtering (used by count_pairs)
        self._raw_pair_counts = {k: len(v[0]) for k, v in pairs.items()}

        # Restrict to active (sampled) individuals when a mask is set
        if self._active is not None:
            for k, (idx1, idx2) in pairs.items():
                if len(idx1) > 0:
                    mask = self._active[idx1] & self._active[idx2]
                    pairs[k] = (idx1[mask].astype(np.intp), idx2[mask].astype(np.intp))
                else:
                    pairs[k] = (empty, empty)
            logger.info(
                "Filtered to sample_mask: %s",
                ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()),
            )

        logger.info("extract_pairs total: %.3fs", time.perf_counter() - t_total)
        return pairs

    def count_pairs(self, skip_2nd_cousins: bool = True) -> dict[str, int]:
        """Count all relationship categories.

        If ``extract_pairs()`` was already called on this instance, returns
        the cached pre-filter counts (nearly free).  Otherwise computes
        counts from scratch using the same methods as ``extract_pairs()``.
        """
        if hasattr(self, "_raw_pair_counts"):
            return dict(self._raw_pair_counts)

        # Compute from scratch
        counts: dict[str, int] = {}

        counts["MZ twin"] = len(self._mz_twin_pairs()[0])

        mo, fo = self._parent_offspring_pairs()
        counts["Mother-offspring"] = len(mo[0])
        counts["Father-offspring"] = len(fo[0])

        full_sib, mat_hs, pat_hs = self._sibling_pairs()
        counts["Full sib"] = len(full_sib[0])
        counts["Maternal half sib"] = len(mat_hs[0])
        counts["Paternal half sib"] = len(pat_hs[0])

        logger.info(
            "Siblings: %d full, %d maternal HS, %d paternal HS",
            counts["Full sib"],
            counts["Maternal half sib"],
            counts["Paternal half sib"],
        )

        counts["1st cousin"] = len(self._cousin_pairs()[0])
        logger.info("1st cousins: %d pairs", counts["1st cousin"])

        counts["Grandparent-grandchild"] = len(self._grandparent_grandchild_pairs()[0])
        logger.info("Grandparent-grandchild: %d pairs", counts["Grandparent-grandchild"])

        counts["Avuncular"] = len(self._avuncular_pairs(full_sib)[0])
        logger.info("Avuncular: %d pairs", counts["Avuncular"])

        if skip_2nd_cousins:
            counts["2nd cousin"] = 0
            logger.info("2nd cousins: skipped")
        else:
            counts["2nd cousin"] = len(self._second_cousin_pairs()[0])
            logger.info("2nd cousins: %d pairs", counts["2nd cousin"])

        self._raw_pair_counts = counts
        return dict(counts)


def _build_graph_and_remap(
    df: pd.DataFrame,
    full_pedigree: pd.DataFrame,
    seed: int,
    skip_2nd_cousins: bool,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int], np.ndarray]:
    """Build PedigreeGraph on full pedigree, extract pairs, remap to df indices.

    Returns (remapped_pairs, full_pedigree_counts, full_ids).
    """
    pheno_ids = set(df["id"].values.astype(np.int64).tolist())
    full_ids = full_pedigree["id"].values.astype(np.int64)
    sample_mask = np.array([int(x) in pheno_ids for x in full_ids], dtype=bool)

    pg = PedigreeGraph(full_pedigree, sample_mask=sample_mask)
    raw_pairs = pg.extract_pairs(seed=seed, skip_2nd_cousins=skip_2nd_cousins)
    full_counts = dict(pg._raw_pair_counts)

    # Remap full-pedigree row indices → phenotype (df) row indices
    pheno_id_arr = df["id"].values.astype(np.int64)
    max_id = int(max(full_ids.max(), pheno_id_arr.max())) + 1
    id_to_pheno = np.full(max_id, -1, dtype=np.int64)
    id_to_pheno[pheno_id_arr] = np.arange(len(df), dtype=np.int64)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ptype, (idx1, idx2) in raw_pairs.items():
        if len(idx1) > 0:
            ids1 = full_ids[idx1]
            ids2 = full_ids[idx2]
            result[ptype] = (
                id_to_pheno[ids1].astype(np.intp),
                id_to_pheno[ids2].astype(np.intp),
            )
        else:
            result[ptype] = (np.array([], dtype=np.intp), np.array([], dtype=np.intp))

    return result, full_counts, full_ids


def extract_and_count_relationship_pairs(
    df: pd.DataFrame,
    seed: int = 42,
    full_pedigree: pd.DataFrame | None = None,
    skip_2nd_cousins: bool = True,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int] | None]:
    """Extract relationship pairs and count full-pedigree pairs in one pass.

    Builds a single PedigreeGraph; the cached matrix products computed during
    extraction make the subsequent count nearly free.

    Returns:
        (pairs_dict, full_counts_dict).  full_counts_dict is None when
        *full_pedigree* is not provided.
    """
    if full_pedigree is not None:
        pairs, full_counts, _ = _build_graph_and_remap(
            df,
            full_pedigree,
            seed,
            skip_2nd_cousins,
        )
        return pairs, full_counts

    pg = PedigreeGraph(df)
    pairs = pg.extract_pairs(seed=seed, skip_2nd_cousins=skip_2nd_cousins)
    return pairs, None


def extract_relationship_pairs(
    df: pd.DataFrame,
    seed: int = 42,
    full_pedigree: pd.DataFrame | None = None,
    skip_2nd_cousins: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract relationship pairs, returning row indices into *df*.

    When *full_pedigree* is provided the graph is built from the complete
    pedigree (all G_ped generations) so that multi-hop relationships
    (grandparent, avuncular, cousin) are detected through ancestors not
    present in *df*.  Output pairs are filtered to individuals in *df*
    and remapped to *df* row indices.

    Returns dict with 10 keys (the original 7 plus Grandparent-grandchild,
    Avuncular, and 2nd cousin).
    """
    pairs, _ = extract_and_count_relationship_pairs(
        df,
        seed=seed,
        full_pedigree=full_pedigree,
        skip_2nd_cousins=skip_2nd_cousins,
    )
    return pairs


def count_relationship_pairs(
    df: pd.DataFrame,
    skip_2nd_cousins: bool = True,
) -> dict[str, int]:
    """Count relationship pairs without materializing full index arrays.

    Memory-efficient alternative to extract_relationship_pairs() when only
    pair counts are needed (e.g. for the full pedigree summary).
    """
    pg = PedigreeGraph(df)
    return pg.count_pairs(skip_2nd_cousins=skip_2nd_cousins)


def extract_sibling_pairs(
    df: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract only sibling pairs (full, maternal HS, paternal HS).

    Much cheaper than extract_relationship_pairs — skips cousin,
    grandparent, avuncular, and 2nd cousin extraction.
    """
    pg = PedigreeGraph(df)
    full_sib, mat_hs, pat_hs = pg._sibling_pairs()
    return {
        "Full sib": full_sib,
        "Maternal half sib": mat_hs,
        "Paternal half sib": pat_hs,
    }


def count_sib_pairs(df: pd.DataFrame) -> dict[str, int]:
    """Drop-in replacement for validate._count_sib_pairs.

    Accepts a DataFrame of non-twin non-founders with columns id, mother, father.
    """
    # Build a minimal full DataFrame for PedigreeGraph
    # The input is a subset; we need to create a graph over the full ID space
    # Instead, replicate the counting logic directly without building a full graph
    _twin_col = df["twin"].values if "twin" in df.columns else np.full(len(df), -1, dtype=np.int64)
    mother_col = df["mother"].values.astype(np.int64)
    father_col = df["father"].values.astype(np.int64)

    n_full_sib = 0
    n_maternal_hs = 0
    n_paternal_hs = 0
    n_offspring_with_sibs = 0
    n_offspring_with_maternal_hs = 0

    # Group by mother
    ids = df["id"].values
    sort_m = np.argsort(mother_col, kind="mergesort")
    sorted_mothers = mother_col[sort_m]
    _sorted_ids = ids[sort_m]
    sorted_fathers = father_col[sort_m]

    u_m, starts_m, counts_m = np.unique(sorted_mothers, return_index=True, return_counts=True)

    for i in range(len(u_m)):
        s, c = starts_m[i], counts_m[i]
        group_fathers = sorted_fathers[s : s + c]

        if c < 2:
            continue

        n_offspring_with_sibs += c

        u_f, f_counts = np.unique(group_fathers, return_counts=True)
        for fc in f_counts:
            if fc >= 2:
                n_full_sib += fc * (fc - 1) // 2

        if len(u_f) >= 2:
            total_pairs = c * (c - 1) // 2
            full_pairs = sum(fc * (fc - 1) // 2 for fc in f_counts)
            n_maternal_hs += total_pairs - full_pairs
            n_offspring_with_maternal_hs += c

    # Group by father for paternal half sibs
    sort_f = np.argsort(father_col, kind="mergesort")
    sorted_fathers2 = father_col[sort_f]
    sorted_mothers2 = mother_col[sort_f]

    u_f, starts_f, counts_f = np.unique(sorted_fathers2, return_index=True, return_counts=True)

    for i in range(len(u_f)):
        s, c = starts_f[i], counts_f[i]
        if c < 2:
            continue

        group_mothers = sorted_mothers2[s : s + c]
        u_m2, m_counts = np.unique(group_mothers, return_counts=True)
        if len(u_m2) < 2:
            continue

        total_pairs = c * (c - 1) // 2
        full_pairs = sum(mc * (mc - 1) // 2 for mc in m_counts)
        n_paternal_hs += total_pairs - full_pairs

    return {
        "n_maternal_half_sib_pairs": n_maternal_hs,
        "n_paternal_half_sib_pairs": n_paternal_hs,
        "n_full_sib_pairs": n_full_sib,
        "n_offspring_with_maternal_half_sib": n_offspring_with_maternal_hs,
        "n_offspring_with_sibs": n_offspring_with_sibs,
    }
