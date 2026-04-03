"""
Pedigree relationship extraction via sparse matrix products.

Builds parent→child CSR matrices and extracts relationship categories
using sparse matrix algebra (A @ A.T for siblings, A² @ (A²).T for cousins, etc.).

Each relationship type is parameterised by (up, down, n_ancestors):
  - up:   meioses from individual A up to common ancestor(s), canonicalised up ≤ down
  - down: meioses from common ancestor(s) down to individual B
  - n_ancestors: 1 (half / lineal) or 2 (full, i.e. mated pair)
  - kinship = n_ancestors × (1/2)^(up + down + 1)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relationship type registry
# ---------------------------------------------------------------------------


class RelType(NamedTuple):
    """Relationship category defined by path through pedigree."""

    up: int  # meioses A → common ancestor(s)
    down: int  # meioses common ancestor(s) → B
    n_anc: int  # 1 = half/lineal, 2 = full (mated-pair ancestors)
    code: str  # short dict key
    label: str  # human-readable display label

    @property
    def kinship(self) -> float:
        if self.code == "MZ":
            return 0.5
        return self.n_anc * 0.5 ** (self.up + self.down + 1)

    @property
    def degree(self) -> int:
        if self.code == "MZ":
            return 0
        return round(-1 - np.log2(self.kinship))


# Ordered registry: kinship-descending, degree-ascending.
# MZ twins are a special case (up=down=n_anc=0).
REL_REGISTRY: dict[str, RelType] = {}
for _rt in [
    # --- special ---
    RelType(0, 0, 0, "MZ", "MZ twin"),
    # --- degree 1 (kinship 1/4) ---
    RelType(1, 0, 1, "MO", "Mother-offspring"),
    RelType(1, 0, 1, "FO", "Father-offspring"),
    RelType(1, 1, 2, "FS", "Full sib"),
    # --- degree 2 (kinship 1/8) ---
    RelType(1, 1, 1, "MHS", "Maternal half sib"),
    RelType(1, 1, 1, "PHS", "Paternal half sib"),
    RelType(2, 0, 1, "GP", "Grandparent"),
    RelType(1, 2, 2, "Av", "Avuncular"),
    # --- degree 3 (kinship 1/16) ---
    RelType(3, 0, 1, "GGP", "Great-grandparent"),
    RelType(1, 2, 1, "HAv", "Half-avuncular"),
    RelType(1, 3, 2, "GAv", "Great-avuncular"),
    RelType(2, 2, 2, "1C", "1st cousin"),
    # --- degree 4 (kinship 1/32) ---
    RelType(4, 0, 1, "GGGP", "Great\u00b2-grandparent"),
    RelType(1, 3, 1, "HGAv", "Half-great-avuncular"),
    RelType(1, 4, 2, "GGAv", "Great\u00b2-avuncular"),
    RelType(2, 2, 1, "H1C", "Half-1st-cousin"),
    RelType(2, 3, 2, "1C1R", "1st cousin 1R"),
    # --- degree 5 (kinship 1/64) ---
    RelType(5, 0, 1, "G3GP", "Great\u00b3-grandparent"),
    RelType(1, 4, 1, "HGGAv", "Half-great\u00b2-avuncular"),
    RelType(1, 5, 2, "G3Av", "Great\u00b3-avuncular"),
    RelType(2, 3, 1, "H1C1R", "Half-1st-cousin 1R"),
    RelType(2, 4, 2, "1C2R", "1st cousin 2R"),
    RelType(3, 3, 2, "2C", "2nd cousin"),
]:
    REL_REGISTRY[_rt.code] = _rt

# Kinship lookup by code — single source of truth for all consumers
PAIR_KINSHIP: dict[str, float] = {rt.code: rt.kinship for rt in REL_REGISTRY.values()}


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

        # Build ID → row index mapping (int32: row indices fit in 2.1B)
        ids_arr = df["id"].values
        self._ids = ids_arr
        if n > np.iinfo(np.int32).max:
            raise ValueError(f"Pedigree has {n:,} rows, exceeding int32 max for row indices")
        id_to_row = np.full(int(ids_arr.max()) + 1, -1, dtype=np.int32)
        id_to_row[ids_arr] = np.arange(n, dtype=np.int32)

        def _remap(col_vals: np.ndarray) -> np.ndarray:
            """Map IDs to row indices; -1 stays -1."""
            out = np.full(len(col_vals), -1, dtype=np.int32)
            valid = (col_vals >= 0) & (col_vals < len(id_to_row))
            out[valid] = id_to_row[col_vals[valid]]
            return out

        # Original pedigree parent IDs (for sibling classification —
        # valid even when the parent isn't in the sample).
        self._orig_mother = df["mother"].values
        self._orig_father = df["father"].values

        # Remap parent/twin IDs to row indices (for sparse matrices)
        self.mother = _remap(self._orig_mother)
        self.father = _remap(self._orig_father)
        self.twin = _remap(df["twin"].values)
        self.sex = df["sex"].values.astype(np.int8)
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

    @cached_property
    def _A4(self):
        """4-hop parent reach (great²-grandparents): A³ @ A."""
        t0 = time.perf_counter()
        result = self._A3 @ self._A
        logger.debug("_A4 = A3 @ A computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    @cached_property
    def _A5(self):
        """5-hop parent reach (great³-grandparents): A⁴ @ A."""
        t0 = time.perf_counter()
        result = self._A4 @ self._A
        logger.debug("_A5 = A4 @ A computed in %.3fs (nnz=%d)", time.perf_counter() - t0, result.nnz)
        return result

    def _get_Ak(self, k: int) -> sp.spmatrix:
        """Return the k-hop parent-reach matrix (k=0 returns identity)."""
        if k == 0:
            return sp.eye(self.n, format="csr")
        if k == 1:
            return self._A
        return getattr(self, f"_A{k}")

    def _ensure_sibling_matrices(self) -> None:
        """Ensure _full_sib_matrix and _half_sib_matrix are computed."""
        if hasattr(self, "_full_sib_matrix"):
            return
        # Trigger sibling extraction which sets _full_sib_matrix
        self._sibling_pairs()

    def _build_half_sib_matrix(
        self,
        mat_hs: tuple[np.ndarray, np.ndarray],
        pat_hs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Build and cache _half_sib_matrix from extracted half-sib pairs."""
        hs1 = np.concatenate([mat_hs[0], pat_hs[0]])
        hs2 = np.concatenate([mat_hs[1], pat_hs[1]])
        if len(hs1) > 0:
            ones = np.ones(len(hs1), dtype=np.float64)
            H = sp.csr_matrix((ones, (hs1, hs2)), shape=(self.n, self.n))
            self._half_sib_matrix = H + H.T
        else:
            self._half_sib_matrix = sp.csr_matrix((self.n, self.n))

    # ------------------------------------------------------------------
    # Shared extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_pairs(a_i: np.ndarray, a_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Canonicalize (lo, hi) and deduplicate pair arrays via int64 keys."""
        if len(a_i) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        lo = np.minimum(a_i, a_j).astype(np.intp)
        hi = np.maximum(a_i, a_j).astype(np.intp)
        max_id = int(hi.max()) + 1
        keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
        _, unique_idx = np.unique(keys, return_index=True)
        return lo[unique_idx], hi[unique_idx]

    def _extract_from_sparse(
        self,
        M: sp.spmatrix,
        subtract: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract nonzero pairs from sparse matrix, dedup, and subtract closer pairs.

        Mutates *M* in place (zeroes diagonal). Callers should not reuse *M*.
        All subtract pairs are batched into a single ``np.isin`` call.
        """
        M.setdiag(0)
        M.eliminate_zeros()
        if M.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        a_i, a_j = M.nonzero()
        lo, hi = self._dedup_pairs(a_i, a_j)

        if subtract and len(lo) > 0:
            # Collect all subtract pairs into one key set, then filter once
            rm_lo_parts: list[np.ndarray] = []
            rm_hi_parts: list[np.ndarray] = []
            for rm_pair in subtract:
                if len(rm_pair[0]) > 0:
                    r1, r2 = rm_pair
                    rm_lo_parts.append(np.minimum(r1, r2))
                    rm_hi_parts.append(np.maximum(r1, r2))
            if rm_lo_parts:
                all_rm_lo = np.concatenate(rm_lo_parts)
                all_rm_hi = np.concatenate(rm_hi_parts)
                max_id = int(max(lo.max(), hi.max(), all_rm_lo.max(), all_rm_hi.max())) + 1
                rm_keys = all_rm_lo.astype(np.int64) * max_id + all_rm_hi.astype(np.int64)
                cand_keys = lo.astype(np.int64) * max_id + hi.astype(np.int64)
                keep = ~np.isin(cand_keys, rm_keys)
                lo, hi = lo[keep], hi[keep]
        return lo, hi

    def _lineal_pairs(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Direct ancestor-descendant pairs at exactly k hops."""
        Ak = self._get_Ak(k)
        desc_i, anc_j = Ak.nonzero()
        if len(desc_i) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        return desc_i.astype(np.intp), anc_j.astype(np.intp)

    def _collateral_pairs(
        self,
        sib_matrix: sp.spmatrix,
        up: int,
        down: int,
        subtract: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pairs connected through a sibling link at depths (up, down).

        Individual B is (down-1) hops below a sibling of someone (up-1)
        hops above individual A, where sibling type is determined by
        *sib_matrix* (full-sib or half-sib).
        """
        if sib_matrix.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        A_down_1 = self._get_Ak(down - 1)
        A_up_1 = self._get_Ak(up - 1)
        M = A_down_1 @ sib_matrix @ A_up_1.T
        return self._extract_from_sparse(M, subtract=subtract)

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
            self._half_sib_matrix = sp.csr_matrix((self.n, self.n))
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
            # int64 cast required: max_id² overflows int32
            family_key = bk_mother.astype(np.int64) * max_parent + bk_father.astype(np.int64)
            full_sib = self._pairs_from_groups(bk_idx, family_key)
        else:
            full_sib = empty

        # --- Maternal half sibs: all pairs sharing known mother, minus full-sib pairs ---
        has_mother = nt_mother >= 0
        m_idx = nt_idx[has_mother]
        m_mother = nt_mother[has_mother]
        if len(m_idx) >= 2:
            mat_all = self._pairs_from_groups(m_idx, m_mother)
            mat_hs = self._subtract_pairs(mat_all, full_sib)
        else:
            mat_hs = empty

        # --- Paternal half sibs: all pairs sharing known father, minus full-sib pairs ---
        has_father = nt_father >= 0
        f_idx = nt_idx[has_father]
        f_father = nt_father[has_father]
        if len(f_idx) >= 2:
            pat_all = self._pairs_from_groups(f_idx, f_father)
            pat_hs = self._subtract_pairs(pat_all, full_sib)
        else:
            pat_hs = empty

        # Build full-sib sparse matrix for _avuncular_pairs and collateral methods
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

        # int64 cast required: max_id² overflows int32
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
        p1, p2 = self._pairs_from_groups(gc_i.astype(np.intp), gp_j)
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

        result = self._dedup_pairs(p1, p2)
        logger.debug("Cousins: %d unique pairs (%.3fs)", len(result[0]), time.perf_counter() - t0)
        return result

    def _grandparent_grandchild_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Grandparent-grandchild pairs: 2-hop ancestor links."""
        return self._lineal_pairs(2)

    def _avuncular_pairs(self, full_sib: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Avuncular (uncle/aunt-nephew/niece) pairs.

        An avuncular pair (child C, uncle U) exists when C's parent P is a
        full sibling of U. In matrix form: A @ S_full, then exclude
        parent-child pairs (which share the same edge structure).
        """
        self._ensure_sibling_matrices()
        if self._full_sib_matrix.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        avunc = self._A @ self._full_sib_matrix
        avunc.setdiag(0)

        # Exclude parent-child pairs
        parent_child = (self._A + self._A.T) > 0
        avunc = avunc - avunc.multiply(parent_child)
        avunc.eliminate_zeros()

        if avunc.nnz == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        return self._dedup_pairs(*avunc.nonzero())

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
        max_degree: int = 2,
        min_kinship: float = 0.0,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Extract all relationship categories.

        When ``sample_mask`` was provided at construction, only pairs where
        both individuals are active are returned.

        Args:
            max_degree: Maximum kinship degree to extract (1-5). Degree 2
                covers through 1st cousins, degree 5 through 2nd cousins.
                Higher degrees require more expensive matrix products.
            min_kinship: Skip pair types with kinship coefficient below this
                threshold. E.g., 0.125 skips 1st cousins (0.0625) and 2nd
                cousins (0.016), avoiding their expensive sparse products.

        Returns:
            Dict mapping relationship code to (idx1, idx2) row-index arrays.
        """
        t_total = time.perf_counter()
        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        empty = np.array([], dtype=np.intp)

        def _needed(code: str) -> bool:
            return PAIR_KINSHIP.get(code, 0) >= min_kinship

        need_hs = _needed("MHS") and max_degree >= 2

        # Pre-trigger cached properties needed by downstream extractions.
        # _Am/_Af are only needed to build _A; delete after to free memory.
        if max_degree >= 2:
            _ = self._A2  # chains: _Am, _Af → _A → _A2
        else:
            _ = self._A
        del self._Am, self._Af

        pairs["MZ"] = self._mz_twin_pairs()

        mo, fo = self._parent_offspring_pairs()
        pairs["MO"] = mo
        pairs["FO"] = fo

        t0 = time.perf_counter()
        full_sib, mat_hs, pat_hs = self._sibling_pairs()
        pairs["FS"] = full_sib
        pairs["MHS"] = mat_hs if need_hs else (empty, empty)
        pairs["PHS"] = pat_hs if need_hs else (empty, empty)
        logger.info(
            "Siblings: %d full, %d maternal HS, %d paternal HS (%.3fs)",
            len(pairs["FS"][0]),
            len(pairs["MHS"][0]),
            len(pairs["PHS"][0]),
            time.perf_counter() - t0,
        )

        # ---- Degree 2 (kinship 1/8): GP, Av, 1C ----
        if max_degree >= 2:
            t0 = time.perf_counter()
            need_cousins = _needed("1C")
            need_gp = _needed("GP")
            need_avunc = _needed("Av")

            futures = {}
            with ThreadPoolExecutor(max_workers=3) as pool:
                if need_cousins:
                    futures["1C"] = pool.submit(self._cousin_pairs)
                if need_gp:
                    futures["GP"] = pool.submit(self._grandparent_grandchild_pairs)
                if need_avunc:
                    futures["Av"] = pool.submit(self._avuncular_pairs, full_sib)
                for k, fut in futures.items():
                    pairs[k] = fut.result()

            for k in ("1C", "GP", "Av"):
                if k not in pairs:
                    pairs[k] = (empty, empty)

            logger.info(
                "Degree 2: cousins=%d, grandparent=%d, avuncular=%d (%.3fs)%s",
                len(pairs["1C"][0]),
                len(pairs["GP"][0]),
                len(pairs["Av"][0]),
                time.perf_counter() - t0,
                f" [min_kinship={min_kinship}]" if min_kinship > 0 else "",
            )
        else:
            for k in ("1C", "GP", "Av"):
                pairs[k] = (empty, empty)

        # ---- Degree 3+ setup (deferred to avoid work at default degree 2) ----
        if max_degree >= 3:
            po_pairs = (
                np.concatenate([pairs["MO"][0], pairs["FO"][0]]),
                np.concatenate([pairs["MO"][1], pairs["FO"][1]]),
            )
            gp_pairs = pairs["GP"]
            fsm = self._full_sib_matrix
            self._build_half_sib_matrix(mat_hs, pat_hs)
            hsm = self._half_sib_matrix
        # sib_all only needed at degree 4+ (1C1R, H1C1R, 1C2R subtract lists)
        if max_degree >= 4:
            sib_all = (
                np.concatenate([pairs["FS"][0], pairs["MHS"][0], pairs["PHS"][0]]),
                np.concatenate([pairs["FS"][1], pairs["MHS"][1], pairs["PHS"][1]]),
            )

        # ---- Degree 3 (kinship 1/16): GGP, HAv, GAv ----
        if max_degree >= 3:
            t0 = time.perf_counter()
            _ = self._A3  # pre-trigger

            futures: dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=3) as pool:
                if _needed("GGP"):
                    futures["GGP"] = pool.submit(self._lineal_pairs, 3)
                if _needed("HAv"):
                    futures["HAv"] = pool.submit(self._collateral_pairs, hsm, 1, 2, [po_pairs, gp_pairs])
                if _needed("GAv"):
                    futures["GAv"] = pool.submit(self._collateral_pairs, fsm, 1, 3, [po_pairs, gp_pairs, pairs["Av"]])
                for k, fut in futures.items():
                    pairs[k] = fut.result()
            for code in ("GGP", "HAv", "GAv"):
                if code not in pairs:
                    pairs[code] = (empty, empty)

            logger.info(
                "Degree 3: GGP=%d, HAv=%d, GAv=%d (%.3fs)",
                len(pairs["GGP"][0]),
                len(pairs["HAv"][0]),
                len(pairs["GAv"][0]),
                time.perf_counter() - t0,
            )
        else:
            for code in ("GGP", "HAv", "GAv"):
                pairs[code] = (empty, empty)

        # ---- Degree 4 (kinship 1/32): GGGP, HGAv, GGAv, H1C, 1C1R ----
        if max_degree >= 4:
            t0 = time.perf_counter()
            # Lazy: _A4 and A2_A3T triggered by types that need them
            A2_A3T = None
            if _needed("1C1R"):
                A2_A3T = self._A2 @ self._A3.T

            def _extract_h1c() -> tuple[np.ndarray, np.ndarray]:
                A2s = self._A2_shared
                half_gp = A2s.copy()
                half_gp.data[half_gp.data != 1] = 0
                half_gp.eliminate_zeros()
                half_gp.setdiag(0)
                share_parent = (self._S > 0).astype(np.float64)
                half_gp = half_gp - half_gp.multiply(share_parent)
                half_gp.eliminate_zeros()
                upper = sp.triu(half_gp, k=1)
                i, j = upper.nonzero()
                return (i.astype(np.intp), j.astype(np.intp)) if len(i) > 0 else (empty, empty)

            def _extract_1c1r() -> tuple[np.ndarray, np.ndarray]:
                P_full = A2_A3T.copy()
                P_full.setdiag(0)
                P_full.data[P_full.data < 2] = 0
                P_full.eliminate_zeros()
                return self._extract_from_sparse(
                    P_full,
                    subtract=[po_pairs, gp_pairs, pairs["GGP"], pairs["Av"], pairs["GAv"], sib_all, pairs["1C"]],
                )

            futures = {}
            with ThreadPoolExecutor(max_workers=5) as pool:
                if _needed("GGGP"):
                    futures["GGGP"] = pool.submit(self._lineal_pairs, 4)
                if _needed("HGAv"):
                    futures["HGAv"] = pool.submit(
                        self._collateral_pairs,
                        hsm,
                        1,
                        3,
                        [po_pairs, gp_pairs, pairs["GGP"], pairs["HAv"]],
                    )
                if _needed("GGAv"):
                    futures["GGAv"] = pool.submit(
                        self._collateral_pairs,
                        fsm,
                        1,
                        4,
                        [po_pairs, gp_pairs, pairs["GGP"], pairs["Av"], pairs["GAv"]],
                    )
                if _needed("H1C"):
                    futures["H1C"] = pool.submit(_extract_h1c)
                if _needed("1C1R"):
                    futures["1C1R"] = pool.submit(_extract_1c1r)
                for k, fut in futures.items():
                    pairs[k] = fut.result()
            for code in ("GGGP", "HGAv", "GGAv", "H1C", "1C1R"):
                if code not in pairs:
                    pairs[code] = (empty, empty)

            # _S only used for H1C; free before degree 5
            self.__dict__.pop("_S", None)

            logger.info(
                "Degree 4: GGGP=%d, HGAv=%d, GGAv=%d, H1C=%d, 1C1R=%d (%.3fs)",
                len(pairs["GGGP"][0]),
                len(pairs["HGAv"][0]),
                len(pairs["GGAv"][0]),
                len(pairs["H1C"][0]),
                len(pairs["1C1R"][0]),
                time.perf_counter() - t0,
            )
        else:
            A2_A3T = None
            for code in ("GGGP", "HGAv", "GGAv", "H1C", "1C1R"):
                pairs[code] = (empty, empty)

        # ---- Degree 5 (kinship 1/64): 2C, G3GP, HGGAv, G3Av, H1C1R, 1C2R ----
        if max_degree >= 5:
            t0 = time.perf_counter()
            _ = self._A5  # pre-trigger
            if A2_A3T is None:
                A2_A3T = self._A2 @ self._A3.T

            def _extract_h1c1r() -> tuple[np.ndarray, np.ndarray]:
                P_half = A2_A3T.copy()
                P_half.setdiag(0)
                P_half.data[P_half.data != 1] = 0
                P_half.eliminate_zeros()
                return self._extract_from_sparse(
                    P_half,
                    subtract=[
                        po_pairs,
                        gp_pairs,
                        pairs["GGP"],
                        pairs["GGGP"],
                        pairs["HAv"],
                        pairs["HGAv"],
                        sib_all,
                        pairs["1C"],
                        pairs["H1C"],
                        pairs["1C1R"],
                    ],
                )

            def _extract_1c2r() -> tuple[np.ndarray, np.ndarray]:
                P_full = self._A2 @ self._A4.T
                P_full.setdiag(0)
                P_full.data[P_full.data < 2] = 0
                P_full.eliminate_zeros()
                return self._extract_from_sparse(
                    P_full,
                    subtract=[
                        po_pairs,
                        gp_pairs,
                        pairs["GGP"],
                        pairs["GGGP"],
                        pairs["Av"],
                        pairs["GAv"],
                        pairs["GGAv"],
                        sib_all,
                        pairs["1C"],
                        pairs["H1C"],
                        pairs["1C1R"],
                    ],
                )

            futures = {}
            with ThreadPoolExecutor(max_workers=6) as pool:
                if _needed("2C"):
                    futures["2C"] = pool.submit(self._second_cousin_pairs)
                if _needed("G3GP"):
                    futures["G3GP"] = pool.submit(self._lineal_pairs, 5)
                if _needed("HGGAv"):
                    futures["HGGAv"] = pool.submit(
                        self._collateral_pairs,
                        hsm,
                        1,
                        4,
                        [po_pairs, gp_pairs, pairs["GGP"], pairs["GGGP"], pairs["HAv"], pairs["HGAv"]],
                    )
                if _needed("G3Av"):
                    futures["G3Av"] = pool.submit(
                        self._collateral_pairs,
                        fsm,
                        1,
                        5,
                        [po_pairs, gp_pairs, pairs["GGP"], pairs["GGGP"], pairs["Av"], pairs["GAv"], pairs["GGAv"]],
                    )
                if _needed("H1C1R"):
                    futures["H1C1R"] = pool.submit(_extract_h1c1r)
                if _needed("1C2R"):
                    futures["1C2R"] = pool.submit(_extract_1c2r)
                for k, fut in futures.items():
                    pairs[k] = fut.result()
            for code in ("2C", "G3GP", "HGGAv", "G3Av", "H1C1R", "1C2R"):
                if code not in pairs:
                    pairs[code] = (empty, empty)

            logger.info(
                "Degree 5: 2C=%d, G3GP=%d, HGGAv=%d, G3Av=%d, H1C1R=%d, 1C2R=%d (%.3fs)",
                len(pairs["2C"][0]),
                len(pairs["G3GP"][0]),
                len(pairs["HGGAv"][0]),
                len(pairs["G3Av"][0]),
                len(pairs["H1C1R"][0]),
                len(pairs["1C2R"][0]),
                time.perf_counter() - t0,
            )
        else:
            for code in ("2C", "G3GP", "HGGAv", "G3Av", "H1C1R", "1C2R"):
                pairs[code] = (empty, empty)

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

        # Free all cached sparse matrices — only pair arrays and _raw_pair_counts
        # are needed after this point.
        for attr in ("_A", "_A2", "_A3", "_A4", "_A5", "_S", "_A2_shared", "_full_sib_matrix", "_half_sib_matrix"):
            self.__dict__.pop(attr, None)

        logger.info("extract_pairs total: %.3fs", time.perf_counter() - t_total)
        return pairs

    def count_pairs(self, max_degree: int = 2) -> dict[str, int]:
        """Count all relationship categories.

        If ``extract_pairs()`` was already called on this instance, returns
        the cached pre-filter counts (nearly free).  Otherwise runs
        ``extract_pairs()`` to compute all types up to *max_degree*.
        """
        if hasattr(self, "_raw_pair_counts"):
            return dict(self._raw_pair_counts)
        self.extract_pairs(max_degree=max_degree)
        return dict(self._raw_pair_counts)

    # ------------------------------------------------------------------
    # Inbreeding and exact kinship
    # ------------------------------------------------------------------

    def compute_inbreeding(self) -> np.ndarray:
        """Compute inbreeding coefficient *F* for each individual.

        Uses sparse matrix algebra, propagating kinship generation by
        generation: ``K[gen_g, :] = P_g @ K`` where ``P`` is the
        parent-averaging matrix (``0.5 × _A``).  ``F_i`` is read from
        ``K[mother_i, father_i]`` each generation.

        For non-consanguineous pedigrees (the ACE default), all values
        are zero and the function returns quickly.

        Returns:
            Float64 array of length *n* with per-individual *F*.
        """
        t0 = time.perf_counter()
        n = self.n
        F = np.zeros(n, dtype=np.float64)
        mother = self.mother
        father = self.father
        generation = self.generation

        unique_gens = np.unique(generation)
        if len(unique_gens) <= 1:
            self._inbreeding = F
            return F

        # Parent-averaging matrix: P[child, parent] = 0.5
        m_mask = mother >= 0
        f_mask = father >= 0
        m_idx = np.where(m_mask)[0]
        f_idx = np.where(f_mask)[0]
        P = sp.csr_matrix(
            (
                np.full(len(m_idx) + len(f_idx), 0.5),
                (np.concatenate([m_idx, f_idx]), np.concatenate([mother[m_idx], father[f_idx]])),
            ),
            shape=(n, n),
        )

        # Build kinship matrix K generation by generation.
        # K is symmetric; we maintain it as CSR and rebuild symmetry
        # each generation via sparse addition.
        K = sp.csr_matrix((n, n), dtype=np.float64)

        for g in unique_gens:
            idx_g = np.where(generation == g)[0]

            if g == unique_gens[0]:
                # Founders: self-kinship = 0.5
                K = sp.csr_matrix(
                    (np.full(len(idx_g), 0.5), (idx_g, idx_g)),
                    shape=(n, n),
                )
                continue

            # F_i = K[mother_i, father_i] for this generation
            m_g = mother[idx_g]
            f_g = father[idx_g]
            both_known = (m_g >= 0) & (f_g >= 0)
            if np.any(both_known):
                bk_idx = np.where(both_known)[0]
                # Extract specific elements from sparse K
                K_csc = K.tocsc()
                for k in bk_idx:
                    F[idx_g[k]] = K_csc[m_g[k], f_g[k]]

            # Kinship of gen-g with all earlier: K_cross = P_g @ K
            P_g = P[idx_g, :]  # (n_g × n) sparse
            K_cross = P_g @ K  # (n_g × n) sparse — kinship with earlier gens

            # Kinship among gen-g individuals: K_within = P_g @ K @ P_g.T
            # Since K only has earlier-gen entries, this correctly uses
            # only parent-generation kinship (no circularity).
            K_within = K_cross @ P_g.T  # (n_g × n_g) sparse

            # Build update: cross-generation rows
            cross_rows, cross_cols, cross_vals = sp.find(K_cross)
            global_cross_rows = idx_g[cross_rows]

            # Build update: within-generation (upper triangle only to avoid double-counting)
            within_upper = sp.triu(K_within, k=1)
            w_rows, w_cols, w_vals = sp.find(within_upper)
            global_w_rows = idx_g[w_rows]
            global_w_cols = idx_g[w_cols]

            # Assemble all new entries
            all_rows = np.concatenate([global_cross_rows, global_w_rows])
            all_cols = np.concatenate([cross_cols, global_w_cols])
            all_vals = np.concatenate([cross_vals, w_vals])
            K_update = sp.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n, n))

            # Add new entries + transpose for symmetry
            K = K + K_update + K_update.T

            # Set self-kinship for gen_g: K[i,i] = (1 + F_i) / 2
            diag_vals = (1.0 + F[idx_g]) / 2.0
            K_diag_fix = sp.csr_matrix(
                (diag_vals - K.diagonal()[idx_g], (idx_g, idx_g)),
                shape=(n, n),
            )
            K = K + K_diag_fix

            logger.debug(
                "Inbreeding gen %d: %d individuals, K nnz=%d, F>0: %d",
                g,
                len(idx_g),
                K.nnz,
                np.count_nonzero(F[idx_g] > 0),
            )

        self._inbreeding = F
        self._kinship_matrix = K
        logger.info(
            "compute_inbreeding: %.1fs, K nnz=%d, n_inbred=%d, max_F=%.4f",
            time.perf_counter() - t0,
            K.nnz,
            np.count_nonzero(F > 0),
            F.max(),
        )
        return F

    def compute_pair_kinship(
        self,
        pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Compute exact kinship for each extracted pair.

        When no inbreeding is detected (all ``F = 0``), returns nominal
        kinship from ``PAIR_KINSHIP`` for each pair — a fast path.
        Otherwise, looks up exact kinship from the sparse kinship matrix
        built by ``compute_inbreeding()``.

        Call *after* ``extract_pairs()``.

        Returns:
            Dict mapping pair-type code to float64 array of kinship values,
            parallel to the index arrays in *pairs*.
        """
        F = self._inbreeding if hasattr(self, "_inbreeding") else self.compute_inbreeding()

        # Fast path: no inbreeding → nominal kinship is exact
        if np.all(F == 0):
            return {code: np.full(len(idx1), PAIR_KINSHIP.get(code, 0.0)) for code, (idx1, _) in pairs.items()}

        # Use the sparse kinship matrix from compute_inbreeding
        K = self._kinship_matrix.tocsr()

        result: dict[str, np.ndarray] = {}
        for code, (idx1, idx2) in pairs.items():
            if len(idx1) == 0:
                result[code] = np.array([], dtype=np.float64)
                continue
            # Vectorized sparse element extraction via array indexing
            kin = np.array(K[idx1, idx2]).ravel()
            result[code] = kin

        return result


def _build_graph_and_remap(
    df: pd.DataFrame,
    full_pedigree: pd.DataFrame,
    max_degree: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int], np.ndarray]:
    """Build PedigreeGraph on full pedigree, extract pairs, remap to df indices.

    Returns (remapped_pairs, full_pedigree_counts, full_ids).
    """
    pheno_ids = set(df["id"].values.tolist())
    full_ids = full_pedigree["id"].values
    sample_mask = np.array([int(x) in pheno_ids for x in full_ids], dtype=bool)

    pg = PedigreeGraph(full_pedigree, sample_mask=sample_mask)
    raw_pairs = pg.extract_pairs(max_degree=max_degree)
    full_counts = dict(pg._raw_pair_counts)

    # Remap full-pedigree row indices → phenotype (df) row indices
    pheno_id_arr = df["id"].values
    max_id = int(max(full_ids.max(), pheno_id_arr.max())) + 1
    id_to_pheno = np.full(max_id, -1, dtype=np.int32)
    id_to_pheno[pheno_id_arr] = np.arange(len(df), dtype=np.int32)

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
    full_pedigree: pd.DataFrame | None = None,
    max_degree: int = 2,
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
            max_degree,
        )
        return pairs, full_counts

    pg = PedigreeGraph(df)
    pairs = pg.extract_pairs(max_degree=max_degree)
    return pairs, None


def extract_relationship_pairs(
    df: pd.DataFrame,
    full_pedigree: pd.DataFrame | None = None,
    max_degree: int = 2,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract relationship pairs, returning row indices into *df*.

    When *full_pedigree* is provided the graph is built from the complete
    pedigree (all G_ped generations) so that multi-hop relationships
    (grandparent, avuncular, cousin) are detected through ancestors not
    present in *df*.  Output pairs are filtered to individuals in *df*
    and remapped to *df* row indices.

    Returns dict keyed by relationship code (e.g. "MZ", "FS", "1C").
    """
    pairs, _ = extract_and_count_relationship_pairs(
        df,
        full_pedigree=full_pedigree,
        max_degree=max_degree,
    )
    return pairs


def count_relationship_pairs(
    df: pd.DataFrame,
    max_degree: int = 2,
) -> dict[str, int]:
    """Count relationship pairs without materializing full index arrays.

    Memory-efficient alternative to extract_relationship_pairs() when only
    pair counts are needed (e.g. for the full pedigree summary).
    """
    pg = PedigreeGraph(df)
    return pg.count_pairs(max_degree=max_degree)


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
        "FS": full_sib,
        "MHS": mat_hs,
        "PHS": pat_hs,
    }


def count_sib_pairs(df: pd.DataFrame) -> dict[str, int]:
    """Drop-in replacement for validate._count_sib_pairs.

    Accepts a DataFrame of non-twin non-founders with columns id, mother, father.
    """
    # Build a minimal full DataFrame for PedigreeGraph
    # The input is a subset; we need to create a graph over the full ID space
    # Instead, replicate the counting logic directly without building a full graph
    _twin_col = df["twin"].values if "twin" in df.columns else np.full(len(df), -1, dtype=np.int32)
    mother_col = df["mother"].values
    father_col = df["father"].values

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
