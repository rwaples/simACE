"""Top-X principal components of a precomputed sparse GRM.

Computes matrix-free top-k eigendecomposition of a sparse 2φ (or any
sparse PSD relationship matrix) via ARPACK
(``scipy.sparse.linalg.eigsh``).  The input matrix is consumed as-is —
kinship construction and on-disk export live in
``fit_ace.kinship.kinship`` and ``fit_ace.kinship.export``.

Scores are returned in ``U·√Λ`` convention so each PC column carries its
own variance scale.  Column signs are fixed by a deterministic
convention (largest |entry| positive) to make output bit-reproducible
given the same seed.

Downstream note: without population structure the leading eigenvector
is close to the all-ones direction (a grand-mean-relatedness axis); PC1
therefore looks like a connectivity score and downstream consumers may
want to drop it.
"""

from __future__ import annotations

__all__ = ["compute_kinship_pcs"]

import logging
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


def compute_kinship_pcs(
    matrix: sp.spmatrix,
    ids: np.ndarray,
    n_pcs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute the top ``n_pcs`` principal components of a sparse PSD matrix.

    Parameters
    ----------
    matrix : square sparse matrix (typically 2φ, the expected A-matrix).
        Any scipy sparse format is accepted; internally converted to CSC
        for the ARPACK matvecs.
    ids : (N,) array of individual IDs in the same row/column order as
        *matrix*.  Returned verbatim (cast to int32 when possible) for
        downstream alignment.
    n_pcs : requested number of PCs.  If the matrix is too small
        (``N < n_pcs + 2``), only ``k = min(n_pcs, N-2)`` are computed;
        remaining columns are NaN-padded so the output schema is stable.
    seed : int used to derive a reproducible ARPACK start vector ``v0``.

    Returns:
    -------
    ids : (N,) int32 when *ids* is numeric, otherwise passed through.
    pcs : (N, n_pcs) float32 — PC scores ``U · √Λ``; trailing columns
        are NaN when ``k < n_pcs``.
    eigenvalues : (n_pcs,) float64 — eigenvalues in descending order;
        trailing entries are NaN when ``k < n_pcs``.
    trace : float — ``Σ diag(matrix)``, the natural denominator for
        variance-explained.
    """
    if n_pcs <= 0:
        raise ValueError(f"n_pcs must be >= 1, got {n_pcs}")
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    ids = np.asarray(ids)
    if ids.shape[0] != n:
        raise ValueError(f"len(ids)={ids.shape[0]} does not match matrix.shape={matrix.shape}")

    # Best-effort int32 cast for integer ids (matches pedigree.id dtype)
    if np.issubdtype(ids.dtype, np.integer):
        ids = ids.astype(np.int32, copy=False)

    t0 = time.perf_counter()
    kmat = matrix.tocsc()
    trace = float(kmat.diagonal().sum())

    pcs_out = np.full((n, n_pcs), np.nan, dtype=np.float32)
    eigvals_out = np.full(n_pcs, np.nan, dtype=np.float64)

    k = min(n_pcs, n - 2)
    if k < 1:
        logger.warning(
            "compute_kinship_pcs: N=%d too small for any PCs (need N >= 3); returning all-NaN output.",
            n,
        )
        return ids, pcs_out, eigvals_out, trace
    if k < n_pcs:
        logger.warning(
            "compute_kinship_pcs: requested n_pcs=%d but N=%d; computing k=%d PCs and NaN-padding.",
            n_pcs,
            n,
            k,
        )

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(np.float64)
    v0 /= np.linalg.norm(v0)

    t_eig = time.perf_counter()
    # tol=1e-6 is ~1.7x faster than tol=0 (machine precision) at N=50K-300K
    # with subspace error ~1e-6 — indistinguishable from exact for downstream
    # use as covariates, scree plots, or structure scatters.
    eigvals, eigvecs = spla.eigsh(
        kmat.astype(np.float64),
        k=k,
        which="LA",
        v0=v0,
        tol=1e-6,
        maxiter=max(1000, 20 * k),
    )

    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Deterministic sign: flip each column so the entry of largest |value|
    # is positive, eliminating ARPACK's arbitrary sign for bit-reproducible
    # output.
    for j in range(k):
        col = eigvecs[:, j]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            eigvecs[:, j] = -col

    # U · √Λ scoring convention (clip λ at 0 to guard small negative values
    # from numerical noise on the rank-deficient tail).
    scale = np.sqrt(np.clip(eigvals, 0.0, None))
    scores = (eigvecs * scale[None, :]).astype(np.float32)

    pcs_out[:, :k] = scores
    eigvals_out[:k] = eigvals

    logger.info(
        "compute_kinship_pcs: N=%d, k=%d, trace=%.3f, eig=%.2fs, total=%.2fs",
        n,
        k,
        trace,
        time.perf_counter() - t_eig,
        time.perf_counter() - t0,
    )

    return ids, pcs_out, eigvals_out, trace
