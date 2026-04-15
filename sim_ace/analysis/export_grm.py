r"""GRM and phenotype exporters for sparseREML and MPH.

Writes the on-disk formats that the two external REML tools consume.

**sparseREML** (R, at ``external/REML-with-sparse-relationship-matrices/``)
reads sparse triplet GRMs plus PLINK-style tab-separated phenotype/covariate
files with no header:

  ``<prefix>.grm.sp`` — ``i\tj\tvalue`` triplets, 0-based indices,
    diagonal + one triangle only
  ``<prefix>.grm.id`` — ``FID\tIID`` per line; reader takes column 2 as IID
  pheno file — ``FID\tIID\tvalue`` per line
  covar file — ``FID\tIID\tc1\tc2...`` per line

**MPH** (C++, at ``external/mph/``) reads dense binary GRMs in GCTA's
classic layout plus CSV phenotype/covariate files with headers:

  ``<prefix>.grm.bin`` — ``int32 n`` + ``float32 sum2pq`` + lower-triangle
    float32s stored column-by-column (column *i* has ``n-i`` entries)
  ``<prefix>.grm.iid`` — one IID per line, no header
  pheno CSV — comma-separated with header; first column is individual id
  covar CSV — comma-separated with header; first column is individual id

MPH's dense binary storage is ~``2·n²`` bytes — practical up to perhaps
N≈10k on a workstation, impossible at N=100k. The sparse exporter is the
one to use at scale.
"""

from __future__ import annotations

__all__ = [
    "ACE_SREML_MAGIC",
    "build_household_matrix",
    "collapse_mz_twins",
    "export_dense_grm_mph",
    "export_household_grm",
    "export_pheno_csv",
    "export_pheno_plink",
    "export_sparse_grm_binary",
    "export_sparse_grm_gcta",
    "require_cols",
]

import logging
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# Binary sparse GRM: 8-byte magic + int64 n + int64 nnz + CSC arrays.
# Row indices + column pointers are int64 so ace_sreml (CHOLMOD_LONG backend)
# can memory-map arrays directly without conversion.  Full-symmetric storage
# (both triangles) lets the reader skip triplet construction entirely.
ACE_SREML_MAGIC = b"ACEGRM\x01\x00"


# ---------------------------------------------------------------------------
# Kinship / GRM helpers
# ---------------------------------------------------------------------------


def _validate_grm_shape(K: sp.spmatrix, iids: np.ndarray, threshold: float) -> tuple[int, np.ndarray]:
    """Shape and dtype checks shared by every GRM exporter."""
    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError(f"K must be square, got {K.shape}")
    iids = np.asarray(iids)
    if iids.shape[0] != n:
        raise ValueError(f"len(iids)={iids.shape[0]} does not match K.shape={K.shape}")
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    return n, iids


def _prepare_sparse_grm(
    K: sp.spmatrix,
    to_grm: bool,
    threshold: float,
    *,
    symmetric_storage: bool,
    log_name: str,
) -> sp.spmatrix:
    """Common pre-processing for the sparse GRM writers.

    Multiplies by 2 (kinship → GRM) if requested, optionally symmetrizes to
    full storage (binary writer wants both triangles; text writer wants
    lower only), and optionally drops off-diagonal entries below a GRM-scale
    threshold.
    """
    M = (K * 2.0) if to_grm else K
    M_csc = sp.csc_matrix(M)

    if symmetric_storage and (M_csc - M_csc.T).nnz != 0:
        # Input stored only one triangle.  Symmetrize and halve the diagonal
        # that the addition double-counted.
        M_csc = (M_csc + M_csc.T).tocsc()
        M_csc.setdiag(M_csc.diagonal() / 2)
        M_csc.eliminate_zeros()
    elif not symmetric_storage:
        M_csc = sp.tril(M_csc).tocsc()

    if threshold > 0.0:
        nnz_before = M_csc.nnz
        coo = M_csc.tocoo()
        keep = (coo.row == coo.col) | (np.abs(coo.data) >= threshold)
        M_csc = sp.coo_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=M_csc.shape,
        ).tocsc()
        logger.info(
            "%s: threshold=%.4g kept %d of %d nonzeros (%.1f%%)",
            log_name,
            threshold,
            M_csc.nnz,
            nnz_before,
            100.0 * M_csc.nnz / max(nnz_before, 1),
        )
    return M_csc


# ---------------------------------------------------------------------------
# Kinship / GRM writers
# ---------------------------------------------------------------------------


def export_sparse_grm_binary(
    K: sp.spmatrix,
    iids: np.ndarray,
    prefix: str | Path,
    to_grm: bool = True,
    threshold: float = 0.0,
) -> tuple[Path, Path]:
    """Write a sparse kinship in ace_sreml's binary CSC format.

    The file layout is a fixed 24-byte header (magic + n + nnz) followed by
    CSC column pointers (``int64[n+1]``), row indices (``int64[nnz]``), and
    values (``double[nnz]``).  Storage is full-symmetric so the C++ reader
    can copy the three arrays straight into Eigen's sparse-matrix storage
    without materializing triplets.

    Typical load time is ~5% of the equivalent ``.grm.sp`` text parse.

    Args:
        K: sparse (n, n) matrix.  Symmetric or lower-only both OK; the writer
            produces a symmetric matrix on disk.
        iids: length-n individual ids (row/col order of *K*).
        prefix: output prefix; writes ``<prefix>.grm.sp.bin`` and
            ``<prefix>.grm.id``.
        to_grm: multiply *K* by 2 on write (kinship → GRM convention).
        threshold: drop off-diagonal entries with absolute value below this
            (after the to_grm rescale, so it's on GRM scale). Diagonal is
            kept. ``0.0`` (default) keeps everything; ``0.05`` matches
            sparseREML's default ``GRM_range[0]`` and drops kinship ≲
            2nd-cousin level — crucial at n ≥ 10⁵ to keep Cholesky fill
            tractable.

    Returns:
        (bin_path, id_path).
    """
    n, iids = _validate_grm_shape(K, iids, threshold)

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = Path(f"{prefix}.grm.sp.bin")
    id_path = Path(f"{prefix}.grm.id")

    M_csc = _prepare_sparse_grm(
        K,
        to_grm,
        threshold,
        symmetric_storage=True,
        log_name="export_sparse_grm_binary",
    )
    M_csc.sort_indices()
    indptr = M_csc.indptr.astype(np.int64, copy=False)
    indices = M_csc.indices.astype(np.int64, copy=False)
    data = M_csc.data.astype(np.float64, copy=False)
    nnz = int(M_csc.nnz)

    with bin_path.open("wb") as fh:
        fh.write(ACE_SREML_MAGIC)
        fh.write(struct.pack("<qq", int(n), nnz))
        fh.write(indptr.tobytes())
        fh.write(indices.tobytes())
        fh.write(data.tobytes())

    _write_id_file(iids, id_path)
    logger.info(
        "export_sparse_grm_binary: %s (%.1f MB, n=%d, nnz=%d), %s",
        bin_path.name,
        bin_path.stat().st_size / (1024**2),
        n,
        nnz,
        id_path.name,
    )
    return bin_path, id_path


def export_sparse_grm_gcta(
    K: sp.spmatrix,
    iids: np.ndarray,
    prefix: str | Path,
    to_grm: bool = True,
    threshold: float = 0.0,
) -> tuple[Path, Path]:
    """Write a sparse kinship in GCTA sparse triplet format.

    Args:
        K: sparse (n, n) matrix. If *to_grm* is True, *K* is interpreted as a
            kinship matrix (diag = 0.5·(1+F_i), off-diag = kinship coef) and
            written out as the corresponding GRM (A = 2·K). If False, the
            matrix is written as-is — use this for auxiliary GRMs like C.
        iids: length-n array of individual ids; row/col order of *K*.
        prefix: output prefix; files land at ``<prefix>.grm.sp`` and
            ``<prefix>.grm.id``.
        to_grm: multiply *K* by 2 before writing.
        threshold: drop off-diagonal entries with |value| below this (after
            the to_grm rescale, so on GRM scale).  ``0`` keeps everything.

    Returns:
        (sp_path, id_path).
    """
    n, iids = _validate_grm_shape(K, iids, threshold)

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sp_path = Path(f"{prefix}.grm.sp")
    id_path = Path(f"{prefix}.grm.id")

    lower = _prepare_sparse_grm(
        K,
        to_grm,
        threshold,
        symmetric_storage=False,
        log_name="export_sparse_grm_gcta",
    ).tocoo()
    mask = lower.data != 0.0
    rows = lower.row[mask]
    cols = lower.col[mask]
    vals = lower.data[mask]

    with sp_path.open("w") as fh:
        for i, j, v in zip(rows, cols, vals, strict=True):
            fh.write(f"{int(i)}\t{int(j)}\t{v:.10g}\n")

    _write_id_file(iids, id_path)
    logger.info(
        "export_sparse_grm_gcta: %s (nnz=%d, n=%d), %s",
        sp_path.name,
        len(vals),
        n,
        id_path.name,
    )
    return sp_path, id_path


def export_dense_grm_mph(
    K: sp.spmatrix | np.ndarray,
    iids: np.ndarray,
    prefix: str | Path,
    sum2pq: float = 1.0,
    to_grm: bool = True,
) -> tuple[Path, Path]:
    """Write a dense kinship/GRM in GCTA binary format that MPH reads.

    Layout of ``<prefix>.grm.bin``: ``int32 n``, ``float32 sum2pq``, then the
    lower triangle column-by-column as float32 (column *i* has ``n-i`` entries,
    starting with the diagonal).

    Memory: the file is ``8 + 4·n·(n+1)/2`` bytes. At n=10k this is 200 MB;
    at n=100k, 20 GB. Use the sparse exporter for large n.

    Args:
        K: (n, n) kinship or GRM matrix; sparse or dense.
        iids: length-n array of individual ids.
        prefix: output prefix; writes ``<prefix>.grm.bin`` and
            ``<prefix>.grm.iid``.
        sum2pq: header value MPH reads but doesn't use for our purposes
            (relevant only for SNP-based GRMs).
        to_grm: if True, multiply K by 2 on write (kinship → GRM).

    Returns:
        (bin_path, iid_path).
    """
    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError(f"K must be square, got {K.shape}")
    iids = np.asarray(iids)
    if iids.shape[0] != n:
        raise ValueError(f"len(iids)={iids.shape[0]} does not match K.shape={K.shape}")

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = Path(f"{prefix}.grm.bin")
    iid_path = Path(f"{prefix}.grm.iid")

    # Densify as float32 for the write (MPH expects float32).
    if sp.issparse(K):
        dense = np.asarray(K.todense(), dtype=np.float32)
    else:
        dense = np.asarray(K, dtype=np.float32)
    if to_grm:
        dense = dense * np.float32(2.0)

    with bin_path.open("wb") as fh:
        fh.write(struct.pack("<i", int(n)))
        fh.write(struct.pack("<f", float(sum2pq)))
        for i in range(n):
            fh.write(dense[i:, i].tobytes(order="C"))

    with iid_path.open("w") as fh:
        for x in iids:
            fh.write(f"{x}\n")

    logger.info(
        "export_dense_grm_mph: %s (%.1f MB, n=%d), %s",
        bin_path.name,
        bin_path.stat().st_size / (1024**2),
        n,
        iid_path.name,
    )
    return bin_path, iid_path


def export_household_grm(
    household_ids: np.ndarray,
    iids: np.ndarray,
    prefix: str | Path,
    dense_for_mph: bool = False,
) -> tuple[Path, Path]:
    """Write the common-environment (C) GRM: block-diagonal per household.

    C[i, j] = 1 if individuals i and j share a household, else 0 (diag = 1).
    Household ids < 0 are treated as singletons (no off-diagonal entries).

    Args:
        household_ids: length-n array mapping individual → household id.
        iids: length-n individual ids.
        prefix: output prefix.
        dense_for_mph: if True, writes GCTA binary via
            ``export_dense_grm_mph``; otherwise sparse triplet via
            ``export_sparse_grm_gcta`` (with ``to_grm=False`` because C is
            already on a GRM-like 0/1 scale).

    Returns:
        File paths written by the underlying exporter.
    """
    n = len(household_ids)
    if len(iids) != n:
        raise ValueError(f"len(iids)={len(iids)} does not match len(household_ids)={n}")

    C = build_household_matrix(np.asarray(household_ids))
    if dense_for_mph:
        return export_dense_grm_mph(C, iids, prefix, to_grm=False)
    return export_sparse_grm_gcta(C, iids, prefix, to_grm=False)


def build_household_matrix(household_ids: np.ndarray) -> sp.csr_matrix:
    """Return the (n, n) sparse C matrix for the given household ids.

    C[i, j] = 1 iff individuals i and j share a household; diag = 1.
    household_ids < 0 are treated as singletons.
    """
    n = len(household_ids)
    hh = household_ids.astype(np.int64, copy=False)

    diag_rows = np.arange(n, dtype=np.int64)
    row_parts: list[np.ndarray] = [diag_rows]
    col_parts: list[np.ndarray] = [diag_rows]
    val_parts: list[np.ndarray] = [np.ones(n, dtype=np.float64)]

    valid = hh >= 0
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        order = np.argsort(hh[valid_idx], kind="stable")
        sorted_idx = valid_idx[order]
        sorted_hh = hh[sorted_idx]
        cuts = np.flatnonzero(np.diff(sorted_hh)) + 1
        for group in np.split(sorted_idx, cuts):
            if len(group) < 2:
                continue
            group_sorted = np.sort(group)
            i_loc, j_loc = np.triu_indices(len(group_sorted), k=1)
            # Off-diagonal pairs for both triangles (symmetric matrix).
            r_upper = group_sorted[i_loc]
            c_upper = group_sorted[j_loc]
            row_parts.extend([r_upper, c_upper])
            col_parts.extend([c_upper, r_upper])
            ones = np.ones(len(i_loc), dtype=np.float64)
            val_parts.extend([ones, ones])

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    vals = np.concatenate(val_parts)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ---------------------------------------------------------------------------
# Phenotype / covariate writers
# ---------------------------------------------------------------------------


def export_pheno_plink(
    df: pd.DataFrame,
    out_path: str | Path,
    pheno_col: str = "y",
    covar_cols: tuple[str, ...] = (),
    id_col: str = "id",
) -> tuple[Path, Path]:
    """Write PLINK-style phenotype/covariate files for sparseREML.

    Both files are tab-separated with no header. Phenotype file:
    ``FID IID <pheno_col>``. Covariate file: ``FID IID <c1> <c2> ...``; an
    explicit intercept column is **not** added because sparseREML prepends
    its own.

    Args:
        df: DataFrame containing *id_col*, *pheno_col*, and all *covar_cols*.
        out_path: path stem; writes ``<out_path>.pheno.txt`` and
            ``<out_path>.covar.txt``. The covariate file is always produced
            — with a single dummy column of 1.0 when *covar_cols* is empty
            (sparseREML treats this as a redundant predictor alongside the
            intercept; harmless at the rank-deficient column level because
            sparseREML solves via pseudoinverse of X'V⁻¹X).
        pheno_col: column name of the phenotype.
        covar_cols: tuple of column names to write as covariates.
        id_col: column name used as both FID and IID.

    Returns:
        (pheno_path, covar_path).
    """
    require_cols(df, [id_col, pheno_col, *covar_cols])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = df[id_col].to_numpy()
    pheno_path = Path(f"{out_path}.pheno.txt")
    covar_path = Path(f"{out_path}.covar.txt")

    pheno_frame = pd.DataFrame({"fid": ids, "iid": ids, pheno_col: df[pheno_col].to_numpy()})
    pheno_frame.to_csv(pheno_path, sep="\t", header=False, index=False)

    if covar_cols:
        covar_frame = pd.DataFrame({"fid": ids, "iid": ids})
        for c in covar_cols:
            covar_frame[c] = df[c].to_numpy()
    else:
        covar_frame = pd.DataFrame({"fid": ids, "iid": ids, "intercept": np.ones(len(df), dtype=np.float64)})
    covar_frame.to_csv(covar_path, sep="\t", header=False, index=False)

    logger.info(
        "export_pheno_plink: %s, %s (n=%d, covars=%d)",
        pheno_path.name,
        covar_path.name,
        len(df),
        len(covar_cols) if covar_cols else 1,
    )
    return pheno_path, covar_path


def export_pheno_csv(
    df: pd.DataFrame,
    out_path: str | Path,
    pheno_col: str = "y",
    covar_cols: tuple[str, ...] = (),
    id_col: str = "id",
) -> tuple[Path, Path]:
    """Write comma-separated header-ful phenotype/covariate files for MPH.

    Phenotype CSV columns: ``id,<pheno_col>`` (additional traits can share
    the same file — call again with a different *pheno_col* to append).
    Covariate CSV columns: ``id,<c1>,<c2>,...``. If *covar_cols* is empty
    MPH uses an intercept only (and requires no covariate file), so we
    still write the file so the bench runner can pass it consistently.

    Args:
        df: DataFrame containing *id_col*, *pheno_col*, all *covar_cols*.
        out_path: path stem; writes ``<out_path>.pheno.csv`` and
            ``<out_path>.covar.csv``.
        pheno_col: column name of the phenotype.
        covar_cols: tuple of column names to write as covariates.
        id_col: column name used as the MPH individual id.

    Returns:
        (pheno_path, covar_path).
    """
    require_cols(df, [id_col, pheno_col, *covar_cols])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pheno_path = Path(f"{out_path}.pheno.csv")
    covar_path = Path(f"{out_path}.covar.csv")

    ids = df[id_col].to_numpy()
    pheno_frame = pd.DataFrame({"id": ids, pheno_col: df[pheno_col].to_numpy()})
    pheno_frame.to_csv(pheno_path, sep=",", header=True, index=False)

    covar_frame = pd.DataFrame({"id": ids})
    if covar_cols:
        for c in covar_cols:
            covar_frame[c] = df[c].to_numpy()
    else:
        covar_frame["intercept"] = np.ones(len(df), dtype=np.float64)
    covar_frame.to_csv(covar_path, sep=",", header=True, index=False)

    logger.info(
        "export_pheno_csv: %s, %s (n=%d, covars=%d)",
        pheno_path.name,
        covar_path.name,
        len(df),
        len(covar_cols) if covar_cols else 1,
    )
    return pheno_path, covar_path


# ---------------------------------------------------------------------------
# MZ twin collapse
# ---------------------------------------------------------------------------


def collapse_mz_twins(
    pedigree: pd.DataFrame,
    K: sp.spmatrix | None = None,
    twin_col: str = "twin",
    id_col: str = "id",
) -> tuple[pd.DataFrame, sp.csr_matrix | None]:
    """Drop the larger-id member of each MZ twin pair.

    The ACE simulator stores the *id* of the other twin in *twin_col* and
    ``-1`` for singletons (``simulate.py:961``). Pairs are symmetric, so
    keeping rows with ``twin == -1 or id < twin`` removes exactly one row
    per pair. This prevents the kinship submatrix from becoming singular
    because of identical A columns in MZ siblings.

    Args:
        pedigree: DataFrame with *id_col* and *twin_col*.
        K: optional sparse (n, n) kinship matrix aligned with *pedigree*
            order. If provided, rows and columns of dropped individuals are
            removed from *K* and the filtered matrix is returned.
        twin_col: column name holding the other twin's id (-1 = not a twin).
        id_col: column name for individual id.

    Returns:
        ``(filtered_df, filtered_K)`` — ``filtered_K`` is None when *K* is
        None. The DataFrame row index is reset.
    """
    require_cols(pedigree, [id_col, twin_col])
    ids = pedigree[id_col].to_numpy()
    twins = pedigree[twin_col].to_numpy()
    if K is not None and K.shape[0] != len(pedigree):
        raise ValueError(f"K.shape[0]={K.shape[0]} does not match len(pedigree)={len(pedigree)}")

    keep = (twins < 0) | (ids < twins)
    n_drop = int((~keep).sum())
    logger.info(
        "collapse_mz_twins: n=%d, dropping %d twin duplicates → n=%d",
        len(pedigree),
        n_drop,
        len(pedigree) - n_drop,
    )

    filt_df = pedigree.loc[keep].reset_index(drop=True)
    if K is None:
        return filt_df, None
    keep_idx = np.flatnonzero(keep)
    K_csr = K.tocsr()
    filt_K = K_csr[keep_idx, :][:, keep_idx]
    return filt_df, filt_K


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_id_file(iids: np.ndarray, path: Path) -> None:
    """Write GCTA ``.grm.id`` — two tab-separated columns (FID, IID)."""
    arr = np.asarray(iids).astype(str)
    with path.open("w") as fh:
        for x in arr:
            fh.write(f"{x}\t{x}\n")


def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if *df* is missing any of *cols*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
