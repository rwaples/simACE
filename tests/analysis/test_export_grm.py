"""Unit tests for sim_ace.analysis.export_grm."""

import struct

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sim_ace.analysis.export_grm import (
    ACE_SREML_MAGIC,
    collapse_mz_twins,
    export_dense_grm_mph,
    export_household_grm,
    export_pheno_csv,
    export_pheno_plink,
    export_sparse_grm_binary,
    export_sparse_grm_gcta,
)

# ---------------------------------------------------------------------------
# Sparse GRM writer
# ---------------------------------------------------------------------------


def _read_gcta_sp(sp_path, n):
    """Read a GCTA sparse-triplet file back into a symmetric dense matrix."""
    M = np.zeros((n, n), dtype=np.float64)
    with open(sp_path) as fh:
        for line in fh:
            i, j, v = line.strip().split("\t")
            i, j, v = int(i), int(j), float(v)
            M[i, j] = v
            if i != j:
                M[j, i] = v
    return M


def _read_gcta_id(id_path):
    ids = []
    with open(id_path) as fh:
        for line in fh:
            fid, iid = line.rstrip("\n").split("\t")
            assert fid == iid
            ids.append(iid)
    return ids


class TestExportSparseGrmGcta:
    def test_roundtrip_kinship_to_grm(self, tmp_path):
        n = 6
        K = np.array(
            [
                [0.50, 0.10, 0.00, 0.00, 0.00, 0.00],
                [0.10, 0.50, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.50, 0.25, 0.00, 0.00],
                [0.00, 0.00, 0.25, 0.50, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.55, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.50],
            ]
        )
        K_sp = sp.csr_matrix(K)
        iids = np.array([f"ind{i}" for i in range(n)])
        sp_path, id_path = export_sparse_grm_gcta(K_sp, iids, tmp_path / "K", to_grm=True)

        M_read = _read_gcta_sp(sp_path, n)
        np.testing.assert_allclose(M_read, 2.0 * K, rtol=1e-9)
        assert _read_gcta_id(id_path) == list(iids)

    def test_to_grm_false_passes_through(self, tmp_path):
        n = 4
        C = np.eye(n)
        C[0, 1] = C[1, 0] = 1.0
        sp_path, _ = export_sparse_grm_gcta(sp.csr_matrix(C), np.arange(n), tmp_path / "C", to_grm=False)
        M = _read_gcta_sp(sp_path, n)
        np.testing.assert_allclose(M, C)

    def test_drops_explicit_zeros(self, tmp_path):
        n = 3
        data = [1.0, 0.0, 1.0]
        row = [0, 0, 1]
        col = [0, 1, 1]
        K = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
        sp_path, _ = export_sparse_grm_gcta(K, np.arange(n), tmp_path / "K", to_grm=False)
        lines = sp_path.read_text().strip().splitlines()
        assert len(lines) == 2  # the explicit zero is dropped

    def test_rejects_non_square(self, tmp_path):
        with pytest.raises(ValueError, match="square"):
            export_sparse_grm_gcta(sp.csr_matrix(np.zeros((3, 4))), np.arange(3), tmp_path / "x")

    def test_rejects_iid_length_mismatch(self, tmp_path):
        K = sp.eye(5, format="csr")
        with pytest.raises(ValueError, match="does not match"):
            export_sparse_grm_gcta(K, np.arange(4), tmp_path / "x")


# ---------------------------------------------------------------------------
# Dense GRM writer for MPH
# ---------------------------------------------------------------------------


def _read_gcta_bin(bin_path, n):
    """Read GCTA binary GRM back into a symmetric dense matrix."""
    with open(bin_path, "rb") as fh:
        (num,) = struct.unpack("<i", fh.read(4))
        (sum2pq,) = struct.unpack("<f", fh.read(4))
        M = np.zeros((num, num), dtype=np.float32)
        for i in range(num):
            entries = np.frombuffer(fh.read((num - i) * 4), dtype=np.float32)
            M[i:, i] = entries
            M[i, i:] = entries  # mirror
    return M, sum2pq


class TestExportSparseGrmBinary:
    def _read_binary(self, path, n_expected):
        with open(path, "rb") as fh:
            magic = fh.read(8)
            assert magic == ACE_SREML_MAGIC
            n, nnz = struct.unpack("<qq", fh.read(16))
            assert n == n_expected
            indptr = np.frombuffer(fh.read((n + 1) * 8), dtype=np.int64)
            indices = np.frombuffer(fh.read(nnz * 8), dtype=np.int64)
            values = np.frombuffer(fh.read(nnz * 8), dtype=np.float64)
        return n, nnz, indptr, indices, values

    def test_roundtrip_symmetric(self, tmp_path):
        n = 5
        K = np.array(
            [
                [0.50, 0.10, 0.00, 0.00, 0.00],
                [0.10, 0.50, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.55, 0.25, 0.00],
                [0.00, 0.00, 0.25, 0.50, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.50],
            ],
            dtype=np.float64,
        )
        K_sp = sp.csc_matrix(K)
        iids = np.array([f"p{i}" for i in range(n)])
        bin_path, id_path = export_sparse_grm_binary(K_sp, iids, tmp_path / "K", to_grm=True)
        _, nnz, indptr, indices, values = self._read_binary(bin_path, n)
        reconstructed = sp.csc_matrix((values, indices, indptr), shape=(n, n)).toarray()
        np.testing.assert_allclose(reconstructed, 2.0 * K, rtol=1e-9)
        assert nnz == sp.csc_matrix(2.0 * K).nnz
        assert id_path.exists()

    def test_symmetrizes_lower_only_input(self, tmp_path):
        """Caller passes only the lower triangle → writer symmetrizes."""
        n = 4
        lower = np.tril(
            np.array([[1, 0, 0, 0], [2, 3, 0, 0], [0, 0, 4, 0], [0, 0, 5, 6]], dtype=float)
        )
        iids = np.arange(n)
        bin_path, _ = export_sparse_grm_binary(sp.csc_matrix(lower), iids, tmp_path / "L", to_grm=False)
        _, _, indptr, indices, values = self._read_binary(bin_path, n)
        M = sp.csc_matrix((values, indices, indptr), shape=(n, n)).toarray()
        expected = lower + lower.T - np.diag(np.diag(lower))
        np.testing.assert_allclose(M, expected, rtol=1e-9)

    def test_rejects_shape_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="does not match"):
            export_sparse_grm_binary(sp.eye(3, format="csr"), np.arange(4), tmp_path / "x")


class TestExportDenseGrmMph:
    def test_roundtrip_small(self, tmp_path):
        n = 5
        K = np.array(
            [
                [0.50, 0.10, 0.00, 0.00, 0.00],
                [0.10, 0.50, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.55, 0.25, 0.00],
                [0.00, 0.00, 0.25, 0.50, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.50],
            ],
            dtype=np.float64,
        )
        iids = np.array([f"p{i}" for i in range(n)])
        bin_path, iid_path = export_dense_grm_mph(sp.csr_matrix(K), iids, tmp_path / "K", sum2pq=1.5, to_grm=True)
        M, s2pq = _read_gcta_bin(bin_path, n)
        assert s2pq == pytest.approx(1.5, rel=1e-6)
        np.testing.assert_allclose(M, (2.0 * K).astype(np.float32), rtol=1e-6)
        assert iid_path.read_text().splitlines() == list(iids)

    def test_to_grm_false(self, tmp_path):
        K = np.eye(3, dtype=np.float32)
        bin_path, _ = export_dense_grm_mph(K, np.arange(3), tmp_path / "K", to_grm=False)
        M, _ = _read_gcta_bin(bin_path, 3)
        np.testing.assert_allclose(M, K)


# ---------------------------------------------------------------------------
# Household GRM
# ---------------------------------------------------------------------------


class TestExportHouseholdGrm:
    def test_block_diag_structure(self, tmp_path):
        hh = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
        iids = np.arange(len(hh))
        sp_path, _ = export_household_grm(hh, iids, tmp_path / "C")
        C = _read_gcta_sp(sp_path, len(hh))
        # Diagonal
        np.testing.assert_allclose(np.diag(C), 1.0)
        # Within-household pairs
        assert C[0, 1] == 1.0
        assert C[2, 3] == 1.0
        assert C[2, 4] == 1.0
        assert C[3, 4] == 1.0
        # Across-household
        assert C[0, 2] == 0.0
        assert C[0, 5] == 0.0

    def test_singleton_household_diag_only(self, tmp_path):
        hh = np.array([0, 1, 2, 3], dtype=np.int64)
        iids = np.arange(4)
        sp_path, _ = export_household_grm(hh, iids, tmp_path / "C")
        C = _read_gcta_sp(sp_path, 4)
        np.testing.assert_allclose(C, np.eye(4))

    def test_negative_ids_as_singletons(self, tmp_path):
        hh = np.array([-1, -1, 0, 0], dtype=np.int64)
        iids = np.arange(4)
        sp_path, _ = export_household_grm(hh, iids, tmp_path / "C")
        C = _read_gcta_sp(sp_path, 4)
        assert C[0, 1] == 0.0  # -1 sentinels do not share a household
        assert C[2, 3] == 1.0
        np.testing.assert_allclose(np.diag(C), 1.0)

    def test_dense_mph_variant(self, tmp_path):
        hh = np.array([0, 0, 1], dtype=np.int64)
        iids = np.arange(3)
        bin_path, _ = export_household_grm(hh, iids, tmp_path / "C", dense_for_mph=True)
        M, _ = _read_gcta_bin(bin_path, 3)
        expected = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_allclose(M, expected)


# ---------------------------------------------------------------------------
# Phenotype writers
# ---------------------------------------------------------------------------


class TestExportPhenoPlink:
    def test_format_tab_sep_no_header(self, tmp_path):
        df = pd.DataFrame({"id": [10, 11, 12], "y": [1.0, 2.0, 3.0]})
        pheno, covar = export_pheno_plink(df, tmp_path / "x")
        lines = pheno.read_text().splitlines()
        assert lines == ["10\t10\t1.0", "11\t11\t2.0", "12\t12\t3.0"]
        # default: no covariates -> a single "intercept" column of 1.0
        clines = covar.read_text().splitlines()
        assert clines[0] == "10\t10\t1.0"

    def test_named_covars(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "y": [0.1, 0.2], "sex": [0, 1], "age": [25.0, 30.0]})
        _, covar = export_pheno_plink(df, tmp_path / "x", covar_cols=("sex", "age"))
        lines = covar.read_text().splitlines()
        assert lines == ["1\t1\t0\t25.0", "2\t2\t1\t30.0"]

    def test_missing_pheno_col_raises(self, tmp_path):
        df = pd.DataFrame({"id": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            export_pheno_plink(df, tmp_path / "x", pheno_col="y")


class TestExportPhenoCsv:
    def test_format_comma_sep_with_header(self, tmp_path):
        df = pd.DataFrame({"id": [10, 11], "y": [1.0, 2.0]})
        pheno, covar = export_pheno_csv(df, tmp_path / "x")
        pheno_text = pheno.read_text().splitlines()
        assert pheno_text[0] == "id,y"
        assert pheno_text[1] == "10,1.0"
        covar_text = covar.read_text().splitlines()
        assert covar_text[0] == "id,intercept"
        assert covar_text[1] == "10,1.0"

    def test_named_covars(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "y": [0.1, 0.2], "pc1": [0.5, -0.5]})
        _, covar = export_pheno_csv(df, tmp_path / "x", covar_cols=("pc1",))
        lines = covar.read_text().splitlines()
        assert lines[0] == "id,pc1"
        assert lines[1] == "1,0.5"


# ---------------------------------------------------------------------------
# Twin collapse
# ---------------------------------------------------------------------------


class TestCollapseMzTwins:
    def test_drops_exactly_one_per_pair(self):
        df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "twin": [-1, 2, 1, -1, 5, 4],  # (1,2) and (4,5) are twin pairs
            }
        )
        out, _ = collapse_mz_twins(df)
        assert list(out["id"]) == [0, 1, 3, 4]  # drop larger-id partner of each pair

    def test_no_twins_preserves_frame(self):
        df = pd.DataFrame({"id": np.arange(5), "twin": np.full(5, -1)})
        out, _ = collapse_mz_twins(df)
        pd.testing.assert_frame_equal(out, df)

    def test_kinship_submatrix_shape(self):
        df = pd.DataFrame({"id": [0, 1, 2, 3], "twin": [-1, 2, 1, -1]})
        K = sp.eye(4, format="csr")
        _, Kf = collapse_mz_twins(df, K=K)
        assert Kf.shape == (3, 3)
        np.testing.assert_allclose(Kf.toarray(), np.eye(3))

    def test_shape_mismatch_raises(self):
        df = pd.DataFrame({"id": [0, 1], "twin": [-1, -1]})
        with pytest.raises(ValueError, match="does not match"):
            collapse_mz_twins(df, K=sp.eye(3, format="csr"))

    def test_row_index_reset(self):
        df = pd.DataFrame({"id": [0, 1, 2], "twin": [-1, 2, 1]}, index=[5, 6, 7])
        out, _ = collapse_mz_twins(df)
        assert list(out.index) == [0, 1]
