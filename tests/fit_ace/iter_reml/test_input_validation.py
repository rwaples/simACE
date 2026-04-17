"""Input-validation tests for ``fit_ace.iter_reml.fit_iter_reml``.

All tests here never reach the subprocess — the wrapper validates
arguments first.  Each test points ``binary=`` at a touched but
non-executable temp file so the FileNotFoundError path (raised before
argument validation) does not fire.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from fit_ace.iter_reml.fit import fit_iter_reml


def _fake_bin(tmp_path):
    f = tmp_path / "fake_bin"
    f.touch()
    return f


class TestShapeValidation:
    def test_kinship_shape_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="kinship shape"):
            fit_iter_reml(
                y=np.zeros(5),
                kinship=sp.eye(4, format="csr"),
                household_id=np.zeros(5, dtype=int),
                binary=_fake_bin(tmp_path),
            )

    def test_household_shape_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="household_id shape"):
            fit_iter_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                household_id=np.array([0, 0]),
                binary=_fake_bin(tmp_path),
            )

    def test_iids_shape_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="iids shape"):
            fit_iter_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                household_id=np.zeros(3, dtype=int),
                iids=np.array(["a", "b"]),
                binary=_fake_bin(tmp_path),
            )


class TestFiniteY:
    def test_nan_in_y(self, tmp_path):
        y = np.array([0.0, np.nan, 0.0])
        with pytest.raises(ValueError, match="NaN or inf"):
            fit_iter_reml(
                y=y,
                kinship=sp.eye(3, format="csr"),
                household_id=np.zeros(3, dtype=int),
                binary=_fake_bin(tmp_path),
            )

    def test_inf_in_y(self, tmp_path):
        y = np.array([0.0, np.inf, 0.0])
        with pytest.raises(ValueError, match="NaN or inf"):
            fit_iter_reml(
                y=y,
                kinship=sp.eye(3, format="csr"),
                household_id=np.zeros(3, dtype=int),
                binary=_fake_bin(tmp_path),
            )


class TestHouseholdRequired:
    def test_household_id_required(self, tmp_path):
        """The A+C+E model (v1) requires household_id — there's no
        AE-only mode in the wrapper."""
        with pytest.raises(ValueError, match="household_id is required"):
            fit_iter_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                binary=_fake_bin(tmp_path),
            )


class TestInputSourceExclusive:
    def test_both_kinship_and_pedigree_arrays(self, tmp_path):
        ped_arrays = (
            np.arange(3),
            np.full(3, -1),
            np.full(3, -1),
            np.full(3, -1),
        )
        with pytest.raises(ValueError, match="pass exactly one"):
            fit_iter_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                pedigree_arrays=ped_arrays,
                household_id=np.zeros(3, dtype=int),
                binary=_fake_bin(tmp_path),
            )

    def test_neither_kinship_nor_pedigree_arrays(self, tmp_path):
        with pytest.raises(ValueError, match="pass exactly one"):
            fit_iter_reml(
                y=np.zeros(3),
                household_id=np.zeros(3, dtype=int),
                binary=_fake_bin(tmp_path),
            )


class TestInvalidChoices:
    @pytest.mark.parametrize(
        ("kw", "bad_val", "match"),
        [
            ("pc_type", "foo", r"pc_type='foo'"),
            ("pc_type", "hypre", r"pc_type='hypre'"),  # binary has it; wrapper doesn't
            ("trace_method", "hutch", r"trace_method='hutch'"),
        ],
    )
    def test_invalid_choice_raises(self, tmp_path, kw, bad_val, match):
        with pytest.raises(ValueError, match=match):
            fit_iter_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                household_id=np.zeros(3, dtype=int),
                binary=_fake_bin(tmp_path),
                **{kw: bad_val},
            )

    def test_negative_hutchpp_sketch_size(self, tmp_path):
        with pytest.raises(ValueError, match="hutchpp_sketch_size"):
            fit_iter_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                household_id=np.zeros(3, dtype=int),
                binary=_fake_bin(tmp_path),
                hutchpp_sketch_size=-1,
            )


class TestMissingBinary:
    def test_missing_binary_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ace_iter_reml binary not found"):
            fit_iter_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                household_id=np.zeros(3, dtype=int),
                binary=tmp_path / "does_not_exist",
            )
