"""Unit tests for fit_ace.bench.reml helpers."""

import math
import textwrap

import numpy as np
import pandas as pd
import pytest

from fit_ace.bench.reml import (
    build_kinship_for_subset,
    parse_mph_vc_csv,
    parse_sparse_reml_tsv,
    run_timed_subprocess,
)

# ---------------------------------------------------------------------------
# parse_sparse_reml_tsv
# ---------------------------------------------------------------------------


class TestParseSparseRemlTsv:
    def test_basic_tsv(self, tmp_path):
        vc_path = tmp_path / "vc.tsv"
        vc_path.write_text(
            textwrap.dedent(
                """\
                vc_name\testimate\tse
                V(A)\t0.4982\t0.0143
                V(C)\t0.1931\t0.0089
                Ve\t0.3024\t0.0072
                Vp\t0.9937\t0.0102
                h2\t0.5014\t0.0141
                c2\t0.1944\t0.0089
                """
            )
        )
        out = parse_sparse_reml_tsv(vc_path)
        assert out["est_V(A)"] == pytest.approx(0.4982)
        assert out["se_V(A)"] == pytest.approx(0.0143)
        assert out["est_h2"] == pytest.approx(0.5014)

    def test_meta_sidecar_parsed(self, tmp_path):
        vc_path = tmp_path / "vc.tsv"
        vc_path.write_text("vc_name\testimate\tse\nV(A)\t0.5\t0.1\n")
        (tmp_path / "vc.tsv.meta").write_text("wall_s\t12.345\nn_iter\t7\nconverged\t1\nlogLik\t-42.0\n")
        out = parse_sparse_reml_tsv(vc_path)
        assert out["reml_wall_s"] == pytest.approx(12.345)
        assert out["n_iter"] == 7
        assert out["converged"] == 1
        assert out["logLik"] == pytest.approx(-42.0)

    def test_missing_meta_is_soft(self, tmp_path):
        vc_path = tmp_path / "vc.tsv"
        vc_path.write_text("vc_name\testimate\tse\nV(A)\t0.5\t0.1\n")
        out = parse_sparse_reml_tsv(vc_path)
        assert "reml_wall_s" not in out  # no meta file → no wall


# ---------------------------------------------------------------------------
# parse_mph_vc_csv
# ---------------------------------------------------------------------------


class TestParseMphVcCsv:
    def _write_example(self, tmp_path, a_var=0.5, c_var=0.2, e_var=0.3, a_pve=0.5, c_pve=0.2):
        # Minimal MPH vc.csv shape: trait_x, trait_y, vc_name, m, var, seV, pve, seP
        # (extra columns ignored by parser)
        path = tmp_path / "out.mq.vc.csv"
        path.write_text(
            "trait_x,trait_y,vc_name,m,var,seV,pve,seP\n"
            f"y,y,/tmp/x/Adense,1.0,{a_var},0.011,{a_pve},0.01\n"
            f"y,y,/tmp/x/Cdense,1.0,{c_var},0.009,{c_pve},0.008\n"
            f"y,y,err,NA,{e_var},0.007,NA,NA\n"
        )
        return path

    def test_positional_mapping(self, tmp_path):
        out = parse_mph_vc_csv(self._write_example(tmp_path))
        assert out["est_V(A)"] == pytest.approx(0.5)
        assert out["est_V(C)"] == pytest.approx(0.2)
        assert out["est_Ve"] == pytest.approx(0.3)
        assert out["se_V(A)"] == pytest.approx(0.011)
        assert out["se_Ve"] == pytest.approx(0.007)

    def test_h2_and_c2_from_pve(self, tmp_path):
        out = parse_mph_vc_csv(self._write_example(tmp_path, a_pve=0.5, c_pve=0.25))
        assert out["est_h2"] == pytest.approx(0.5)
        assert out["est_c2"] == pytest.approx(0.25)

    def test_vp_is_sum_of_vcs(self, tmp_path):
        out = parse_mph_vc_csv(self._write_example(tmp_path, 0.4, 0.2, 0.4))
        assert out["est_Vp"] == pytest.approx(1.0)
        assert math.isnan(out["se_Vp"])

    def test_single_grm_no_c_row(self, tmp_path):
        path = tmp_path / "single.mq.vc.csv"
        path.write_text(
            "trait_x,trait_y,vc_name,m,var,seV,pve,seP\ny,y,/tmp/A,1.0,0.6,0.01,0.6,0.01\ny,y,err,NA,0.4,0.008,NA,NA\n"
        )
        out = parse_mph_vc_csv(path)
        assert out["est_V(A)"] == pytest.approx(0.6)
        assert "est_V(C)" not in out
        assert out["est_Ve"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# run_timed_subprocess
# ---------------------------------------------------------------------------


class TestRunTimedSubprocess:
    def test_true_returns_zero(self, tmp_path):
        wall, rss, rc = run_timed_subprocess(["true"], tmp_path / "log.txt")
        assert rc == 0
        assert wall >= 0.0
        assert rss >= 0.0

    def test_false_returns_nonzero(self, tmp_path):
        _, _, rc = run_timed_subprocess(["false"], tmp_path / "log.txt")
        assert rc != 0

    def test_stdout_captured(self, tmp_path):
        log = tmp_path / "hello.log"
        run_timed_subprocess(["printf", "hello\n"], log)
        assert log.read_text() == "hello\n"


# ---------------------------------------------------------------------------
# build_kinship_for_subset
# ---------------------------------------------------------------------------


def _tiny_pedigree():
    """3-generation pedigree: 2 founders, 2 children, 1 grandchild.

    Layout:
        id  mother  father  twin  sex  generation
         0     -1      -1    -1    0          0
         1     -1      -1    -1    1          0
         2      0       1    -1    0          1
         3      0       1    -1    1          1
         4      2       1    -1    0          2    (avoid inbreeding loops; id=4 is
                                                     child of id=2 and id=1 here
                                                     just for a tiny test case)

    For a cleaner test we keep it non-inbred by having gen-2's father be a
    generation-0 founder (id=1).  That produces grandparent/half-sib edges.
    """
    rows = [
        {"id": 0, "mother": -1, "father": -1, "twin": -1, "sex": 0, "generation": 0, "household_id": -1},
        {"id": 1, "mother": -1, "father": -1, "twin": -1, "sex": 1, "generation": 0, "household_id": -1},
        {"id": 2, "mother": 0, "father": 1, "twin": -1, "sex": 0, "generation": 1, "household_id": 0},
        {"id": 3, "mother": 0, "father": 1, "twin": -1, "sex": 1, "generation": 1, "household_id": 0},
        {"id": 4, "mother": 2, "father": 1, "twin": -1, "sex": 0, "generation": 2, "household_id": 1},
    ]
    return pd.DataFrame(rows)


class TestBuildKinshipForSubset:
    def test_subset_shape(self):
        ped = _tiny_pedigree()
        subset = ped.iloc[2:5].reset_index(drop=True)  # gen 1 and 2 individuals
        K_sub = build_kinship_for_subset(ped, subset)
        assert K_sub.shape == (3, 3)

    def test_full_subset_matches_full_kinship(self):
        """Subsetting to the entire pedigree should return the full kinship."""
        from sim_ace.core.pedigree_graph import PedigreeGraph

        ped = _tiny_pedigree()
        pg = PedigreeGraph(ped)
        pg.compute_inbreeding()
        K_full = pg._kinship_matrix.toarray()
        K_sub = build_kinship_for_subset(ped, ped).toarray()
        np.testing.assert_allclose(K_sub, K_full)

    def test_siblings_have_expected_kinship(self):
        """Full siblings (id=2, id=3) should have kinship 0.25 in a non-inbred pedigree."""
        ped = _tiny_pedigree()
        subset = ped.iloc[[2, 3]].reset_index(drop=True)  # full sib pair
        K_sub = build_kinship_for_subset(ped, subset).toarray()
        # Diagonal: 0.5 for non-inbred founders' offspring
        np.testing.assert_allclose(np.diag(K_sub), 0.5, rtol=1e-9)
        # Off-diagonal: FS kinship = 0.25
        assert K_sub[0, 1] == pytest.approx(0.25, abs=1e-9)

    def test_missing_id_raises(self):
        ped = _tiny_pedigree()
        bogus = ped.copy()
        bogus.loc[0, "id"] = 999  # a subset referencing an id not in the "full" df
        subset = pd.DataFrame({"id": [999]})
        # Now `full_ped` doesn't contain id 999 that subset expects.
        with pytest.raises(ValueError, match="absent"):
            # Use ped (which now has id 999 changed... wait, we need full_ped WITHOUT 999)
            build_kinship_for_subset(ped.iloc[1:].reset_index(drop=True), subset)
