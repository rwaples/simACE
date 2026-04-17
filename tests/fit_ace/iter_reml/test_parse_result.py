"""Unit tests for ``fit_ace.iter_reml.fit._parse_result``.

Builds synthetic TSVs that match the binary's on-disk schema (from
``ai_reml.cpp`` and ``main.cpp``) and verifies the parser rehydrates
them into an ``IterREMLResult`` correctly.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from fit_ace.iter_reml.fit import IterREMLResult, _parse_result


def _write_vc_tsv(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            vc_name\testimate\tse
            V(A)\t0.5021\t0.0412
            V(C)\t0.1987\t0.0203
            Ve\t0.3014\t0.0189
            Vp\t1.0022\t0.0283
            h2\t0.5010\t0.0311
            c2\t0.1983\t0.0177
            """
        )
    )


def _write_cov_tsv(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            name\tV(A)\tV(C)\tVe
            V(A)\t0.001697\t-0.000410\t-0.000251
            V(C)\t-0.000410\t0.000412\t-0.000092
            Ve\t-0.000251\t-0.000092\t0.000357
            """
        )
    )


def _write_iter_tsv(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            iter\tlogLik\tdLLpred\ttau_nr\tgrad_norm\tVC_A\tVC_C\tVC_E\tpcg_iters_avg\tgrad_A\tgrad_C\tgrad_E\tpcg_iters_min\tpcg_iters_max\tdelta_trust\trss_mb\ttr_hat_A\ttr_hat_C\ttr_hat_E\tlog_det_V\ty_Vinv_y
            0\t-420.5\t85.1\t1.00\t210.4\t0.333\t0.333\t0.334\t48.2\t-120.0\t45.0\t10.0\t42\t55\t1.00\t48.2\t140.0\t90.0\t150.0\t-540.0\t301.0
            1\t-355.1\t2.10\t1.00\t22.0\t0.498\t0.201\t0.301\t33.6\t-15.0\t8.0\t2.0\t30\t40\t1.00\t52.8\t155.0\t80.0\t145.0\t-510.0\t200.2
            2\t-354.7\t0.002\t1.00\t0.3\t0.5021\t0.1987\t0.3014\t28.0\t-0.15\t0.10\t0.05\t25\t33\t1.00\t55.0\t160.0\t82.0\t143.0\t-508.0\t201.4
            """
        )
    )


def _write_bench_tsv(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            stage\twall_s\tcalls\tmean_ms
            phase1/total\t0.85\t1\t850.0
            phase1/AZ\t0.21\t30\t7.0
            phase2/total\t2.10\t1\t2100.0
            phase2/pcg_solve\t1.45\t90\t16.1
            io/read_grm_binary\t0.02\t2\t10.0
            """
        )
    )


def _write_phase1_tsv(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            vc_name\testimate\tse
            V(A)\t0.505\t0.0615
            V(C)\t0.195\t0.0304
            Ve\t0.300\t0.0282
            Vp\t1.000\t0.0424
            h2\t0.505\t0.0466
            c2\t0.195\t0.0266
            """
        )
    )


def _write_meta(path: Path, *, logLik: str = "-354.7") -> None:
    path.write_text(f"n_iter\t3\nconverged\t1\nlogLik\t{logLik}\nwall_s\t3.45\npeak_rss_mb\t62.1\n")


def _write_full(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_vc_tsv(Path(f"{out}.vc.tsv"))
    _write_cov_tsv(Path(f"{out}.cov.tsv"))
    _write_iter_tsv(Path(f"{out}.iter.tsv"))
    _write_bench_tsv(Path(f"{out}.bench.tsv"))
    _write_phase1_tsv(Path(f"{out}.phase1.tsv"))
    _write_meta(Path(f"{out}.vc.tsv.meta"))


class TestParseResultFull:
    def test_all_fields_populated(self, tmp_path):
        out = tmp_path / "out"
        _write_full(out)
        r = _parse_result(out, ["dummy"])

        assert isinstance(r, IterREMLResult)
        assert list(r.vc.columns) == ["vc_name", "estimate", "se"]
        assert r.vc.loc[r.vc["vc_name"] == "h2", "estimate"].iloc[0] == pytest.approx(0.5010)
        assert r.cov.loc["V(A)", "V(A)"] == pytest.approx(0.001697)
        assert r.cov.shape == (3, 3)
        assert len(r.iter_log) == 3
        assert {"iter", "logLik", "dLLpred", "grad_norm", "VC_A", "VC_C", "VC_E"}.issubset(r.iter_log.columns)
        assert r.wall_s == pytest.approx(3.45)
        assert r.n_iter == 3
        assert r.converged is True
        assert r.logLik == pytest.approx(-354.7)
        assert r.bench is not None
        assert "phase1/total" in r.bench["stage"].values
        assert "phase2/total" in r.bench["stage"].values
        assert r.phase1_vc is not None
        assert r.phase1_vc.loc[r.phase1_vc["vc_name"] == "V(A)", "estimate"].iloc[0] == pytest.approx(0.505)
        assert r.command == ["dummy"]


class TestParseResultMeta:
    def test_missing_meta_defaults(self, tmp_path):
        out = tmp_path / "out"
        _write_full(out)
        Path(f"{out}.vc.tsv.meta").unlink()
        r = _parse_result(out, [])
        assert r.converged is False
        assert r.n_iter == 0
        assert np.isnan(r.wall_s)
        assert np.isnan(r.logLik)

    def test_logLik_nan_in_meta(self, tmp_path):
        out = tmp_path / "out"
        _write_full(out)
        _write_meta(Path(f"{out}.vc.tsv.meta"), logLik="nan")
        r = _parse_result(out, [])
        assert np.isnan(r.logLik)
        # Other meta fields still parse normally.
        assert r.converged is True
        assert r.n_iter == 3

    def test_converged_zero(self, tmp_path):
        out = tmp_path / "out"
        _write_full(out)
        Path(f"{out}.vc.tsv.meta").write_text("n_iter\t50\nconverged\t0\nlogLik\t-400.0\nwall_s\t12.0\n")
        r = _parse_result(out, [])
        assert r.converged is False
        assert r.n_iter == 50


class TestParseResultBench:
    def test_missing_bench_returns_none(self, tmp_path):
        out = tmp_path / "out"
        _write_full(out)
        Path(f"{out}.bench.tsv").unlink()
        r = _parse_result(out, [])
        assert r.bench is None


class TestParseResultPhase1:
    def test_phase1_present_populates(self, tmp_path):
        out = tmp_path / "out"
        _write_full(out)
        r = _parse_result(out, [])
        assert r.phase1_vc is not None
        assert set(r.phase1_vc.columns) == {"vc_name", "estimate", "se"}

    def test_phase1_absent_is_none(self, tmp_path):
        out = tmp_path / "out"
        _write_full(out)
        Path(f"{out}.phase1.tsv").unlink()
        r = _parse_result(out, [])
        assert r.phase1_vc is None


class TestParseResultSkipPhase2:
    def test_skip_phase2_reads_only_phase1(self, tmp_path):
        """When Phase 2 is skipped the binary does not emit .vc.tsv /
        .cov.tsv / .iter.tsv; ``_parse_result`` in that mode populates
        ``vc`` from the phase1.tsv and returns empty DataFrames for
        ``cov`` and ``iter_log``."""
        out = tmp_path / "out"
        out.parent.mkdir(exist_ok=True)
        _write_phase1_tsv(Path(f"{out}.phase1.tsv"))
        # No vc.tsv / cov.tsv / iter.tsv — matches skip_phase2 output.
        r = _parse_result(out, [], skip_phase2=True)
        assert len(r.vc) == 6  # V(A), V(C), Ve, Vp, h2, c2
        assert r.cov.empty
        assert r.iter_log.empty

    def test_skip_phase1_but_still_reads_vc(self, tmp_path):
        """skip_phase1=True suppresses the phase1_vc attachment but the
        Phase 2 files (vc/cov/iter) still parse normally."""
        out = tmp_path / "out"
        _write_full(out)
        # Even though .phase1.tsv exists, skip_phase1=True → phase1_vc None.
        r = _parse_result(out, [], skip_phase1=True)
        assert r.phase1_vc is None
        assert len(r.vc) == 6
        assert r.cov.shape == (3, 3)
