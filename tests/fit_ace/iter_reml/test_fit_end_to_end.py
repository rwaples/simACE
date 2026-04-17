"""End-to-end integration tests for ``fit_iter_reml``.

These shell out to the compiled ``ace_iter_reml`` binary and are
skipped when neither the fp32 nor fp64 build is present on disk.

All tests run with ``threads=1`` to avoid the known PETSc thread-safety
race (see ``memory/project_iter_reml_fp32_cleanup_crash.md``).  Probe
counts are reduced from production defaults so the tiny N≈300 fixture
finishes in a few seconds per case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fit_ace.iter_reml.fit import fit_iter_reml
from tests.fit_ace.iter_reml.conftest import needs_bin

# Tolerance on σ² recovery at N≈300 with 30 Hutchinson probes.  Matches
# the sparse_reml smoke test band — Hutchinson trace variance dominates
# at this sample size.
_VC_TOL = 0.25
# Looser band for Hutch++ on pedigree kinship (known ~1.2× worse variance;
# see fit.py:204 and notes/iter_reml_adaptive_baselines/).
_HUTCHPP_TOL = 0.35
# Looser band when Phase 1 warm-start is skipped.
_COLD_START_TOL = 0.35


def _assert_vc_close_to_truth(r, truth, tol=_VC_TOL):
    vc = r.vc.set_index("vc_name")
    for name, key in [("V(A)", "var_a"), ("V(C)", "var_c"), ("Ve", "var_e")]:
        est = float(vc.loc[name, "estimate"])
        ref = truth[key]
        assert est == pytest.approx(ref, abs=tol), f"{name}: est={est:.4f} truth={ref:.4f}"


@needs_bin
class TestFitJacobi:
    def test_smoke_ace_jacobi(self, tiny_ace_inputs, fast_kwargs):
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            **fast_kwargs,
        )
        assert r.converged, "Jacobi smoke should converge at N≈300"
        _assert_vc_close_to_truth(r, inp["truth"])
        assert r.cov.shape == (3, 3)
        assert {"iter", "logLik", "dLLpred", "grad_norm", "VC_A", "VC_C", "VC_E"}.issubset(r.iter_log.columns)
        assert np.isfinite(r.logLik), "compute_logdet=True by default → finite logLik"
        assert r.phase1_vc is not None, "Phase 1 RHE-mc warm-start should be emitted"
        assert r.bench is not None
        assert len(r.bench) > 0

    def test_pedigree_arrays_path_matches_kinship_path(self, tiny_ace_inputs, fast_kwargs):
        """The fast pedigree_arrays path builds kinship inside the
        wrapper via numba+ACEGRM in one pass.  It must produce identical
        σ² to the scipy-sparse path on the same inputs and seed."""
        inp = tiny_ace_inputs
        r_ped = fit_iter_reml(
            y=inp["y"],
            pedigree_arrays=inp["pedigree_arrays"],
            phen_ids_in_pedigree=inp["phen_ids"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            **fast_kwargs,
        )
        r_sp = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            **fast_kwargs,
        )
        assert r_ped.converged
        assert r_sp.converged
        vc_ped = r_ped.vc.set_index("vc_name")["estimate"]
        vc_sp = r_sp.vc.set_index("vc_name")["estimate"]
        for name in ("V(A)", "V(C)", "Ve"):
            assert float(vc_ped[name]) == pytest.approx(float(vc_sp[name]), abs=1e-3), (
                f"pedigree_arrays vs kinship diverged on {name}: {vc_ped[name]:.6f} vs {vc_sp[name]:.6f}"
            )

    def test_deterministic_seed(self, tiny_ace_inputs, fast_kwargs):
        """Same seed + threads=1 → bit-identical σ² across two runs."""
        inp = tiny_ace_inputs
        kw = {**fast_kwargs, "seed": 12345}
        r1 = fit_iter_reml(y=inp["y"], kinship=inp["K"], household_id=inp["household_id"], iids=inp["iids"], **kw)
        r2 = fit_iter_reml(y=inp["y"], kinship=inp["K"], household_id=inp["household_id"], iids=inp["iids"], **kw)
        v1 = r1.vc.set_index("vc_name")["estimate"]
        v2 = r2.vc.set_index("vc_name")["estimate"]
        for name in ("V(A)", "V(C)", "Ve"):
            assert float(v1[name]) == float(v2[name]), f"non-deterministic on {name}: {v1[name]} vs {v2[name]}"

    def test_emit_probe_traces(self, tiny_ace_inputs, fast_kwargs, tmp_path):
        """emit_probe_traces=True writes Phase 1 and Phase 2 probe
        contribution TSVs into the work dir (column schemas from
        ai_reml.cpp:852 and rhe_mc.cpp:155)."""
        inp = tiny_ace_inputs
        work = tmp_path / "work"
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            emit_probe_traces=True,
            work_dir=work,
            cleanup=False,
            **fast_kwargs,
        )
        assert r.converged

        ph2 = work / "out.probes.tsv"
        ph1 = work / "out.probes.tsv.phase1"
        assert ph2.exists(), "Phase 2 probe traces not emitted"
        assert ph1.exists(), "Phase 1 probe traces not emitted"

        df2 = pd.read_csv(ph2, sep="\t")
        assert set(df2.columns) == {"iter", "vc_name", "probe_idx", "trace_contrib"}
        assert set(df2["vc_name"].unique()) <= {"A", "C", "E"}
        assert df2["iter"].max() + 1 == r.n_iter, "probe iter range should match n_iter"
        assert len(df2) > 0

        df1 = pd.read_csv(ph1, sep="\t")
        assert set(df1.columns) == {"probe_idx", "T_AA", "T_AC", "T_CC"}
        assert len(df1) == fast_kwargs["phase1_probes"]


@needs_bin
class TestFitComputeLogdet:
    def test_no_compute_logdet(self, tiny_ace_inputs, fast_kwargs):
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            compute_logdet=False,
            **fast_kwargs,
        )
        assert r.converged
        assert np.isnan(r.logLik)
        assert r.iter_log["logLik"].isna().all()
        # σ² path unaffected.
        _assert_vc_close_to_truth(r, inp["truth"])

    def test_compute_logdet_produces_finite_logLik(self, tiny_ace_inputs, fast_kwargs):
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            compute_logdet=True,
            **fast_kwargs,
        )
        assert np.isfinite(r.logLik)
        assert r.iter_log["logLik"].notna().all()


@needs_bin
class TestFitSkipPhases:
    def test_skip_phase1(self, tiny_ace_inputs, fast_kwargs):
        """Phase 2 only; σ² initialised from var(y)/3 with no warm-start.
        Tolerance is loosened because convergence may take more
        iterations without the RHE-mc warm-start."""
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            skip_phase1=True,
            **fast_kwargs,
        )
        assert r.converged
        _assert_vc_close_to_truth(r, inp["truth"], tol=_COLD_START_TOL)
        assert r.phase1_vc is None

    def test_skip_phase2(self, tiny_ace_inputs, fast_kwargs):
        """Only Phase 1 runs; vc populated from phase1.tsv; no AI
        covariance; no per-iter log."""
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            skip_phase2=True,
            **fast_kwargs,
        )
        assert r.cov.empty
        assert r.iter_log.empty
        assert r.converged is False  # no Phase 2 meta written
        assert r.n_iter == 0
        # vc populated from phase1.tsv
        assert len(r.vc) >= 3
        names = set(r.vc["vc_name"])
        assert {"V(A)", "V(C)", "Ve"}.issubset(names)


@needs_bin
class TestFitPreconditioners:
    def test_deflation_pc(self, tiny_ace_inputs, fast_kwargs):
        """pc_type='deflation' (top-k SVD of A + Jacobi on complement)
        must converge to the same σ² band as Jacobi on this tiny
        fixture.  Skipped if the build lacks SLEPc."""
        inp = tiny_ace_inputs
        try:
            r = fit_iter_reml(
                y=inp["y"],
                kinship=inp["K"],
                household_id=inp["household_id"],
                iids=inp["iids"],
                pc_type="deflation",
                deflation_k=50,
                **fast_kwargs,
            )
        except RuntimeError as e:
            if "SLEPc" in str(e) or "slepc" in str(e).lower():
                pytest.skip(f"binary built without SLEPc: {e}")
            raise
        assert r.converged
        _assert_vc_close_to_truth(r, inp["truth"])


@needs_bin
class TestFitTraceMethods:
    def test_hutchpp_schema(self, tiny_ace_inputs, fast_kwargs):
        """Hutch++ is known to be ~1.2× worse on pedigree kinship
        (fit.py:204).  Loose σ² band; primary goal is that the code
        path produces the full output contract."""
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            trace_method="hutchpp",
            **fast_kwargs,
        )
        assert r.converged
        _assert_vc_close_to_truth(r, inp["truth"], tol=_HUTCHPP_TOL)
        assert r.cov.shape == (3, 3)
        assert len(r.iter_log) >= 1

    def test_hutchpp_explicit_sketch_size(self, tiny_ace_inputs, fast_kwargs):
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            trace_method="hutchpp",
            hutchpp_sketch_size=10,
            **fast_kwargs,
        )
        assert r.converged
        _assert_vc_close_to_truth(r, inp["truth"], tol=_HUTCHPP_TOL)


@needs_bin
class TestFitOutputContract:
    def test_work_dir_preserved_when_cleanup_false(self, tiny_ace_inputs, fast_kwargs, tmp_path):
        """cleanup=False + explicit work_dir leaves both the staged
        binary inputs and the binary's output files on disk for
        inspection."""
        inp = tiny_ace_inputs
        work = tmp_path / "work"
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            work_dir=work,
            cleanup=False,
            **fast_kwargs,
        )
        assert r.converged
        # Staged inputs.
        assert (work / "inputs" / "A.grm.sp.bin").exists()
        assert (work / "inputs" / "A.grm.id").exists()
        assert (work / "inputs" / "C.grm.sp.bin").exists()
        assert (work / "inputs" / "plink.pheno.txt").exists()
        # Binary outputs.
        assert (work / "out.vc.tsv").exists()
        assert (work / "out.iter.tsv").exists()
        assert (work / "out.cov.tsv").exists()
        assert (work / "out.bench.tsv").exists()

    def test_iter_log_columns_match_binary_schema(self, tiny_ace_inputs, fast_kwargs):
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            **fast_kwargs,
        )
        # Exact columns the binary writes at ai_reml.cpp; wrapper does
        # not add/remove.
        expected_core = {
            "iter",
            "logLik",
            "dLLpred",
            "tau_nr",
            "grad_norm",
            "VC_A",
            "VC_C",
            "VC_E",
            "pcg_iters_avg",
        }
        assert expected_core.issubset(r.iter_log.columns)
        # All σ² columns are positive at every iter (REML with floor).
        for col in ("VC_A", "VC_C", "VC_E"):
            assert (r.iter_log[col] > 0).all(), f"{col} has non-positive entry"

    def test_bench_has_expected_stages(self, tiny_ace_inputs, fast_kwargs):
        inp = tiny_ace_inputs
        r = fit_iter_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            iids=inp["iids"],
            **fast_kwargs,
        )
        assert r.bench is not None
        stages = set(r.bench["stage"].values)
        # At least the top-level phase timers — exact inner stages may
        # shift as the binary evolves; don't over-pin.
        assert any(s.startswith("phase1") for s in stages)
        assert any(s.startswith("phase2") for s in stages)
        assert (r.bench["wall_s"] >= 0).all()
        assert (r.bench["calls"] >= 0).all()
