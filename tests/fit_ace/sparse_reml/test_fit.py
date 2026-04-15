"""Tests for fit_ace.sparse_reml.fit_sparse_reml.

Unit tests that don't need the compiled binary run unconditionally; the
integration tests are skipped when ``ace_sreml`` isn't on disk.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from fit_ace.sparse_reml.fit import (
    SparseREMLResult,
    _parse_result,
    default_binary,
    fit_sparse_reml,
)

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


class TestDefaultBinary:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        fake = tmp_path / "fake_bin"
        fake.touch()
        monkeypatch.setenv("ACE_SREML_BIN", str(fake))
        assert default_binary() == fake

    def test_falls_back_to_repo_default(self, monkeypatch):
        monkeypatch.delenv("ACE_SREML_BIN", raising=False)
        p = default_binary()
        # Only shape, not presence — the default lives under external/ace_sreml/build.
        assert p.name == "ace_sreml"


# ---------------------------------------------------------------------------
# TSV parser (unit tests against synthetic files)
# ---------------------------------------------------------------------------


def _write_fake_outputs(out_prefix: Path) -> None:
    """Drop vc/cov/iter TSVs + meta file that mimic real binary output."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out_prefix}.vc.tsv").write_text(
        textwrap.dedent(
            """\
            vc_name\testimate\tse
            V(A)\t0.5001\t0.0132
            V(C)\t0.1998\t0.0071
            Ve\t0.3003\t0.0084
            Vp\t1.0002\t0.0099
            h2\t0.5000\t0.0108
            c2\t0.1997\t0.0066
            """
        )
    )
    Path(f"{out_prefix}.cov.tsv").write_text(
        textwrap.dedent(
            """\
            name\tV(A)\tV(C)\tVe
            V(A)\t0.000175\t0.0000\t-0.0001
            V(C)\t0.0000\t0.0000505\t-0.0001
            Ve\t-0.0001\t-0.0001\t0.0000706
            """
        )
    )
    Path(f"{out_prefix}.iter.tsv").write_text(
        textwrap.dedent(
            """\
            iter\tlogLik\tdLLpred\ttau_nr\tgrad_norm\tV(A)\tV(C)\tVe
            0\t-1300.0\t150.0\t1.0\t500.0\t0.1\t0.1\t0.8
            1\t-1150.0\t0.5\t1.0\t30.0\t0.48\t0.20\t0.30
            2\t-1148.0\t0.001\t1.0\t1.3\t0.50\t0.20\t0.30
            """
        )
    )
    Path(f"{out_prefix}.vc.tsv.meta").write_text("wall_s\t12.34\nn_iter\t3\nconverged\t1\nlogLik\t-1148.0\n")
    Path(f"{out_prefix}.bench.tsv").write_text(
        "stage\twall_s\tcalls\tmean_ms\nreml/factorize\t3.2\t3\t1070\nreml/iter/solve_gz\t5.9\t3\t1970\n"
    )


class TestParseResult:
    def test_full_outputs(self, tmp_path):
        out = tmp_path / "out"
        _write_fake_outputs(out)
        r = _parse_result(out, ["dummy_cmd"])
        assert isinstance(r, SparseREMLResult)
        assert list(r.vc.columns) == ["vc_name", "estimate", "se"]
        assert r.vc.loc[r.vc["vc_name"] == "h2", "estimate"].iloc[0] == pytest.approx(0.5)
        assert r.cov.loc["V(A)", "V(A)"] == pytest.approx(0.000175)
        assert len(r.iter_log) == 3
        assert r.wall_s == pytest.approx(12.34)
        assert r.n_iter == 3
        assert r.converged is True
        assert r.logLik == pytest.approx(-1148.0)
        assert r.bench is not None
        assert "reml/factorize" in r.bench["stage"].values

    def test_missing_meta_defaults(self, tmp_path):
        out = tmp_path / "out"
        _write_fake_outputs(out)
        Path(f"{out}.vc.tsv.meta").unlink()
        r = _parse_result(out, [])
        assert r.converged is False
        assert np.isnan(r.wall_s)

    def test_missing_bench_returns_none(self, tmp_path):
        out = tmp_path / "out"
        _write_fake_outputs(out)
        Path(f"{out}.bench.tsv").unlink()
        r = _parse_result(out, [])
        assert r.bench is None


# ---------------------------------------------------------------------------
# Input validation (no binary needed: we fail fast)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_kinship_shape_mismatch(self, tmp_path):
        fake = tmp_path / "fake_bin"
        fake.touch()
        with pytest.raises(ValueError, match="kinship shape"):
            fit_sparse_reml(y=np.zeros(5), kinship=sp.eye(4, format="csr"), binary=fake)

    def test_nan_in_y(self, tmp_path):
        fake = tmp_path / "fake_bin"
        fake.touch()
        y = np.array([0.0, np.nan, 0.0])
        with pytest.raises(ValueError, match="NaN"):
            fit_sparse_reml(y=y, kinship=sp.eye(3, format="csr"), binary=fake)

    def test_household_shape_mismatch(self, tmp_path):
        fake = tmp_path / "fake_bin"
        fake.touch()
        with pytest.raises(ValueError, match="household_id"):
            fit_sparse_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                household_id=np.array([0, 0]),
                binary=fake,
            )

    def test_covariates_shape_mismatch(self, tmp_path):
        fake = tmp_path / "fake_bin"
        fake.touch()
        with pytest.raises(ValueError, match="covariates"):
            fit_sparse_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                covariates=np.zeros((4, 1)),
                binary=fake,
            )

    def test_missing_binary_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ace_sreml binary not found"):
            fit_sparse_reml(
                y=np.zeros(3),
                kinship=sp.eye(3, format="csr"),
                binary=tmp_path / "does_not_exist",
            )


# ---------------------------------------------------------------------------
# Integration — real binary, small simulated pedigree.  Skipped if the
# binary hasn't been built.
# ---------------------------------------------------------------------------


_BIN = default_binary()
_HAS_BIN = _BIN.exists()
needs_bin = pytest.mark.skipif(not _HAS_BIN, reason=f"ace_sreml binary not built at {_BIN}")


@pytest.fixture(scope="module")
def tiny_ace_inputs():
    """A small pedigree's (y, kinship, household_id) for a round-trip fit.

    Uses the simulator + PedigreeGraph to produce a 2-generation pedigree
    where the ACE signal is well-identified.  Output sizes are small enough
    that the binary finishes in seconds.
    """
    from sim_ace.core.pedigree_graph import PedigreeGraph
    from sim_ace.phenotyping.gaussian_ace import simulate_gaussian_ace
    from sim_ace.simulation.simulate import run_simulation

    ped = run_simulation(
        seed=2024,
        N=300,
        G_ped=3,
        G_sim=3,
        mating_lambda=0.5,
        p_mztwin=0.0,  # skip twins — collapse happens elsewhere
        A1=0.5,
        C1=0.2,
        A2=0.4,
        C2=0.15,
        rA=0.3,
        rC=0.2,
        assort1=0.0,
        assort2=0.0,
    )
    pg = PedigreeGraph(ped)
    pg.compute_inbreeding()
    K = pg._kinship_matrix

    pheno = simulate_gaussian_ace(ped, var_a=0.5, var_c=0.2, var_e=0.3, seed=1)
    return {
        "y": pheno["y"].to_numpy(),
        "K": K,
        "household_id": ped["household_id"].to_numpy(),
        "n": len(ped),
        "truth": pheno.attrs["truth"],
    }


@needs_bin
class TestFitEndToEnd:
    def test_smoke_ace(self, tiny_ace_inputs):
        inp = tiny_ace_inputs
        r = fit_sparse_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            n_rand_vec=50,
            max_iter=30,
            tol=1e-3,
            threads=2,
            ordering="auto",
            log_level="warn",
        )
        assert r.converged
        vc = r.vc.set_index("vc_name")
        # With Hutchinson + only 300 individuals, generous bounds: truth ± 0.2
        # on each component; sign only matters to ensure it didn't blow up.
        for name, truth in [
            ("V(A)", inp["truth"]["var_a"]),
            ("V(C)", inp["truth"]["var_c"]),
            ("Ve", inp["truth"]["var_e"]),
        ]:
            est = float(vc.loc[name, "estimate"])
            assert est == pytest.approx(truth, abs=0.25), f"{name}: est={est:.4f} truth={truth:.4f}"
        assert 0.0 < float(vc.loc["Vp", "estimate"]) < 2.0
        assert r.cov.shape == (3, 3)
        assert {"iter", "logLik", "V(A)", "V(C)", "Ve"}.issubset(r.iter_log.columns)

    def test_ae_only_no_household(self, tiny_ace_inputs):
        inp = tiny_ace_inputs
        r = fit_sparse_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=None,  # A-only; no C
            n_rand_vec=50,
            max_iter=30,
            tol=1e-3,
            threads=2,
            log_level="warn",
        )
        assert r.converged
        # Without a C GRM, only V(A) and Ve come back.
        names = set(r.vc["vc_name"])
        assert "V(A)" in names
        assert "Ve" in names
        assert "V(C)" not in names

    def test_work_dir_is_kept_when_cleanup_false(self, tiny_ace_inputs, tmp_path):
        inp = tiny_ace_inputs
        work = tmp_path / "work"
        r = fit_sparse_reml(
            y=inp["y"],
            kinship=inp["K"],
            household_id=inp["household_id"],
            n_rand_vec=30,
            max_iter=10,
            tol=1e-2,
            threads=2,
            log_level="warn",
            work_dir=work,
            cleanup=False,
        )
        assert r.converged
        # Intermediate inputs are preserved for inspection.
        assert (work / "inputs" / "A.grm.sp.bin").exists()
        assert (work / "out.vc.tsv").exists()
