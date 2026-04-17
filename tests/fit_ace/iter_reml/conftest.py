"""Shared fixtures for fit_ace.iter_reml tests.

Binary discovery is done here so every integration-level test file can
import ``_FP32``, ``_FP64``, ``needs_bin``, and ``needs_both`` directly
rather than re-resolving paths.

The ``tiny_ace_inputs`` fixture simulates a small 2-to-3-generation
pedigree once per session and exposes both the scipy-sparse and the
pedigree-arrays input shapes so tests can exercise both paths without
re-simulating.
"""

from __future__ import annotations

import pytest

from fit_ace.iter_reml.fit import (
    _DEFAULT_BINARY_FP32,
    _DEFAULT_BINARY_FP64,
)

_FP32 = _DEFAULT_BINARY_FP32
_FP64 = _DEFAULT_BINARY_FP64
_HAS_FP32 = _FP32.exists()
_HAS_FP64 = _FP64.exists()
_HAS_ANY = _HAS_FP32 or _HAS_FP64

needs_bin = pytest.mark.skipif(
    not _HAS_ANY,
    reason=f"no ace_iter_reml build found (looked for {_FP32}, {_FP64})",
)
needs_both = pytest.mark.skipif(
    not (_HAS_FP32 and _HAS_FP64),
    reason="both fp32 and fp64 builds required for parity tests",
)


@pytest.fixture(scope="session")
def tiny_ace_inputs():
    """Small simulated pedigree with known A+C+E truth.

    Uses the simulator + PedigreeGraph to produce a 3-generation pedigree
    (~300 individuals) where the ACE signal is well-identified.  Output
    sizes are small enough that the binary finishes in a few seconds
    even at ``threads=1``.

    Returns a dict with:
        y: (n,) phenotype (float64).
        K: (n, n) scipy sparse kinship matrix.
        household_id: (n,) household labels.
        iids: (n,) string IDs matching pedigree rows.
        n: int.
        truth: {'var_a', 'var_c', 'var_e'} dict from simulate_gaussian_ace.
        pedigree_arrays: (ids, mothers, fathers, twins) for the fast path.
        phen_ids: (n,) same as iids; pedigree == phenotyped here.
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
        p_mztwin=0.0,
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
    iids = ped["id"].astype(str).to_numpy()

    pedigree_arrays = (
        ped["id"].to_numpy(),
        ped["mother"].to_numpy(),
        ped["father"].to_numpy(),
        ped["twin"].to_numpy(),
    )

    return {
        "y": pheno["y"].to_numpy(),
        "K": K,
        "household_id": ped["household_id"].to_numpy(),
        "iids": iids,
        "n": len(ped),
        "truth": pheno.attrs["truth"],
        "pedigree_arrays": pedigree_arrays,
        "phen_ids": ped["id"].to_numpy(),
    }


@pytest.fixture
def fast_kwargs():
    """Wrapper kwargs that keep the binary under ~3s on the tiny fixture.

    Fewer probes than production defaults; single-threaded to dodge the
    PETSc thread-safety race (memory/project_iter_reml_fp32_cleanup_crash).
    """
    return dict(
        phase1_probes=30,
        phase1_blocks=5,
        phase2_probes=30,
        max_iter=30,
        tol=1e-3,
        pcg_tol=1e-5,
        pcg_max_iter=500,
        threads=1,
        log_level="warn",
    )
