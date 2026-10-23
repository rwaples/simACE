"""Golden characterization of ``run_simulation`` output.

Hashes the full pedigree DataFrame for a fixed seed across the mating-model and
assortative-mating branches. This is a *characterization* test: it pins the
exact bytes the current code produces so that behavior-preserving refactors
(e.g. extracting ``SimulationParams`` / ``AssortmentPlan``) are provably
non-mutating. A change to any digest means simulation output moved — verify
that is intended, then regenerate the constant below.

Coverage by config:
  * ``wf``                 — Wright-Fisher path (``_mating_wf``).
  * ``standard_no_am``     — standard mating, no assortment.
  * ``standard_single_am`` — single-trait Gaussian-copula assortment, then
                             the 1-D Metropolis refinement (ADR 0032).
  * ``standard_both_am``   — both-trait 4-variate Gaussian copula + Metropolis + per-gen
                             ``R_mf`` (the most complex standard-only path).
  * ``standard_pergen``    — per-generation assort1 and E1 dicts, locking the
                             per-generation indexing (``assort*_per_gen[i]`` and
                             the ``parent_ce_gen = max(0, i-1)`` rho_w lookup).

Portability note: ``wf`` and ``standard_no_am`` are fully portable — they
draw exclusively from the seeded ``numpy.random.Generator``. The assortative
configs additionally route through a numba Metropolis kernel
(``_metropolis_full`` or ``_metropolis_single``, ``fastmath=True``) and the
legacy global ``np.random`` stream, so their digests can differ across numba
versions / CPUs; regenerate them if they drift on a new platform.
"""

import hashlib

import numpy as np
import pytest

from simace.simulation.simulate import run_simulation

_BASE = dict(
    seed=12345,
    N=200,
    G_ped=2,
    G_sim=3,
    mating_lambda=0.8,
    p_mztwin=0.05,
    A1=0.5,
    C1=0.2,
    E1=0.3,
    A2=0.4,
    C2=0.1,
    E2=0.5,
    rA=0.3,
    rC=0.2,
    rE=0.0,
)

_CONFIGS = {
    "wf": {**_BASE, "mating_model": "wright_fisher"},
    "standard_no_am": {**_BASE, "mating_model": "standard", "assort1": 0.0, "assort2": 0.0},
    "standard_single_am": {**_BASE, "mating_model": "standard", "assort1": 0.4, "assort2": 0.0},
    "standard_both_am": {**_BASE, "mating_model": "standard", "assort1": 0.4, "assort2": 0.3},
    "standard_pergen": {
        **_BASE,
        "mating_model": "standard",
        "assort1": {0: 0.0, 2: 0.6},
        "assort2": 0.0,
        "E1": {0: 0.3, 1: 0.6, 2: 0.9},
    },
}

# Captured from the current code (see module docstring on regeneration).
_GOLDEN = {
    "wf": "c110f24df422ef867cda7a9316ba36df93119ac8a96bb19c06af84242e9fd0a0",
    "standard_no_am": "694cd183c0068be1dca268a568915188c700043f26fcdec37d4474a31b8ea081",
    "standard_single_am": "7c690ac217607921717c232478cb351e54954569a84b94601a37a606747c0533",
    "standard_both_am": "1e874c8af0e95b24835af9ee154a961c6f475302713974a5a2db69529cfb3594",
    "standard_pergen": "48c3c56d744de4e44b636f5d976b0576103a565cbb1008afef3bb3a60bbe326f",
}


def _digest(df) -> str:
    """SHA-256 over each column's name and raw little-endian bytes."""
    h = hashlib.sha256()
    for col in df.columns:
        h.update(col.encode())
        h.update(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()


_ZERO_C_TRANSMISSION = {f"Ctm_{side}{t}": 0.0 for side in ("maternal", "paternal") for t in (1, 2)}


@pytest.mark.parametrize("config_name", ["wf", "standard_no_am", "standard_single_am"])
def test_zero_c_transmission_reproduces_the_golden_output(config_name):
    df = run_simulation(**_CONFIGS[config_name], **_ZERO_C_TRANSMISSION)
    assert _digest(df) == _GOLDEN[config_name]


@pytest.mark.parametrize("config_name", list(_CONFIGS))
def test_run_simulation_golden(config_name):
    df = run_simulation(**_CONFIGS[config_name])
    assert _digest(df) == _GOLDEN[config_name], (
        f"{config_name}: simulation output changed. If intended, regenerate the "
        f"_GOLDEN digest; otherwise the refactor altered behavior."
    )
