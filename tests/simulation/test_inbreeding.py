"""The inbreeding modes in the simulate loop (ADR 0033 §8, §10): exact F, the Mendelian scale and mode off."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import simace.simulation.simulate as sim
from simace.core.cohort import INBREEDING_KEY, read_inbreeding
from simace.simulation.simulate import run_simulation
from tests.simulation.inbreeding_support import BASE, DENSEP, dense64, force_plan, install_capture, run_engine

_CASES = {
    "standard_burnin": {},
    "standard_no_burnin": {"G_ped": 6, "G_sim": 6},
    "wf_burnin": {"mating_model": "wright_fisher", "p_mztwin": 0.0},
}


@pytest.fixture(params=list(_CASES))
def exact_run(request, monkeypatch):
    """A densep run of one case: its pedigree and the full captured ancestry."""
    force_plan(monkeypatch, DENSEP)
    cap = install_capture(monkeypatch)
    kw = {**BASE, **_CASES[request.param], "inbreeding_mode": "exact"}
    ped = run_simulation(**kw)
    return kw, ped, cap


def _recorded(kw, values: np.ndarray) -> np.ndarray:
    """The rows of the last ``G_ped`` iterations of a ``(G_sim, N)`` per-iteration array, as the pedigree orders them."""
    return values[kw["G_sim"] - kw["G_ped"] :].ravel()


def test_F_column_matches_pedigree_graph(exact_run) -> None:
    kw, ped, cap = exact_run
    N = kw["N"]
    graph = cap.ancestry(ped["sex"].to_numpy()[-N:])
    oracle = graph.inbreeding()[N:].reshape(kw["G_sim"], N)
    F = ped["F"].to_numpy()
    assert ped["F"].dtype == pl.Float32
    assert _recorded(kw, oracle).max() > 0.01  # the case is inbred enough to test
    np.testing.assert_allclose(F, _recorded(kw, oracle), rtol=0, atol=1e-6)


def test_densep_matches_dense_float64_in_every_iteration(exact_run) -> None:
    kw, ped, cap = exact_run
    F = run_engine(DENSEP, cap.gens[: kw["G_sim"]], mother_sorted=True)
    np.testing.assert_allclose(F, dense64(cap.gens[: kw["G_sim"]]), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.float32(_recorded(kw, F)), ped["F"].to_numpy())


def test_twins_share_F_and_A(exact_run) -> None:
    _kw, ped, _cap = exact_run
    twins = ped.filter(pl.col("twin") >= 0)
    if twins.is_empty():
        pytest.skip("no MZ twins in this case")
    pairs = twins.join(ped, left_on="twin", right_on="id", suffix="_co")
    assert len(pairs) == len(twins)
    for col in ("F", "A1", "A2"):
        np.testing.assert_array_equal(pairs[col].to_numpy(), pairs[f"{col}_co"].to_numpy())


# ---------------------------------------------------------------- scale indexing (Fable finding 6)


def _hand_mating(rng, sex, *_args, state, **_kwargs):
    """Deterministic pedigrees over 4 iterations: monogamous couples, then 10 sibships, full-sib matings, any."""
    i = state["i"]
    state["i"] += 1
    N = len(sex)
    females, males = np.flatnonzero(sex == 0), np.flatnonzero(sex == 1)
    if i in (0, 3):  # monogamous founder couples, or any pairing: F of the parents is all that matters
        couple = np.arange(N) % min(len(females), len(males))
        mothers, fathers = females[couple], males[couple]
    elif i == 1:  # 10 couples of children of distinct founder couples, so F = 0; 20 children each
        couple_of = state["parents"][:, 0]  # founder couples are monogamous: the mother names the couple
        f10 = [x for k, x in enumerate(females) if couple_of[x] not in couple_of[females[:k]]][:10]
        m10 = [y for y in males if couple_of[y] not in couple_of[f10]]
        m10 = [y for k, y in enumerate(m10) if couple_of[y] not in couple_of[m10[:k]]][:10]
        assert len(f10) == 10
        assert len(m10) == 10
        couples = np.repeat(np.arange(10), N // 10)
        mothers, fathers = np.array(f10)[couples], np.array(m10)[couples]
    else:  # i == 2: every child's parents are full sibs (one sibship per block of 20)
        mothers = np.empty(N, np.int64)
        fathers = np.empty(N, np.int64)
        for b in range(10):
            block = np.arange(b * (N // 10), (b + 1) * (N // 10))
            fem, mal = block[sex[block] == 0], block[sex[block] == 1]
            assert len(fem), "a sibship of one sex; pick another seed"
            assert len(mal), "a sibship of one sex; pick another seed"
            mothers[block] = fem[0]
            fathers[block] = mal[0]
    order = np.argsort(mothers, kind="stable")
    parents = np.column_stack([mothers[order], fathers[order]])
    state["parents"] = parents
    household = np.unique(parents[:, 0], return_inverse=True)[1]
    return parents, np.zeros((0, 2), np.int64), household


def test_scale_uses_the_parents_F_not_the_offspring_F(monkeypatch) -> None:
    force_plan(monkeypatch, DENSEP)
    state = {"i": 0}
    monkeypatch.setattr(sim, "_mating_standard", lambda *a, **k: _hand_mating(*a, state=state, **k))
    scales = []
    noise = sim.generate_mendelian_noise

    def spy(rng, n, sd1, sd2, rA, scale=None):
        scales.append(scale)
        return noise(rng, n, sd1, sd2, rA, scale)

    monkeypatch.setattr(sim, "generate_mendelian_noise", spy)
    kw = {**BASE, "N": 200, "G_ped": 4, "G_sim": 4, "p_mztwin": 0.0, "inbreeding_mode": "exact"}
    ped = run_simulation(**kw)
    F = ped["F"].to_numpy().reshape(4, 200)
    np.testing.assert_array_equal(F[:2], 0.0)
    np.testing.assert_allclose(F[2], 0.25, rtol=0, atol=1e-7)  # full-sib matings
    np.testing.assert_array_equal(scales[2], 1.0)  # their parents are not inbred
    np.testing.assert_allclose(scales[3], np.sqrt(0.75), rtol=1e-12)  # their children's parents have F = 0.25

    # Both traits carry the scale: same seed and pedigree, mode off draws the same noise unscaled.
    state["i"] = 0
    off = run_simulation(**{**kw, "inbreeding_mode": "off"})
    mp = {}
    for run, df in (("on", ped), ("off", off)):
        for t in ("A1", "A2"):
            a = df[t].to_numpy().astype(np.float64).reshape(4, 200)
            m = df["mother"].to_numpy().reshape(4, 200)[3] - 400
            f = df["father"].to_numpy().reshape(4, 200)[3] - 400
            mp[run, t] = a[3] - (a[2][m] + a[2][f]) / 2
    for t in ("A1", "A2"):
        np.testing.assert_allclose(mp["on", t], mp["off", t] * np.sqrt(0.75), rtol=0, atol=1e-5)


def test_random_mating_leaves_the_random_stream_and_pedigree_unchanged(monkeypatch) -> None:
    force_plan(monkeypatch, DENSEP)
    rngs = []
    default_rng = np.random.default_rng

    def recording(*a, **k):
        rngs.append(default_rng(*a, **k))
        return rngs[-1]

    monkeypatch.setattr(sim.np.random, "default_rng", recording)
    off = run_simulation(**BASE)
    on = run_simulation(**BASE, inbreeding_mode="exact")
    assert len(rngs) == 2
    assert rngs[0].bit_generator.state == rngs[1].bit_generator.state
    same = ["id", "sex", "mother", "father", "twin", "generation", "household_id", "C1", "E1", "C2", "E2"]
    assert on.select(same).equals(off.select(same))
    assert not on["A1"].equals(off["A1"])
    assert on.columns == [*off.columns[:7], "F", *off.columns[7:]]


# ---------------------------------------------------------------- threads, mode off, metadata

_THREAD_SCRIPT = """
import hashlib, json, sys
import numba
sys.path.insert(0, {root!r})
from tests.simulation.inbreeding_support import BASE, DENSEP, install_capture, run_engine
import pytest
mp = pytest.MonkeyPatch()
cap = install_capture(mp)
from simace.simulation.simulate import run_simulation
run_simulation(**BASE)
F = run_engine(DENSEP, cap.gens, mother_sorted=True)
print(json.dumps({{"threads": numba.get_num_threads(), "sha": hashlib.sha256(F.tobytes()).hexdigest()}}))
"""


def _engine_digest(threads: int, script: str) -> dict:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {**os.environ, "NUMBA_NUM_THREADS": str(threads)}
    out = subprocess.run(
        [sys.executable, "-c", script.format(root=root)], capture_output=True, text=True, check=True, env=env
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_densep_is_bitwise_thread_invariant() -> None:
    one, four = _engine_digest(1, _THREAD_SCRIPT), _engine_digest(4, _THREAD_SCRIPT)
    assert (one["threads"], four["threads"]) == (1, 4)
    assert one["sha"] == four["sha"]


def _cli(tmp_path, *extra: str) -> str:
    out = tmp_path / "pedigree.parquet"
    sim.cli(
        [
            "--N",
            "1000",
            "--G-ped",
            "2",
            "--G-sim",
            "30",
            "--E1",
            "0.3",
            "--E2",
            "0.3",
            "--output-pedigree",
            str(out),
            *extra,
        ]
    )
    return str(out)


def test_mode_off_writes_no_inbreeding_metadata_or_column(tmp_path) -> None:
    path = _cli(tmp_path)
    assert INBREEDING_KEY not in pl.read_parquet_metadata(path)
    assert read_inbreeding(path) is None
    assert "F" not in pl.read_parquet(path).columns


def test_exact_mode_records_the_plan_in_the_pedigree(tmp_path) -> None:
    path = _cli(tmp_path, "--inbreeding-mode", "exact")
    assert read_inbreeding(path) == {"mode": "exact", "engine": "densep", "depth": None, "packed": False}
    F = pl.read_parquet(path)["F"]
    assert F.dtype == pl.Float32
    assert F.max() > 0
