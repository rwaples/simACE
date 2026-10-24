"""The inbreeding modes in the simulate loop (ADR 0033 §8, §10): exact F, the Mendelian scale and mode off."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import polars.testing
import pytest

import simace.simulation.simulate as sim
from simace.core.cohort import INBREEDING_KEY, read_inbreeding
from simace.simulation.inbreeding_plan import InbreedingPlan, resolve
from simace.simulation.simulate import run_simulation
from tests.simulation.inbreeding_support import BASE, DENSEP, dense64, force_plan, install_capture, run_engine

WALKD2 = InbreedingPlan("exact", "walkd2")

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


@pytest.mark.parametrize(
    ("model", "N", "G_sim"),
    [("standard", 2000, 12), ("wright_fisher", 2000, 12), ("standard", 300, 30)],  # G_sim > 23: float64 scratch
)
def test_walkd2_matches_densep(monkeypatch, model, N, G_sim) -> None:
    cap = install_capture(monkeypatch)
    run_simulation(**{**BASE, "N": N, "G_ped": 2, "G_sim": G_sim, "mating_model": model})
    walk = run_engine(WALKD2, cap.gens, mother_sorted=model == "standard")
    dense = run_engine(DENSEP, cap.gens, mother_sorted=model == "standard")
    assert walk[-1].mean() > 1e-3
    np.testing.assert_allclose(walk, dense, rtol=0, atol=1e-6)


def test_exact_through_the_resolver_matches_pedigree_graph(monkeypatch) -> None:
    cap = install_capture(monkeypatch)
    kw = {**BASE, "inbreeding_mode": "exact"}
    assert resolve("exact", 6, kw["N"], kw["G_sim"]).engine == "walkd2"
    ped = run_simulation(**kw)
    N = kw["N"]
    oracle = cap.ancestry(ped["sex"].to_numpy()[-N:]).inbreeding()[N:].reshape(kw["G_sim"], N)
    np.testing.assert_allclose(ped["F"].to_numpy(), _recorded(kw, oracle), rtol=0, atol=1e-6)


# ---------------------------------------------------------------- approx (xexact)


@pytest.fixture(scope="module")
def deep_capture():
    """A 2k/G 20 random-mating ancestry per mating model and its exact F (densep, float32)."""
    mp = pytest.MonkeyPatch()
    out = {}
    for model in ("standard", "wright_fisher"):
        cap = install_capture(mp)
        pmz = 0.004 if model == "standard" else 0.0
        run_simulation(**{**BASE, "N": 2000, "G_ped": 2, "G_sim": 20, "mating_model": model, "p_mztwin": pmz})
        mp.undo()
        out[model] = (cap.gens, run_engine(DENSEP, cap.gens, mother_sorted=model == "standard"))
    return out


def _xexact(depth: int, packed: bool = False) -> InbreedingPlan:
    return InbreedingPlan("approx", "xexact", depth, packed)


@pytest.mark.parametrize("depth", [3, 6])
def test_xexact_meets_the_rms_law_at_random_mating(deep_capture, depth) -> None:
    # The statistic of the evidence (notes/issue-76-recommendation/summ/b2.txt): last-generation RMS error over
    # 2^-L / (2 sqrt(N)); about 0.8 at N >= 2k and L <= 6. Forced past the small-N handoff.
    gens, exact = deep_capture["standard"]
    F = run_engine(_xexact(depth), gens, mother_sorted=True)
    ratio = np.sqrt(((F[-1] - exact[-1]) ** 2).mean()) / (2.0**-depth / (2 * np.sqrt(2000)))
    assert 0.5 <= ratio <= 1.5


@pytest.mark.parametrize("model", ["standard", "wright_fisher"])
@pytest.mark.parametrize("depth", [3, 6])
def test_packed_and_unpacked_tails_give_bitwise_equal_F(deep_capture, model, depth) -> None:
    gens, _ = deep_capture[model]
    sorted_ = model == "standard"
    packed = run_engine(_xexact(depth, packed=True), gens, mother_sorted=sorted_)  # raises on a multiplicity overflow
    np.testing.assert_array_equal(packed, run_engine(_xexact(depth), gens, mother_sorted=sorted_))


def test_serial_path_matches_the_parallel_path_on_a_mother_sorted_pedigree(deep_capture) -> None:
    # The Wright-Fisher path (serial family grouping and descending pass) sums the boundary's 1'A 1 in another
    # order, so F agrees to rounding (5.6e-17 measured), not bitwise.
    gens, _ = deep_capture["standard"]
    parallel = run_engine(_xexact(6), gens, mother_sorted=True)
    np.testing.assert_allclose(run_engine(_xexact(6), gens, mother_sorted=False), parallel, rtol=0, atol=1e-15)


def test_xexact_needs_mother_order_on_the_parallel_path(deep_capture) -> None:
    gens, _ = deep_capture["wright_fisher"]
    with pytest.raises(ValueError, match="sorted by mother"):
        run_engine(_xexact(6), gens, mother_sorted=True)


def test_xexact_at_full_depth_is_exact(deep_capture) -> None:
    gens, _ = deep_capture["standard"]
    exact = run_engine(WALKD2, gens, mother_sorted=True)
    np.testing.assert_allclose(run_engine(_xexact(20), gens, mother_sorted=True), exact, rtol=0, atol=1e-12)


def test_approx_with_depth_at_g_sim_runs_the_exact_mode(monkeypatch) -> None:
    kw = {**BASE, "N": 1000, "G_sim": 30}
    assert resolve("approx", 30, 1000, 30) == resolve("exact", 6, 1000, 30)
    pl.testing.assert_frame_equal(
        run_simulation(**kw, inbreeding_mode="approx", inbreeding_depth=30),
        run_simulation(**kw, inbreeding_mode="exact"),
    )


def test_approx_runs_xexact_above_the_handoff(monkeypatch) -> None:
    cap = install_capture(monkeypatch)
    kw = {**BASE, "N": 12_000, "G_ped": 2, "G_sim": 5, "p_mztwin": 0.004}
    assert resolve("approx", 3, 12_000, 5) == _xexact(3, packed=True)
    ped = run_simulation(**kw, inbreeding_mode="approx", inbreeding_depth=3)
    F = run_engine(_xexact(3, packed=True), cap.gens, mother_sorted=True)
    np.testing.assert_array_equal(ped["F"].to_numpy(), np.float32(F[-2:].ravel()))
    exact = run_engine(WALKD2, cap.gens, mother_sorted=True)
    assert np.sqrt(((F[-1] - exact[-1]) ** 2).mean()) / (2.0**-3 / (2 * np.sqrt(12_000))) <= 1.5


def test_small_n_handoff_runs_the_cost_models_engine() -> None:
    # At N 1k the 8-thread fits cross between G 12 and G 16 (summ/fit_exact2.txt); a refit may flip these.
    assert resolve("approx", 6, 1000, 12) == WALKD2
    assert resolve("approx", 6, 1000, 16) == DENSEP


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
from simace.simulation.inbreeding_plan import InbreedingPlan, resolve
from tests.simulation.inbreeding_support import BASE, install_capture, run_engine
import pytest
mp = pytest.MonkeyPatch()
cap = install_capture(mp)
from simace.simulation.simulate import run_simulation
run_simulation(**{{**BASE, **{kw!r}}})
F = run_engine(InbreedingPlan(*{plan!r}), cap.gens, mother_sorted={sorted!r})
print(json.dumps({{"threads": numba.get_num_threads(), "sha": hashlib.sha256(F.tobytes()).hexdigest()}}))
"""


def _engine_digest(threads: int, plan: InbreedingPlan, **kw) -> dict:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {**os.environ, "NUMBA_NUM_THREADS": str(threads)}
    sorted_ = kw.get("mating_model", "standard") == "standard"
    script = _THREAD_SCRIPT.format(root=root, kw=kw, plan=tuple(plan.record().values()), sorted=sorted_)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


_THREAD_CASES = {
    "densep": (DENSEP, {}),
    "walkd2": (WALKD2, {"N": 2000, "G_sim": 12}),
    "walkd2_wf": (WALKD2, {"N": 2000, "G_sim": 12, "mating_model": "wright_fisher", "p_mztwin": 0.0}),
    "xexact": (InbreedingPlan("approx", "xexact", 4, True), {"N": 2000, "G_sim": 12}),
    "xexact_wf": (
        InbreedingPlan("approx", "xexact", 4, False),
        {"N": 2000, "G_sim": 12, "mating_model": "wright_fisher", "p_mztwin": 0.0},
    ),
}


@pytest.mark.parametrize("case", list(_THREAD_CASES))
def test_engine_is_bitwise_thread_invariant(case) -> None:
    plan, kw = _THREAD_CASES[case]
    one, four = _engine_digest(1, plan, **kw), _engine_digest(4, plan, **kw)
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


def test_approx_handoff_records_the_mode_that_ran(tmp_path) -> None:
    path = _cli(tmp_path, "--inbreeding-mode", "approx", "--inbreeding-depth", "4")
    assert read_inbreeding(path) == {"mode": "exact", "engine": "densep", "depth": None, "packed": False}
    F = pl.read_parquet(path)["F"]
    assert F.dtype == pl.Float32
    assert F.max() > 0
