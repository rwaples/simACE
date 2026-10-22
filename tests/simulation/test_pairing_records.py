"""Single-trait Metropolis refinement and the per-iteration pairing records (ADR 0032; issues #73, #74)."""

import numpy as np
import pytest

import simace.simulation.simulate as simulate_mod
from simace.simulation.assortment import METROPOLIS_TOL, read_mating_log
from simace.simulation.simulate import (
    _assortative_pair_partners,
    _find_duplicate_pairs,
    _metropolis_single,
    balance_mating_slots,
    draw_mating_counts,
    run_simulation,
)

_SIM = dict(
    N=1000,
    G_ped=3,
    G_sim=5,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.5,
    C1=0.2,
    A2=0.4,
    C2=0.2,
    rA=0.3,
    rC=0.0,
    E1=0.3,
    E2=0.4,
)


def _population(rng, n, lam):
    sex = rng.integers(0, 2, n)
    pheno = np.zeros((n, 6))
    pheno[:, 0] = rng.standard_normal(n)
    pheno[:, 3] = rng.standard_normal(n)
    mi, fi = np.where(sex == 1)[0], np.where(sex == 0)[0]
    mc, fc = balance_mating_slots(rng, draw_mating_counts(rng, len(mi), lam), draw_mating_counts(rng, len(fi), lam))
    return mi, mc, fi, fc, pheno


def _slot_r(pheno, matings, col):
    x, y = pheno[matings[:, 0], col], pheno[matings[:, 1], col]
    return float(np.mean((x - x.mean()) / x.std() * (y - y.mean()) / y.std()))


@pytest.mark.parametrize("assort", [-0.8, -0.2, 0.2, 0.8])
@pytest.mark.parametrize(("n", "lam"), [(150, 3.0), (2000, 0.5), (2000, 3.0)])
@pytest.mark.parametrize("trait", [1, 2])
def test_single_trait_swaps_pin_the_slot_correlation(assort, n, lam, trait):
    rng = np.random.default_rng(abs(hash((assort, n, lam, trait))) % 2**32)
    mi, mc, fi, fc, pheno = _population(rng, n, lam)
    log: list = []
    a1, a2 = (assort, 0.0) if trait == 1 else (0.0, assort)
    matings = _assortative_pair_partners(rng, mi, mc, fi, fc, pheno, a1, a2, pairing_log=log)
    (record,) = log
    assert record["mode"] == "single"
    assert record["converged"]
    assert set(record["errors"]) == {f"r{trait}"}
    realized = _slot_r(pheno, matings, 0 if trait == 1 else 3)
    assert abs(realized - assort) < METROPOLIS_TOL
    assert record["errors"][f"r{trait}"] == pytest.approx(realized - assort, abs=1e-12)
    assert record["duplicates_left"] == int(_find_duplicate_pairs(matings).sum())


def test_the_single_trait_kernel_never_creates_a_duplicate_couple():
    rng = np.random.default_rng(0)
    # 300 mothers with 3 slots each, 900 distinct fathers: no duplicate to start with.
    mothers = np.repeat(np.arange(300), 3)
    fathers = np.arange(1000, 1900, dtype=np.int64)
    rng.shuffle(fathers)
    fz = rng.standard_normal(300)[mothers]
    fz = (fz - fz.mean()) / fz.std()
    mz = rng.standard_normal(900)
    mz = (mz - mz.mean()) / mz.std()
    rows = np.argsort(mothers, kind="stable")
    rows_ptr = np.arange(0, 901, 3, dtype=np.int64)
    S, proposals = _metropolis_single(
        fz, mz, fathers, mothers, rows_ptr, rows, float(fz @ mz), 0.9 * 900, 900, 1e-9, 8 * 900, 7
    )
    assert proposals == 8 * 900  # the tolerance is unreachable, so it runs to the cap
    assert not _find_duplicate_pairs(np.column_stack([mothers, fathers])).any()
    assert float(fz @ mz) == pytest.approx(S, abs=1e-6)  # S tracks the swapped fathers' values


def test_an_unreachable_tolerance_is_recorded_as_not_converged(monkeypatch):
    rng = np.random.default_rng(1)
    mi, mc, fi, fc, pheno = _population(rng, 1000, 0.5)
    monkeypatch.setattr(simulate_mod, "METROPOLIS_TOL", 0.0)
    log: list = []
    _assortative_pair_partners(rng, mi, mc, fi, fc, pheno, 0.4, 0.0, pairing_log=log)
    assert not log[0]["converged"]
    assert log[0]["proposals"] == log[0]["max_proposals"] == 8 * log[0]["M"]


@pytest.mark.parametrize(
    ("overrides", "modes"),
    [
        ({}, {"random"}),
        ({"assort1": 0.3}, {"single"}),
        ({"assort1": 0.3, "assort2": 0.3}, {"both"}),
        ({"assort1": {0: 0.0, 3: 0.3}}, {"random", "single"}),
        ({"mating_model": "wright_fisher"}, {"wright_fisher"}),
    ],
)
def test_run_simulation_records_one_pairing_per_iteration(overrides, modes):
    log: list = []
    run_simulation(seed=3, **{**_SIM, **overrides}, mating_log=log)
    assert [r["iteration"] for r in log] == list(range(_SIM["G_sim"]))
    burnin = _SIM["G_sim"] - _SIM["G_ped"]
    assert [r["generation"] for r in log] == [i - burnin for i in range(_SIM["G_sim"])]
    assert {r["mode"] for r in log} == modes
    for r in log:
        if r["mode"] in ("single", "both"):
            assert r["converged"]


def test_the_cli_writes_the_pairing_records(tmp_path):
    from simace.simulation.simulate import cli as simulate_cli

    args = [
        "--seed",
        "3",
        "--N",
        "300",
        "--G-ped",
        "2",
        "--G-sim",
        "3",
        "--E1",
        "0.3",
        "--E2",
        "0.4",
        "--assort1",
        "0.3",
        "--output-pedigree",
        str(tmp_path / "pedigree.parquet"),
        "--output-mating",
        str(tmp_path / "mating.yaml"),
    ]
    simulate_cli(args)
    log = read_mating_log(tmp_path / "mating.yaml")
    assert [(r["iteration"], r["generation"], r["mode"]) for r in log] == [
        (0, -1, "single"),
        (1, 0, "single"),
        (2, 1, "single"),
    ]
    assert all(r["converged"] and set(r["errors"]) == {"r1"} for r in log)
