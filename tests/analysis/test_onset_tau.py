"""Onset τ and its realized expectation (ADR 0027)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import polars as pl
import pytest
from pedigree_graph import PedigreeGraph

from simace.analysis.stats import onset_tau as ot
from simace.phenotype.hazards import standardize_beta
from simace.phenotype.runner import run_phenotype
from simace.simulation.simulate import run_simulation

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ordering_expectation(lam: dict[int, float], x: int, xf: int, y: int, yf: int) -> float:
    """E[sign(V_x − V_xf)·sign(V_y − V_yf)] for independent exponentials, by enumerating every ordering."""
    people = sorted(lam)
    total = 0.0
    for order in itertools.permutations(people):
        prob, rest = 1.0, sum(lam.values())
        for p in order:
            prob *= lam[p] / rest
            rest -= lam[p]
        pos = {p: i for i, p in enumerate(order)}
        total += np.sign(pos[x] - pos[xf]) * np.sign(pos[y] - pos[yf]) * prob
    return total


def _brute_target(eta_x: np.ndarray, eta_y: np.ndarray, first, second, symmetric: bool, same_trait: bool) -> float:
    """Mean exact expectation over every comparison the statistic counts, pair of entries by pair of entries."""
    X, Y, mirror, _ = ot._cell_entries(np.asarray(first), np.asarray(second), symmetric)
    total, count = 0.0, 0
    for e, f in itertools.combinations(range(len(X)), 2):
        if mirror[e] == f:
            continue
        x, y, xf, yf = X[e], Y[e], X[f], Y[f]
        if same_trait:
            # One onset per person: label people, rates from one η.
            lam = {p: math.exp(eta_x[p]) for p in {x, y, xf, yf}}
            total += _ordering_expectation(lam, x, xf, y, yf)
        else:
            # Two traits: side-x onsets and side-y onsets are independent.
            total += math.tanh((eta_x[x] - eta_x[xf]) / 2) * math.tanh((eta_y[y] - eta_y[yf]) / 2)
        count += 1
    return total / count


def _ranks(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_concordance_kernel``'s inputs: x's sort order and both sides' dense ranks."""
    order, rx = ot._dense_ranks(x)
    return order, rx, ot._dense_ranks(y)[1]


def _sibships(sizes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete sibships of the given sizes: first and second members, and each person's sibship."""
    first, second, family = [], [], []
    start = 0
    for f, size in enumerate(sizes):
        for i, j in itertools.combinations(range(start, start + size), 2):
            first.append(i)
            second.append(j)
        family.extend([f] * size)
        start += size
    return np.array(first, dtype=np.int64), np.array(second, dtype=np.int64), np.array(family, dtype=np.int64)


# ---------------------------------------------------------------------------
# tau_oracle
# ---------------------------------------------------------------------------


class TestTauOracle:
    def test_zero_effect_is_zero(self):
        assert ot.tau_oracle(0.0, 1.0, 0.5) == 0.0

    def test_uncorrelated_is_zero(self):
        assert abs(ot.tau_oracle(1.5, 0.7, 0.0)) < 1e-12

    def test_sign_flips_with_r(self):
        assert ot.tau_oracle(1.2, 0.8, -0.4) == pytest.approx(-ot.tau_oracle(1.2, 0.8, 0.4), abs=1e-12)

    def test_symmetric_in_the_two_sides(self):
        assert ot.tau_oracle(0.6, 2.0, 0.3) == pytest.approx(ot.tau_oracle(2.0, 0.6, 0.3), abs=1e-10)

    @pytest.mark.parametrize("r", [0.25, 0.5, 0.9])
    def test_large_effect_reaches_the_gaussian_copula_limit(self, r):
        assert ot.tau_oracle(500.0, 500.0, r) == pytest.approx(2 / math.pi * math.asin(r), abs=3e-3)

    @pytest.mark.parametrize("s", [1.0, 5.0])
    def test_perfect_correlation_is_continuous(self, s):
        # r = 1: E[tanh(s X / 2)^2] with X ~ N(0, 2).
        assert ot.tau_oracle(s, s, 1.0) == pytest.approx(ot.tau_oracle(s, s, 1.0 - 1e-9), abs=1e-6)

    @pytest.mark.parametrize("s", [0.5, 1.5, 2.9])
    def test_hermite_rule_matches_adaptive_quadrature(self, s, monkeypatch):
        fast = ot.tau_oracle(s, 0.8 * s, 0.35)
        monkeypatch.setattr(ot, "_HERMITE_MAX_SCALE", 0.0)
        assert fast == pytest.approx(ot.tau_oracle(s, 0.8 * s, 0.35), abs=1e-8)

    @pytest.mark.parametrize(("s1", "s2", "r"), [(1.0, 1.0, 0.5), (1.5, 1.5, 0.25), (2.0, 0.7, 0.4)])
    def test_matches_monte_carlo_of_independent_pairs(self, s1, s2, r):
        rng = np.random.default_rng(1)
        n = 2_000_000
        zi, zk = rng.standard_normal(n), rng.standard_normal(n)
        zj = r * zi + math.sqrt(1 - r * r) * rng.standard_normal(n)
        zl = r * zk + math.sqrt(1 - r * r) * rng.standard_normal(n)
        g = np.log(rng.exponential(size=(4, n)))
        wi, wk = g[0] - s1 * zi, g[1] - s1 * zk
        wj, wl = g[2] - s2 * zj, g[3] - s2 * zl
        concord = np.sign(wi - wk) * np.sign(wj - wl)
        se = concord.std() / math.sqrt(n)
        assert abs(concord.mean() - ot.tau_oracle(s1, s2, r)) < 4 * se


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def test_concordance_rows_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 8, 300).astype(np.float64)
    y = rng.integers(0, 8, 300).astype(np.float64)
    brute = (np.sign(x[:, None] - x[None, :]) * np.sign(y[:, None] - y[None, :])).sum(axis=1)
    assert np.array_equal(ot._concordance_kernel(*_ranks(x, y)), brute)


def test_middle_matches_enumeration_and_monte_carlo():
    rng = np.random.default_rng(2)
    lam = np.exp(rng.normal(size=3))
    exact = ot._middle(lam[0], lam[1], lam[2])
    assert exact == pytest.approx(_ordering_expectation(dict(enumerate(lam)), 0, 1, 2, 0), abs=1e-14)
    n = 2_000_000
    v = rng.exponential(size=(3, n)) / lam[:, None]
    mc = np.sign(v[0] - v[1]) * np.sign(v[2] - v[0])
    assert abs(mc.mean() - exact) < 4 * mc.std() / math.sqrt(n)


def test_shared_person_comparison_with_equal_rates_is_minus_one_third():
    assert ot._middle(1.0, 1.0, 1.0) == pytest.approx(-1 / 3, abs=1e-15)


def test_taylor_bound_constants():
    t = np.linspace(-40, 40, 2_000_001)
    th = np.tanh(t / 2)
    k1 = 0.5 * (1 - th**2)
    k2 = -0.5 * th * (1 - th**2)
    k3 = 0.25 * (1 - th**2) * (3 * th**2 - 1)
    assert np.abs(k1).max() == pytest.approx(ot._M1, rel=1e-9)
    assert np.abs(k2).max() == pytest.approx(ot._M2, rel=1e-6)
    assert np.abs(k3).max() == pytest.approx(ot._M3, rel=1e-9)


@pytest.mark.parametrize(("scale", "grid"), [(1.0, 64), (1.0, 256), (2.5, 128), (4.0, 512)])
def test_taylor_rowsum_within_its_bound_at_every_entry(scale, grid):
    rng = np.random.default_rng(3)
    shared = rng.normal(size=1500)
    u = scale * (0.6 * shared + 0.8 * rng.normal(size=1500))
    v = scale * (0.6 * shared + 0.8 * rng.normal(size=1500)) + 0.5 * (rng.random(1500) < 0.5)
    exact = ot._exact_rowsum(u, v)
    approx, per_term = ot._taylor_rowsum(u, v, grid)
    worst = np.abs(approx - exact).max() / len(u)
    assert worst <= per_term
    assert worst <= per_term / 5  # the bound is a worst case; in practice it is loose


def test_tied_comparisons_counts_both_clips_and_ignores_the_same_person():
    values = np.array([1e-10, 1e-10, 1e6, 1e6, 1e6, 3.0, 3.0])
    persons = np.array([0, 1, 2, 3, 4, 5, 5])  # the two 3.0 entries are one person
    assert ot._tied_comparisons(ot._dense_ranks(values)[1], persons) == 1 + 3


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


def _cell(eta, first, second, onsets, *, symmetric=True, component=None):
    if component is None:
        component = ot._component_labels(first, second, len(eta))
    return ot.onset_tau_cell(onsets, onsets, eta, eta, first, second, component, symmetric=symmetric)


class TestCompleteSibship:
    """Fifteen full sibs: τ̂ is −1/24 for every ordering, so the target is too, and the SE is 0."""

    first, second, _ = _sibships([15])

    @pytest.mark.parametrize("seed", [0, 1])
    def test_target_and_observed_are_minus_one_twenty_fourth(self, seed):
        rng = np.random.default_rng(seed)
        eta = np.zeros(15) if seed == 0 else rng.normal(size=15)
        onsets = np.log(rng.exponential(size=15)) - eta
        cell = _cell(eta, self.first, self.second, onsets)
        assert cell["observed"] == pytest.approx(-1 / 24, abs=1e-14)
        assert cell["expected"] == pytest.approx(-1 / 24, abs=1e-12)
        assert cell["se"] <= 1e-12
        assert cell["n_components"] == 1


def test_all_tied_onsets_give_zero():
    first, second = np.array([0, 2]), np.array([1, 3])
    cell = _cell(np.zeros(4), first, second, np.ones(4))
    assert cell["observed"] == 0.0
    assert cell["tie_share"] > 0


@pytest.mark.parametrize("symmetric", [True, False])
def test_target_matches_brute_force_enumeration(symmetric):
    rng = np.random.default_rng(4)
    n_people = 30
    first = rng.integers(0, n_people, 60)
    second = rng.integers(0, n_people, 60)
    keep = first != second
    first, second = first[keep], second[keep]
    if symmetric:
        lo, hi = np.minimum(first, second), np.maximum(first, second)
        key = np.unique(lo * 1000 + hi)
        first, second = key // 1000, key % 1000
    else:
        key = np.unique(first * 1000 + second)
        first, second = key // 1000, key % 1000
    eta = 1.5 * rng.normal(size=n_people)
    cell = _cell(eta, first, second, np.log(rng.exponential(size=n_people)) - eta, symmetric=symmetric)
    assert cell["expected"] == pytest.approx(_brute_target(eta, eta, first, second, symmetric, True), abs=1e-12)


def test_within_person_target_matches_brute_force():
    rng = np.random.default_rng(5)
    n = 40
    e1, e2 = rng.normal(size=n), 1.5 * rng.normal(size=n)
    people = np.arange(n)
    cell = ot.onset_tau_cell(
        np.log(rng.exponential(size=n)) - e1,
        np.log(rng.exponential(size=n)) - e2,
        e1,
        e2,
        people,
        people,
        people,
        symmetric=False,
        within_person=True,
    )
    assert cell["expected"] == pytest.approx(_brute_target(e1, e2, people, people, False, False), abs=1e-12)


def test_target_is_the_mean_of_the_statistic_over_onset_redraws():
    """A dense cell with many shared-person comparisons: target vs 20,000 redraws of the onsets."""
    rng = np.random.default_rng(6)
    first, second, _ = _sibships([3, 4, 5, 2, 6])
    eta = rng.normal(size=20) * 1.5
    X, Y, _, per = ot._cell_entries(first, second, True)
    target = _cell(eta, first, second, np.zeros(20))["expected"]
    draws = np.array(
        [ot._ranked_rows(w[X], w[Y], True, per)[0].mean() for w in np.log(rng.exponential(size=(20000, 20))) - eta]
    )
    assert abs(draws.mean() - target) < 4 * draws.std() / math.sqrt(len(draws))


def test_taylor_path_agrees_with_the_exact_path_within_the_bound(monkeypatch):
    rng = np.random.default_rng(7)
    first, second, _ = _sibships(rng.integers(2, 5, 150))
    n_people = int(max(first.max(), second.max())) + 1
    eta = 2.0 * rng.normal(size=n_people)
    onsets = np.log(rng.exponential(size=n_people)) - eta
    exact = _cell(eta, first, second, onsets)
    assert exact["grid"] == 0
    monkeypatch.setattr(ot, "EXACT_MAX_ENTRIES", 0)
    binned = _cell(eta, first, second, onsets)
    assert binned["grid"] >= ot.GRID_START
    assert abs(binned["expected"] - exact["expected"]) <= binned["numerical_bound"]
    assert binned["numerical_bound"] <= ot.GRID_BOUND_SE_SHARE * binned["se"] or binned["grid"] == ot.GRID_MAX


def test_one_huge_component_gives_small_n_eff():
    first, second, _ = _sibships([40] + [2] * 99)
    n_people = int(second.max()) + 1
    cell = _cell(np.zeros(n_people), first, second, np.arange(n_people, dtype=np.float64))
    assert cell["n_components"] == 100
    assert cell["n_eff"] < 2
    assert cell["max_component_share"] > 0.85


def test_shared_workload_cap_leaves_the_target_uncomputed(monkeypatch):
    monkeypatch.setattr(ot, "SHARED_WORKLOAD_MAX", 10)
    first, second, _ = _sibships([6])
    cell = _cell(np.zeros(6), first, second, np.arange(6, dtype=np.float64))
    assert cell["expected"] is None
    assert cell["se"] is None
    assert "workload" in cell["note"]


# ---------------------------------------------------------------------------
# SE calibration against onset redraws on a simulated pedigree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["global", "per_generation"])
def test_clustered_se_matches_the_spread_over_onset_redraws(mode):
    ped = run_simulation(
        seed=11, N=700, G_ped=3, mating_lambda=0.5, p_mztwin=0.02, A1=0.5, C1=0.2, E1=0.3,
        A2=0.4, C2=0.2, E2=0.4, rA=0.3, rC=0.5,
    )  # fmt: skip
    L, sex, gen = ped["liability1"].to_numpy(), ped["sex"].to_numpy().astype(float), ped["generation"].to_numpy()
    mean, scaled = standardize_beta(L, 1.2, mode, gen)
    eta = scaled * (L - mean) + 0.5 * sex
    graph = PedigreeGraph.from_frame(ped.select("id", "mother", "father", "twin", "sex"))
    rng = np.random.default_rng(12)
    checked = 0
    for code in ("FS", "MHS", "PHS", "MO", "FO"):
        block = graph.relationship_pairs(categories=[code])[code]
        first, second = np.asarray(block.first_rows, np.int64), np.asarray(block.second_rows, np.int64)
        symmetric = code in ot._SYMMETRIC
        component = ot._component_labels(first, second, len(L))
        cell = ot.onset_tau_cell(eta, eta, eta, eta, first, second, component, symmetric=symmetric)
        if cell["n_eff"] < 100:
            continue
        X, Y, mirror, per = ot._cell_entries(first, second, symmetric)
        shared, _ = ot._shared_terms(eta, X, Y, mirror, len(L))
        extra = shared + (np.tanh((eta[X] - eta[Y]) / 2) ** 2 if symmetric else 0.0)
        expected, _ = ot._expected_rows(eta[X], eta[Y], extra, per, cell["grid"])
        taus, variances = [], []
        for _ in range(400):
            w = np.log(rng.exponential(size=len(L))) - eta
            observed = ot._ranked_rows(w[X], w[Y], symmetric, per)[0]
            taus.append(observed.mean())
            variances.append(ot._clustered_se(observed, expected, component[X]) ** 2)
        ratio = math.sqrt(np.mean(variances)) / np.std(taus, ddof=1)
        assert 0.85 <= ratio <= 1.15, (code, ratio)
        assert abs(np.mean(taus) - expected.mean()) < 4 * np.std(taus, ddof=1) / math.sqrt(400), code
        checked += 1
    assert checked >= 3


def test_a_dominant_component_makes_the_se_conservative_not_anti_conservative():
    """400 small sibships plus one sparse component holding about half the entries: n_eff fails the gate,
    and the clustered SE over-covers rather than under-covers (ADR 0027 decision 6)."""
    rng = np.random.default_rng(13)
    first, second, family = _sibships(rng.integers(2, 5, 400))
    start = len(family)
    k = 600
    edges = {(start + i, start + i + 1) for i in range(k - 1)}
    while len(edges) < 2 * k:
        i, j = sorted(int(v) for v in rng.choice(k, 2, replace=False) + start)
        edges.add((i, j))
    big = np.array(sorted(edges), dtype=np.int64)
    first, second = np.concatenate([first, big[:, 0]]), np.concatenate([second, big[:, 1]])
    family = np.concatenate([family, np.full(k, family.max() + 1)])
    eta = np.sqrt(0.5) * rng.normal(size=family.max() + 1)[family] + np.sqrt(0.5) * rng.normal(size=len(family))
    component = ot._component_labels(first, second, len(eta))
    cell = ot.onset_tau_cell(eta, eta, eta, eta, first, second, component, symmetric=True)
    assert cell["n_eff"] < 100
    X, Y, mirror, per = ot._cell_entries(first, second, True)
    shared, _ = ot._shared_terms(eta, X, Y, mirror, len(eta))
    expected, _ = ot._expected_rows(eta[X], eta[Y], shared + np.tanh((eta[X] - eta[Y]) / 2) ** 2, per, cell["grid"])
    taus, variances = [], []
    for w in np.log(rng.exponential(size=(1000, len(eta)))) - eta:
        observed = ot._ranked_rows(w[X], w[Y], True, per)[0]
        taus.append(observed.mean())
        variances.append(ot._clustered_se(observed, expected, component[X]) ** 2)
    assert math.sqrt(np.mean(variances)) / np.std(taus, ddof=1) >= 0.95


# ---------------------------------------------------------------------------
# The phenotyped population
# ---------------------------------------------------------------------------

_PHENOTYPE = {
    "G_pheno": 2,
    "standardize": "global",
    "phenotype_model1": "frailty",
    "beta1": 1.0,
    "beta_sex1": 0.5,
    "phenotype_params1": {"distribution": "weibull", "scale": 2160, "rho": 0.8},
    "phenotype_model2": "frailty",
    "beta2": 1.5,
    "beta_sex2": 0.0,
    "phenotype_params2": {"distribution": "gamma", "shape": 2.0, "scale": 30.0, "standardize_hazard": "per_generation"},
}


@pytest.fixture(scope="module")
def simulated():
    ped = run_simulation(
        seed=21, N=400, G_ped=3, mating_lambda=0.5, p_mztwin=0.02, A1=0.5, C1=0.2, E1=0.3,
        A2=0.4, C2=0.2, E2=0.4, rA=0.3, rC=0.5,
    )  # fmt: skip
    raw = run_phenotype(ped, seed=5, **_PHENOTYPE)
    return ped, raw


def test_compute_onset_tau_reports_every_cell(simulated):
    ped, raw = simulated
    result = ot.compute_onset_tau(raw, ped, _PHENOTYPE, seed=5)
    assert set(result["cells"]) == {"trait1", "trait2", "within_person"}
    for t in ("trait1", "trait2"):
        assert set(result["cells"][t]) == set(ot.CELL_CATEGORIES)
        fs = result["cells"][t]["FS"]
        assert -1 <= fs["observed"] <= 1
        assert fs["se"] > 0
        assert fs["tau_oracle"] is not None
    assert result["skipped_traits"] == {}


def test_compute_onset_tau_skips_non_frailty_traits(simulated):
    ped, raw = simulated
    result = ot.compute_onset_tau(raw, ped, {**_PHENOTYPE, "phenotype_model2": "adult"}, seed=5)
    assert set(result["cells"]) == {"trait1"}
    assert result["skipped_traits"] == {"trait2": "model adult is not frailty"}


@pytest.mark.parametrize("trait", [1, 2])
def test_linear_predictor_orders_onsets_like_the_model(simulated, trait):
    """Onsets increase with log E − η for the model's own E draws: η matches the model's composition."""
    ped, raw = simulated
    rows = raw.join(ped, on="id", how="left", maintain_order="left")
    from simace.phenotype.models import MODELS

    model = MODELS["frailty"].from_config({k: _PHENOTYPE[k] for k in (f"phenotype_params{trait}", f"beta{trait}", f"beta_sex{trait}")}, trait)  # fmt: skip
    eta = ot._linear_predictor(
        model,
        rows[f"liability{trait}"].to_numpy(),
        rows["sex"].to_numpy().astype(float),
        rows["generation"].to_numpy(),
        _PHENOTYPE["standardize"],
    )
    e = np.random.default_rng(5 if trait == 1 else 105).exponential(size=len(rows))
    w = np.log(e) - eta
    t = rows[f"t{trait}"].to_numpy()
    order = np.argsort(w)
    assert np.all(np.diff(t[order]) >= 0)


class TestEntryCap:
    def test_small_cells_are_kept_whole(self):
        first, second, _ = _sibships([3, 4])
        component = ot._component_labels(first, second, 7)
        kept = ot._cap_components(first, second, component, 2 * len(first), seed=(1, 0))
        assert kept[0] is first
        assert kept[1] is second

    def test_large_cells_keep_whole_components_near_the_cap(self, monkeypatch):
        monkeypatch.setattr(ot, "ENTRY_CAP", 2000)
        rng = np.random.default_rng(8)
        first, second, family = _sibships(rng.integers(2, 6, 3000))
        component = ot._component_labels(first, second, len(family))
        n_entries = 2 * len(first)
        kf, _ = ot._cap_components(first, second, component, n_entries, seed=(3, 1))
        again = ot._cap_components(first, second, component, n_entries, seed=(3, 1))
        assert np.array_equal(kf, again[0])
        assert 1500 < 2 * len(kf) < 2500
        # Whole components: every kept component keeps all of its pairs.
        kept_components = np.unique(component[kf])
        assert np.isin(component[first], kept_components).sum() == len(kf)

    def test_a_capped_cell_is_unbiased_for_its_target(self, monkeypatch):
        monkeypatch.setattr(ot, "ENTRY_CAP", 600)
        rng = np.random.default_rng(9)
        first, second, family = _sibships(rng.integers(2, 5, 600))
        component = ot._component_labels(first, second, len(family))
        eta = rng.normal(size=len(family))
        kf, ks = ot._cap_components(first, second, component, 2 * len(first), seed=(4, 1))
        target = ot.onset_tau_cell(eta, eta, eta, eta, kf, ks, component, symmetric=True)["expected"]
        X, Y, _, per = ot._cell_entries(kf, ks, True)
        draws = [
            ot._ranked_rows(w[X], w[Y], True, per)[0].mean()
            for w in np.log(rng.exponential(size=(4000, len(eta)))) - eta
        ]
        assert abs(np.mean(draws) - target) < 4 * np.std(draws) / math.sqrt(len(draws))


@pytest.fixture(scope="module")
def family():
    ped = run_simulation(
        seed=31, N=600, G_ped=4, mating_lambda=1.5, p_mztwin=0.05, A1=0.5, C1=0.2, E1=0.3,
        A2=0.4, C2=0.2, E2=0.4, rA=0.3, rC=0.5,
    )  # fmt: skip
    return ped.filter(pl.col("generation") >= 1).select("id", "mother", "father", "twin", "sex")


class TestFamilyUnits:
    """Unit sampling must find exactly the full graph's pairs within the kept units, in whole components."""

    @staticmethod
    def _keys(first, second, symmetric):
        lo, hi = (np.minimum(first, second), np.maximum(first, second)) if symmetric else (first, second)
        return set(zip(lo.tolist(), hi.tolist(), strict=True))

    @pytest.mark.parametrize("code", ot.CELL_CATEGORIES)
    def test_fraction_one_reproduces_the_full_graph(self, family, code):
        full = PedigreeGraph.from_frame(family).relationship_pairs(categories=[code])[code]
        units = ot._family_units(*(family[c].to_numpy() for c in ("id", "mother", "father", "twin")))
        first, second = ot._unit_pairs(family, code, units[code][0], 1.0, seed=(1, 0, 0))
        symmetric = code in ot._SYMMETRIC
        assert self._keys(first, second, symmetric) == self._keys(full.first_rows, full.second_rows, symmetric)
        assert units[code][1] == pytest.approx(len(first) * (2 if symmetric else 1), rel=0.05)

    @pytest.mark.parametrize("code", ot.CELL_CATEGORIES)
    def test_sampled_units_keep_whole_components(self, family, code):
        full = PedigreeGraph.from_frame(family).relationship_pairs(categories=[code])[code]
        units = ot._family_units(*(family[c].to_numpy() for c in ("id", "mother", "father", "twin")))
        first, second = ot._unit_pairs(family, code, units[code][0], 0.3, seed=(2, 0, 0))
        symmetric = code in ot._SYMMETRIC
        kept = self._keys(first, second, symmetric)
        everything = self._keys(full.first_rows, full.second_rows, symmetric)
        assert kept <= everything
        assert 0 < len(kept) < len(everything)
        # Every full-graph pair touching a kept person is kept: components are whole.
        people = {p for pair in kept for p in pair}
        assert {pair for pair in everything if pair[0] in people or pair[1] in people} == kept


def test_null_onsets_count_as_ties_at_infinity():
    first, second, _ = _sibships([2, 2, 2])
    onsets = np.array([1.0, 2.0, math.inf, math.inf, 3.0, 4.0])
    eta = np.zeros(6)
    cell = _cell(eta, first, second, onsets)
    assert cell["tie_share"] > 0
    assert pl.Series(onsets).is_infinite().sum() == 2
