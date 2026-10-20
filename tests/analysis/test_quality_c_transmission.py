"""Validators under C transmission (ADR 0025): the regression check, C sharing, PHS C, AM equilibrium, total variance."""

import numpy as np
import polars as pl
import pytest

from simace.analysis.mate_correlation import mate_pedigree
from simace.analysis.quality import build_quality_report, validate_half_sibs, validate_statistical
from simace.analysis.quality._common import cluster_codes, mate_correlations, sibling_moments
from simace.analysis.quality.am_equilibrium import validate_am_equilibrium
from simace.analysis.quality.c_transmission import cluster_robust_ols, validate_c_transmission
from simace.core.pedigree_arrays import PedigreeArrays
from simace.simulation.simulate import run_simulation

_SIM = dict(
    N=4000,
    G_ped=3,
    G_sim=3,
    mating_lambda=0.5,
    p_mztwin=0.02,
    A1=0.4,
    C1=0.3,
    E1=0.3,
    A2=0.5,
    C2=0.2,
    E2=0.3,
    rA=0.3,
    rC=0.5,
    rE=0.0,
)
_CT = {"Ctm_maternal1": 0.5, "Ctm_paternal1": 0.3, "Ctm_maternal2": 0.4, "Ctm_paternal2": 0.0}


def _params(**overrides):
    return {**_SIM, "assort1": 0.0, "assort2": 0.0, "assort_matrix": None, "mating_model": "standard", **overrides}


@pytest.fixture(scope="module")
def rm_pedigree():
    return run_simulation(seed=7, **_SIM, **_CT)


@pytest.fixture(scope="module")
def am_pedigree():
    return run_simulation(seed=8, **_SIM, assort1=0.3, **_CT)


def test_cr1_standard_errors_cover_under_large_sibships():
    """Clustered errors (sibships of 20 sharing u) need cluster-robust SEs: row-wise SEs under-cover."""
    rng = np.random.default_rng(0)
    covered_cr1, covered_rowwise = 0, 0
    n_reps, n_clusters, size = 400, 60, 20
    clusters = np.repeat(np.arange(n_clusters), size)
    for _ in range(n_reps):
        x = rng.normal(size=n_clusters)[clusters] + 0.2 * rng.normal(size=n_clusters * size)
        y = 0.5 * x + rng.normal(size=n_clusters)[clusters] + 0.1 * rng.normal(size=n_clusters * size)
        X = np.column_stack([np.ones_like(x), x])
        beta, se = cluster_robust_ols(y, X, clusters)
        resid = y - X @ beta
        rowwise = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=2))
        covered_cr1 += abs(beta[1] - 0.5) <= 1.96 * se[1]
        covered_rowwise += abs(beta[1] - 0.5) <= 1.96 * rowwise[1]
    assert covered_cr1 / n_reps == pytest.approx(0.95, abs=0.03)
    assert covered_rowwise / n_reps < 0.6


def test_cluster_codes_are_dense_and_match_unique():
    keys = np.array([7, 3, 7, 11, 3, 3, 42], dtype=np.int64)
    _, expected = np.unique(keys, return_inverse=True)
    assert np.array_equal(cluster_codes(keys), expected)


def test_transmission_check_passes_on_random_and_assortative_mating(rm_pedigree, am_pedigree):
    for ped, params in ((rm_pedigree, _params(**_CT)), (am_pedigree, _params(assort1=0.3, **_CT))):
        result = validate_c_transmission(ped, params, PedigreeArrays.from_frame(ped))
        for t in (1, 2):
            assert result[f"c{t}_transmission"]["passed"], result[f"c{t}_transmission"]["details"]
            assert result[f"c{t}_transmission"]["expected"] == [_CT[f"Ctm_maternal{t}"], _CT[f"Ctm_paternal{t}"]]


def test_transmission_check_fails_on_the_wrong_coefficients(rm_pedigree):
    wrong = _params(**{**_CT, "Ctm_maternal1": 0.3})
    result = validate_c_transmission(rm_pedigree, wrong, PedigreeArrays.from_frame(rm_pedigree))
    assert not result["c1_transmission"]["passed"]


def test_transmission_check_is_informational_when_off_or_underpowered(rm_pedigree):
    ped = PedigreeArrays.from_frame(rm_pedigree)
    off = validate_c_transmission(rm_pedigree, _params(), ped)
    assert all(r.get("informational") for r in off.values())
    no_c = run_simulation(seed=1, **{**_SIM, "N": 300, "C1": 0.0}, Ctm_maternal1=0.5)
    small = validate_c_transmission(no_c, _params(C1=0.0, Ctm_maternal1=0.5), PedigreeArrays.from_frame(no_c))
    assert "no variance" in small["c1_transmission"]["details"]
    tiny = run_simulation(seed=1, **{**_SIM, "N": 60}, Ctm_maternal1=0.5)
    few = validate_c_transmission(tiny, _params(Ctm_maternal1=0.5), PedigreeArrays.from_frame(tiny))
    assert "households <" in few["c1_transmission"]["details"]


def test_exact_copying_is_judged_on_the_numerical_floor():
    sim = {**_SIM, "N": 20_000, "rC": 0.0}  # b_m1 = 1 beside b_m2 = 0 is infeasible under rC != 0
    ped = run_simulation(seed=3, **sim, Ctm_maternal1=1.0)
    result = validate_c_transmission(ped, _params(rC=0.0, Ctm_maternal1=1.0), PedigreeArrays.from_frame(ped))
    assert result["c1_transmission"]["passed"], result["c1_transmission"]["details"]


def test_c_sharing_groups_by_mating_pair_under_paternal_transmission(rm_pedigree):
    result = validate_statistical(rm_pedigree, _params(**_CT))
    assert result["c1_inheritance"]["passed"]
    assert "mating pairs" in result["c1_inheritance"]["details"]
    assert result["c2_inheritance"]["passed"]  # b_f2 = 0: householdmates still share C2
    assert "families" in result["c2_inheritance"]["details"]
    by_mother = validate_statistical(rm_pedigree, _params(**{**_CT, "Ctm_paternal1": 0.0}))
    assert not by_mother["c1_inheritance"]["passed"]


def test_total_variance_is_the_variance_of_the_sum(am_pedigree):
    result = validate_statistical(am_pedigree, _params(assort1=0.3, **_CT))
    founders = am_pedigree.filter(am_pedigree["mother"] == -1)
    direct = np.var((founders["A1"] + founders["C1"] + founders["E1"]).to_numpy(), ddof=1)
    assert result["total_variance_trait1"]["observed"] == pytest.approx(direct)


def test_phs_shared_c_reports_bf_squared_under_the_gate(rm_pedigree, am_pedigree):
    params = _params(**_CT)
    result = validate_half_sibs(
        rm_pedigree,
        params,
        sibling_moments(rm_pedigree, PedigreeArrays.from_frame(rm_pedigree)),
        mate_correlations(rm_pedigree, mate_pedigree(rm_pedigree)),
    )
    assert result["half_sib_shared_C1"]["expected"] == pytest.approx(0.09)
    assert result["half_sib_shared_C1"].get("informational")
    assert result["half_sib_shared_C1"]["observed"] == pytest.approx(0.09, abs=0.05)
    am = validate_half_sibs(
        am_pedigree,
        _params(assort1=0.3, **_CT),
        sibling_moments(am_pedigree, PedigreeArrays.from_frame(am_pedigree)),
        mate_correlations(am_pedigree, mate_pedigree(am_pedigree)),
    )
    assert am["half_sib_shared_C1"].get("informational")


def test_am_equilibrium_is_informational_under_transmission(am_pedigree):
    result = validate_am_equilibrium(am_pedigree, _params(assort1=0.3, **_CT))
    assert result["am_equilibrium_A1"].get("informational")
    assert "observed" in result["am_equilibrium_A1"]


def test_sibling_e_independence_uses_an_se_based_tolerance():
    """Independent sib E passes at 4 SE (a fixed 0.1 failed ~2.5% of reps); genuinely shared sib E still fails."""
    rng = np.random.default_rng(0)
    n = 500
    shared = rng.normal(size=n)
    frame = pl.DataFrame(
        {
            "id": np.arange(3 * n),
            "mother": np.r_[np.full(n, -1), np.repeat(np.arange(n), 2)],
            "father": np.r_[np.full(n, -1), np.repeat(np.arange(n), 2)],
            **{c: rng.normal(size=3 * n) for c in ("A1", "C1", "A2", "C2", "E2")},
        }
    )
    independent = frame.with_columns(E1=rng.normal(size=3 * n))
    result = validate_statistical(independent, _params())["e1_independence"]
    assert result["passed"], result["details"]
    assert result["tolerance"] == pytest.approx(4 / np.sqrt(n - 1))
    correlated = frame.with_columns(E1=np.r_[rng.normal(size=n), np.repeat(shared, 2) + 0.5 * rng.normal(size=2 * n)])
    assert not validate_statistical(correlated, _params())["e1_independence"]["passed"]


def test_full_report_passes_on_a_random_mating_transmission_pedigree(rm_pedigree):
    report = build_quality_report(rm_pedigree, _params(**_CT))
    failed = [
        f"{cat}.{name}: {r.get('details')}"
        for cat, checks in report.items()
        if isinstance(checks, dict)
        for name, r in checks.items()
        if isinstance(r, dict) and r.get("passed") is False
    ]
    assert failed == []
