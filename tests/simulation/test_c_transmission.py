"""C transmission in ``reproduce`` and ``run_simulation`` (ADR 0025).

Exact properties (who shares C, which inputs reach offspring C) use
constructed reproduction inputs. Statistical properties (recovered
coefficients, per-generation Var(C) and cross-trait C correlation) use
``run_simulation`` at a size where the tolerances are several standard
errors wide.
"""

import numpy as np
import polars as pl
import pytest

from simace.core.relationships import reference_corr, reference_midparent_slope
from simace.simulation.params import CTransmission
from simace.simulation.simulate import _transmitted_c, reproduce, run_simulation

_BASE = dict(
    N=20_000,
    G_ped=3,
    G_sim=3,
    mating_lambda=0.5,
    p_mztwin=0.0,
    A1=0.4,
    C1=0.3,
    E1=0.3,
    A2=0.3,
    C2=0.2,
    E2=0.5,
    rA=0.2,
    rC=0.4,
    rE=0.0,
)


def _family_inputs(rng: np.random.Generator, c_sd: float = 0.5):
    """Two mothers, three fathers, maternal half-sibs in household 0 and paternal half-sibs across households."""
    pheno = rng.normal(0.0, c_sd, size=(5, 6))  # rows 0-1 mothers, 2-4 fathers
    parents = np.array([[0, 2], [0, 2], [0, 3], [1, 3], [1, 4], [1, 4]])
    households = np.array([0, 0, 0, 1, 1, 1])
    return pheno, parents, households


def _reproduce(seed: int, pheno, parents, households, ct: CTransmission, sd_c: float = 3.0):
    return reproduce(
        np.random.default_rng(seed),
        pheno,
        parents,
        np.empty((0, 2), dtype=np.int64),
        households,
        sd_A1=0.5,
        sd_E1=0.5,
        sd_C1=sd_c,
        sd_A2=0.5,
        sd_E2=0.5,
        sd_C2=sd_c,
        rA=0.0,
        rC=0.3,
        rE=0.0,
        c_transmission=ct,
    )


def _constant_within(values: np.ndarray, groups: np.ndarray) -> bool:
    _, inverse = np.unique(groups, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    representative = np.empty(inverse.max() + 1)
    representative[inverse] = values
    return np.array_equal(values, representative[inverse])


@pytest.mark.parametrize("seed", range(5))
def test_householdmates_share_c_without_paternal_transmission(seed):
    pheno, parents, households = _family_inputs(np.random.default_rng(seed))
    offspring, _ = _reproduce(seed, pheno, parents, households, CTransmission(maternal=(0.6, 0.4)))
    for col in (1, 4):
        assert _constant_within(offspring[:, col], households)


@pytest.mark.parametrize("seed", range(5))
def test_paternal_transmission_shares_c_within_mating_pairs_only(seed):
    pheno, parents, households = _family_inputs(np.random.default_rng(seed))
    offspring, _ = _reproduce(seed, pheno, parents, households, CTransmission((0.5, 0.5), (0.4, 0.3)))
    for col in (1, 4):
        assert _constant_within(offspring[:, col], parents)
        assert offspring[1, col] != offspring[2, col]  # maternal half-sibs, one household


@pytest.mark.parametrize("seed", range(5))
def test_parental_c_reaches_offspring_and_nothing_else_moves(seed):
    pheno, parents, households = _family_inputs(np.random.default_rng(seed))
    altered = pheno.copy()
    altered[:2, [1, 4]] += 1.0  # mothers' C
    ct = CTransmission(maternal=(0.5, 0.5))
    offspring, sex = _reproduce(seed, pheno, parents, households, ct)
    offspring_alt, sex_alt = _reproduce(seed, altered, parents, households, ct)
    assert not np.array_equal(offspring[:, [1, 4]], offspring_alt[:, [1, 4]])
    assert np.array_equal(offspring[:, [0, 2, 3, 5]], offspring_alt[:, [0, 2, 3, 5]])
    assert np.array_equal(sex, sex_alt)


def test_transmitted_c_tops_up_to_the_target_when_c_grows():
    rng = np.random.default_rng(3)
    n_mothers = 40_000
    pheno = np.zeros((2 * n_mothers, 6))
    pheno[:n_mothers, 1] = rng.normal(0.0, np.sqrt(0.1), n_mothers)
    parents = np.column_stack([np.arange(n_mothers), np.arange(n_mothers) + n_mothers])
    households = np.arange(n_mothers)
    target = np.diag([0.2, 0.0])
    c1, _ = _transmitted_c(rng, pheno, parents, households, n_mothers, target, CTransmission(maternal=(1.0, 0.0)))
    fresh = c1 - pheno[:n_mothers, 1]
    assert np.var(fresh) == pytest.approx(0.1, abs=0.005)
    assert np.var(c1) == pytest.approx(0.2, abs=0.006)


def test_transmitted_covariance_beyond_sampling_raises():
    rng = np.random.default_rng(0)
    n_mothers = 5_000
    pheno = np.zeros((2 * n_mothers, 6))
    pheno[:n_mothers, 1] = rng.normal(0.0, 1.0, n_mothers)
    parents = np.column_stack([np.arange(n_mothers), np.arange(n_mothers) + n_mothers])
    with pytest.raises(ValueError, match="exceeds the configured"):
        _transmitted_c(
            rng, pheno, parents, np.arange(n_mothers), n_mothers, np.diag([0.2, 0.0]), CTransmission((1.0, 0.0))
        )


def _parent_child(ped: pl.DataFrame, trait: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = ped[f"C{trait}"].to_numpy()
    mother, father = ped["mother"].to_numpy(), ped["father"].to_numpy()
    kids = np.flatnonzero(mother >= 0)
    return c[kids], c[mother[kids]], c[father[kids]]


def _ols(y: np.ndarray, *x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(y), *x])
    return np.linalg.lstsq(design, y, rcond=None)[0][1:]


@pytest.mark.parametrize(
    "assort",
    [0.0, 0.3],
    ids=["random_mating", "assortative"],
)
def test_ols_of_child_c_on_parent_c_recovers_the_coefficients(assort):
    ct = {"Ctm_maternal1": 0.5, "Ctm_paternal1": 0.3, "Ctm_maternal2": 0.4, "Ctm_paternal2": 0.0}
    ped = run_simulation(seed=11, **_BASE, assort1=assort, **ct)
    for trait in (1, 2):
        child, mom, dad = _parent_child(ped, trait)
        b_m, b_f = _ols(child, mom, dad)
        assert b_m == pytest.approx(ct[f"Ctm_maternal{trait}"], abs=0.03)
        assert b_f == pytest.approx(ct[f"Ctm_paternal{trait}"], abs=0.03)


def test_c_variance_and_cross_trait_correlation_hold_each_generation():
    ped = run_simulation(seed=5, **_BASE, Ctm_maternal1=0.6, Ctm_paternal1=0.5, Ctm_maternal2=0.7, Ctm_paternal2=0.2)
    for g in range(3):
        gen = ped.filter(pl.col("generation") == g)
        c1, c2 = gen["C1"].to_numpy(), gen["C2"].to_numpy()
        assert np.var(c1) == pytest.approx(0.3, rel=0.06), g
        assert np.var(c2) == pytest.approx(0.2, rel=0.06), g
        assert np.corrcoef(c1, c2)[0, 1] == pytest.approx(0.4, abs=0.04), g


@pytest.mark.parametrize("seed", range(3))
def test_full_maternal_copying_of_constant_c_adds_almost_nothing(seed):
    """The fresh draw only absorbs sampling error in S, a few percent of Var(C) at this N."""
    ped = run_simulation(seed=seed, **_BASE, Ctm_maternal1=1.0, Ctm_maternal2=1.0)
    for trait in (1, 2):
        child, mom, _ = _parent_child(ped, trait)
        assert np.var(child - mom) < 0.05 * np.var(child)


def test_analytic_parent_offspring_references_match_a_random_mating_simulation():
    """Numeric check of the RM analytic lines: MO, FO, and the midparent slope (A + (b_m + b_f) C) / V_P."""
    ct = {"Ctm_maternal1": 0.5, "Ctm_paternal1": 0.3}
    params = {**_BASE, "assort1": 0.0, "assort2": 0.0, "assort_matrix": None, **ct}
    ped = run_simulation(seed=21, **{**_BASE, "N": 40_000}, **ct)
    liab = ped["liability1"].to_numpy()
    mother, father = ped["mother"].to_numpy(), ped["father"].to_numpy()
    kids = np.flatnonzero(mother >= 0)
    child, mom, dad = liab[kids], liab[mother[kids]], liab[father[kids]]
    assert np.corrcoef(child, mom)[0, 1] == pytest.approx(reference_corr("MO", params, 1), abs=0.015)
    assert np.corrcoef(child, dad)[0, 1] == pytest.approx(reference_corr("FO", params, 1), abs=0.015)
    assert _ols(child, (mom + dad) / 2)[0] == pytest.approx(reference_midparent_slope(params, 1), abs=0.03)


@pytest.mark.parametrize("seed", range(20))
def test_full_maternal_copying_at_small_test_size_never_trips_the_sampling_check(seed):
    """small_test's N and G_sim with b_m = 1; CLIP_SE_MULTIPLIER was calibrated on seeds 0-999 at this size."""
    run_simulation(seed=seed, **{**_BASE, "N": 1000, "G_ped": 6, "G_sim": 8}, Ctm_maternal1=1.0, Ctm_maternal2=1.0)


def test_wright_fisher_runs_with_transmission():
    ped = run_simulation(
        seed=2, **{**_BASE, "N": 5_000}, mating_model="wright_fisher", Ctm_maternal1=0.5, Ctm_paternal1=0.3
    )
    child, mom, dad = _parent_child(ped, 1)
    b_m, b_f = _ols(child, mom, dad)
    assert b_m == pytest.approx(0.5, abs=0.05)
    assert b_f == pytest.approx(0.3, abs=0.05)


_BOUND = {"Ctm_maternal1": 0.7, "Ctm_paternal1": 0.7}  # b_m² + b_f² = 0.98


def test_assortment_pushing_transmitted_c_past_the_target_raises():
    """Mates' C correlation (r_CC ~ 0.3 * 0.3 / 1) adds ~0.05 C beyond the 0.98 C own share: structural, not sampling."""
    with pytest.raises(ValueError, match="mates' correlated C"):
        run_simulation(seed=1, **{**_BASE, "rC": 0.0}, assort1=0.3, **_BOUND)


@pytest.mark.parametrize("seed", range(10))
def test_random_mating_at_the_bound_does_not_trip_the_cross_term(seed):
    """At b_m² + b_f² = 1 under random mating the mates' cross term is pure noise and must not raise."""
    half = float(np.sqrt(0.5))
    run_simulation(seed=seed, **{**_BASE, "N": 2000, "rC": 0.0}, Ctm_maternal1=half, Ctm_paternal1=half)


def test_clips_are_logged_with_their_recorded_generation():
    clips: list = []
    run_simulation(
        seed=0, **{**_BASE, "N": 1000, "G_ped": 4, "G_sim": 6}, Ctm_maternal1=1.0, Ctm_maternal2=1.0, clip_log=clips
    )
    assert clips, "constant C with b_m = 1 clips in some generation at N = 1000"
    assert {c["generation"] for c in clips} <= set(range(-2, 4))
    assert all(0 < c["excess"] <= c["tolerance"] and len(c["direction"]) == 2 for c in clips)


def test_weak_assortment_at_small_n_is_clipped_and_logged_not_silent():
    """Below its standard error the structural excess is indistinguishable from noise: it must at least be recorded."""
    clipped = 0
    for seed in range(5):
        clips: list = []
        run_simulation(seed=seed, **{**_BASE, "N": 1000, "rC": 0.0, "C1": 0.2}, assort1=0.3, **_BOUND, clip_log=clips)
        clipped += any(c["generation"] >= 0 for c in clips)
    assert clipped >= 4  # 39 of seeds 0-39 clip in a recorded generation
