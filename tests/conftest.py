"""Shared fixtures for sim_ace test suite."""

import numpy as np
import pytest

from sim_ace.simulate import (
    generate_correlated_components,
    mating,
    reproduce,
    run_simulation,
)


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def default_params():
    """Default simulation parameters matching config defaults."""
    return dict(
        seed=42,
        N=1000,
        G_ped=3,
        G_sim=3,
        fam_size=2.3,
        p_mztwin=0.02,
        p_nonsocial_father=0.05,
        A1=0.5,
        C1=0.2,
        A2=0.5,
        C2=0.2,
        rA=0.3,
        rC=0.5,
    )


@pytest.fixture
def small_pedigree(default_params):
    """A small pedigree (N=1000, G_ped=3) for fast tests."""
    return run_simulation(**default_params)


@pytest.fixture
def founders_and_offspring(rng):
    """Create a one-generation setup: founders + one generation of offspring.

    Returns (pheno, sex, parents, twins, household_ids, offspring, sex_offspring)
    with known variance components.
    """
    N = 2000
    A1, C1 = 0.5, 0.2
    A2, C2 = 0.5, 0.2
    rA, rC = 0.3, 0.5
    E1 = 1.0 - A1 - C1
    E2 = 1.0 - A2 - C2

    sd_A1, sd_C1, sd_E1 = np.sqrt(A1), np.sqrt(C1), np.sqrt(E1)
    sd_A2, sd_C2, sd_E2 = np.sqrt(A2), np.sqrt(C2), np.sqrt(E2)

    sex = rng.binomial(size=N, n=1, p=0.5)
    a1, a2 = generate_correlated_components(rng, N, sd_A1, sd_A2, rA)
    c1, c2 = generate_correlated_components(rng, N, sd_C1, sd_C2, rC)
    e1 = rng.normal(size=N, scale=sd_E1)
    e2 = rng.normal(size=N, scale=sd_E2)
    pheno = np.stack([a1, c1, e1, a2, c2, e2], axis=-1)

    parents, twins, household_ids = mating(rng, sex, fam_size=2.3, p_nonsocial_father=0.05, p_mztwin=0.02)
    offspring, sex_offspring = reproduce(
        rng,
        pheno,
        parents,
        twins,
        household_ids,
        sd_A1,
        sd_E1,
        sd_C1,
        sd_A2,
        sd_E2,
        sd_C2,
        rA,
        rC,
    )
    return dict(
        pheno=pheno,
        sex=sex,
        parents=parents,
        twins=twins,
        household_ids=household_ids,
        offspring=offspring,
        sex_offspring=sex_offspring,
        sd_A1=sd_A1,
        sd_A2=sd_A2,
        sd_C1=sd_C1,
        sd_C2=sd_C2,
        sd_E1=sd_E1,
        sd_E2=sd_E2,
        rA=rA,
        rC=rC,
    )
