"""Shared fixtures for simace test suite."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simace.simulation.simulate import (
    generate_correlated_components,
    mating,
    reproduce,
    run_simulation,
)


def schema_pad(df: pd.DataFrame, schema: Mapping[str, str]) -> pd.DataFrame:
    """Pad ``df`` with zero/false defaults for any columns required by ``schema``.

    Lets unit-test fixtures stay focused on the columns under test while still
    producing a frame that satisfies the pipeline-stage schema contract.
    """
    n = len(df)
    for col, kinds in schema.items():
        if col in df.columns:
            continue
        if "f" in kinds:
            df[col] = np.zeros(n, dtype=np.float32)
        elif "b" in kinds:
            df[col] = np.zeros(n, dtype=bool)
        else:
            df[col] = np.zeros(n, dtype=np.int32)
    return df


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
        mating_lambda=0.5,
        p_mztwin=0.02,
        A1=0.5,
        C1=0.2,
        E1=0.3,
        A2=0.5,
        C2=0.2,
        E2=0.3,
        rA=0.3,
        rC=0.5,
        assort1=0.0,
        assort2=0.0,
    )


@pytest.fixture
def tiny_pedigree_parquet(tmp_path: Path) -> Path:
    """Tiny schema-valid pedigree parquet (N=100, two generations)."""
    pedigree = run_simulation(
        seed=42,
        N=100,
        G_ped=2,
        G_sim=2,
        mating_lambda=0.5,
        p_mztwin=0.0,
        A1=0.5,
        C1=0.2,
        E1=0.3,
        A2=0.5,
        C2=0.2,
        E2=0.3,
        rA=0.3,
        rC=0.5,
        assort1=0.0,
        assort2=0.0,
    )
    path = tmp_path / "pedigree.parquet"
    pedigree.to_parquet(path)
    return path


@pytest.fixture
def tiny_phenotype_parquet(tmp_path: Path, tiny_pedigree_parquet: Path) -> Path:
    """Tiny schema-valid phenotype parquet built from the pedigree fixture."""
    from simace.phenotype import run_phenotype

    pedigree = pd.read_parquet(tiny_pedigree_parquet)
    phenotype = run_phenotype(
        pedigree,
        G_pheno=1,
        seed=42,
        standardize="global",
        phenotype_model1="frailty",
        phenotype_params1={"distribution": "weibull", "scale": 316.228, "rho": 2.0},
        beta1=1.0,
        beta_sex1=0.0,
        phenotype_model2="frailty",
        phenotype_params2={"distribution": "weibull", "scale": 316.228, "rho": 2.0},
        beta2=1.0,
        beta_sex2=0.0,
    )
    path = tmp_path / "phenotype.parquet"
    phenotype.to_parquet(path)
    return path


@pytest.fixture
def tiny_censored_parquet(tmp_path: Path, tiny_phenotype_parquet: Path) -> Path:
    """Tiny schema-valid censored parquet built from the phenotype fixture."""
    from simace.censoring.censor import run_censor

    phenotype = pd.read_parquet(tiny_phenotype_parquet)
    censored = run_censor(
        phenotype,
        censor_age=80.0,
        seed=42,
        gen_censoring={},
        death_scale=79.433,
        death_rho=10.0,
    )
    path = tmp_path / "censored.parquet"
    censored.to_parquet(path)
    return path


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

    parents, twins, household_ids = mating(rng, sex, mating_lambda=0.5, p_mztwin=0.02)
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
