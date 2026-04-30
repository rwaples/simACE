"""Smoke tests for the quarantined prototype phenotype models.

These models are not part of the supported simulation surface (they are not
registered in ``simace.phenotyping.models`` and not reachable from Snakemake).
This module exists to catch silent rot when shared dependencies (notably
``simace.phenotyping.hazards``) evolve.
"""

import numpy as np

from simace.phenotyping._prototypes.bimodal_phenotype import (
    phenotype_mixture_cip,
    phenotype_mixture_cure_frailty,
    phenotype_two_threshold,
)


def _liability(n: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def test_mixture_cip_runs():
    t = phenotype_mixture_cip(
        liability=_liability(),
        prevalence=0.1,
        beta=1.0,
        pi=0.5,
        cip_x0_1=10.0,
        cip_k_1=0.3,
        cip_x0_2=40.0,
        cip_k_2=0.2,
        seed=42,
    )
    assert t.shape == (200,)
    assert np.all(np.isfinite(t))
    assert np.all(t > 0)


def test_mixture_cure_frailty_runs():
    weibull = {"scale": 316.228, "rho": 2.0}
    t = phenotype_mixture_cure_frailty(
        liability=_liability(),
        prevalence=0.1,
        beta=1.0,
        pi=0.5,
        baseline="weibull",
        hazard_params_1=weibull,
        hazard_params_2={"scale": 100.0, "rho": 1.5},
        seed=42,
    )
    assert t.shape == (200,)
    assert np.all(np.isfinite(t))
    assert np.all(t > 0)


def test_two_threshold_runs():
    t = phenotype_two_threshold(
        liability=_liability(),
        prevalence_early=0.05,
        prevalence_late=0.10,
        beta=1.0,
    )
    assert t.shape == (200,)
    assert np.all(np.isfinite(t))
    assert np.all(t > 0)
