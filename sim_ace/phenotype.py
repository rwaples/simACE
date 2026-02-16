"""
Frailty model phenotype simulation for two correlated traits.

Converts liability to binary affected status with age-at-onset
using proportional hazards frailty model with Weibull baseline.

Model (per trait):
    - Liability: L = A + C + E (from pedigree)
    - Frailty: z = exp(beta * L)
    - Survival function: S(t|z) = exp(-rate * t^k * z)
    - Age-at-onset via inverse CDF: t = ((-log(U)) / (rate * z))^(1/k)
"""

from __future__ import annotations

import argparse
from typing import Any

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_phenotype(liability: np.ndarray, beta: float, rate: float, k: float, seed: int, standardize: bool = True) -> np.ndarray:
    """Apply frailty model to convert liability to phenotype.

    Args:
        liability: quantitative phenotype (array)
        beta: effect of liability on log-hazard
        rate: Weibull scale parameter
        k: Weibull shape parameter
        seed: random seed
        standardize: standardize liability before phenotype

    Returns:
        Array of simulated time-to-onset values

    Raises:
        ValueError: if rate or k is non-positive, or beta is non-finite
    """
    if not (rate > 0):
        raise ValueError(f"rate must be > 0, got {rate}")
    if not (k > 0):
        raise ValueError(f"k must be > 0, got {k}")
    if not np.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")

    rng = np.random.default_rng(seed)
    if standardize:
        lmean = liability.mean()
        lstd = np.std(liability)
        if lstd > 0:
            liability = (liability - lmean) / lstd
        else:
            liability = liability - lmean

    frailty = np.exp(beta * liability)

    # Simulate age-at-onset via inverse CDF
    # Use 1 - uniform to sample from (0, 1] and avoid log(0)
    u = 1.0 - rng.uniform(size=len(liability))
    t = ((-np.log(u)) / (rate * frailty)) ** (1 / k)

    return t


def age_censor(t: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-individual [left, right] age censoring.

    - t < left: left-censored (onset before observation window), t set to left
    - t > right: right-censored (onset after observation window), t set to right

    Args:
        t: array of time-to-onset values
        left: array of left-censoring ages (per individual)
        right: array of right-censoring ages (per individual)

    Returns:
        (t_censored, age_censored): tuple of arrays
    """
    left_trunc = t < left
    right_cens = t > right
    censored = left_trunc | right_cens
    t_out = np.clip(t, left, right)

    return t_out, censored


def death_censor(t: np.ndarray, seed: int, rate: float = 1e-19, k: float = 10) -> tuple[np.ndarray, np.ndarray]:
    """Apply competing risk death censoring with Weibull hazard.

    Args:
        t: array of time-to-onset values
        seed: random seed
        rate: Weibull scale parameter for death hazard
        k: Weibull shape parameter for death hazard

    Returns:
        (t_censored, death_censored): tuple of arrays
    """
    rng = np.random.default_rng(seed)
    u = 1.0 - rng.uniform(size=len(t))
    dt = ((-np.log(u)) / rate) ** (1 / k)
    censored = t > dt

    t[censored] = dt[censored]

    return t, censored


def run_phenotype(pedigree: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Orchestrate phenotype simulation from pedigree and parameter dict.

    Args:
        pedigree: DataFrame with pedigree data
        params: dict with keys: G_pheno, censor_age, young_gen_censoring,
                middle_gen_censoring, old_gen_censoring, seed, death_rate,
                death_k, beta1, rate1, k1, beta2, rate2, k2, standardize

    Returns:
        phenotype DataFrame
    """
    logger.info("Running phenotype simulation for %d individuals", len(pedigree))
    t0 = time.perf_counter()
    # Filter to last G_pheno generations
    max_gen = pedigree["generation"].max()
    min_pheno_gen = max_gen - params["G_pheno"] + 1
    assert min_pheno_gen >= 0, f"G_pheno ({params['G_pheno']}) > available generations ({max_gen + 1})"
    pedigree = pedigree[pedigree["generation"] >= min_pheno_gen].reset_index(drop=True)

    # Per-generation censoring windows
    generations = pedigree["generation"].values
    max_gen = generations.max()
    gen_windows = {
        max_gen: params["young_gen_censoring"],
        max_gen - 1: params["middle_gen_censoring"],
        max_gen - 2: params["old_gen_censoring"],
    }
    left_censor = np.zeros(len(pedigree))
    right_censor = np.full(len(pedigree), float(params["censor_age"]))
    for gen, (lo, hi) in gen_windows.items():
        mask = generations == gen
        left_censor[mask] = lo
        right_censor[mask] = hi

    # Single death age per individual, shared across both traits
    rng_death = np.random.default_rng(params["seed"] + 1000)
    u_death = 1.0 - rng_death.uniform(size=len(pedigree))
    death_age = ((-np.log(u_death)) / params["death_rate"]) ** (1 / params["death_k"])

    # === Trait 1 ===
    t1_raw = simulate_phenotype(
        liability=pedigree["liability1"].values,
        beta=params["beta1"], rate=params["rate1"], k=params["k1"],
        seed=params["seed"], standardize=params["standardize"],
    )
    t1_after_age, age_censored1 = age_censor(t1_raw.copy(), left_censor, right_censor)
    death_censored1 = t1_after_age > death_age
    t_observed1 = np.where(death_censored1, death_age, t1_after_age)

    # === Trait 2 ===
    t2_raw = simulate_phenotype(
        liability=pedigree["liability2"].values,
        beta=params["beta2"], rate=params["rate2"], k=params["k2"],
        seed=params["seed"] + 100, standardize=params["standardize"],
    )
    t2_after_age, age_censored2 = age_censor(t2_raw.copy(), left_censor, right_censor)
    death_censored2 = t2_after_age > death_age
    t_observed2 = np.where(death_censored2, death_age, t2_after_age)

    phenotype = pd.DataFrame(
        {
            "id": pedigree["id"].values,
            "generation": pedigree["generation"].values,
            "mother": pedigree["mother"].values,
            "father": pedigree["father"].values,
            "twin": pedigree["twin"].values,
            "A1": pedigree["A1"].values,
            "C1": pedigree["C1"].values,
            "E1": pedigree["E1"].values,
            "liability1": pedigree["liability1"].values,
            "A2": pedigree["A2"].values,
            "C2": pedigree["C2"].values,
            "E2": pedigree["E2"].values,
            "liability2": pedigree["liability2"].values,
            "death_age": death_age,
            # Trait 1
            "t1": t1_raw,
            "age_censored1": age_censored1,
            "t_observed1": t_observed1,
            "death_censored1": death_censored1,
            "affected1": ~age_censored1 & ~death_censored1,
            # Trait 2
            "t2": t2_raw,
            "age_censored2": age_censored2,
            "t_observed2": t_observed2,
            "death_censored2": death_censored2,
            "affected2": ~age_censored2 & ~death_censored2,
        }
    )

    # Data shape: prevalence after censoring
    prev1 = phenotype["affected1"].mean()
    prev2 = phenotype["affected2"].mean()
    logger.info("Prevalence after censoring: trait1=%.3f, trait2=%.3f", prev1, prev2)

    elapsed = time.perf_counter() - t0
    logger.info("Phenotype simulation complete in %.1fs: %d individuals", elapsed, len(phenotype))

    return phenotype


def cli() -> None:
    """Command-line interface for phenotype simulation."""
    from sim_ace import setup_logging
    parser = argparse.ArgumentParser(description="Simulate Weibull frailty phenotype")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG output")
    parser.add_argument("-q", "--quiet", action="store_true", help="WARNING+ only")
    parser.add_argument("--pedigree", required=True, help="Input pedigree parquet")
    parser.add_argument("--output", required=True, help="Output phenotype parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--G-pheno", type=int, default=3, help="Number of generations to assign phenotypes")
    parser.add_argument("--censor-age", type=float, default=100, help="Maximum follow-up age")
    parser.add_argument("--beta1", type=float, default=1.0, help="Weibull frailty coefficient for trait 1")
    parser.add_argument("--rate1", type=float, default=1e-5, help="Weibull baseline rate for trait 1")
    parser.add_argument("--k1", type=float, default=2.0, help="Weibull shape parameter for trait 1")
    parser.add_argument("--beta2", type=float, default=1.0, help="Weibull frailty coefficient for trait 2")
    parser.add_argument("--rate2", type=float, default=1e-5, help="Weibull baseline rate for trait 2")
    parser.add_argument("--k2", type=float, default=2.0, help="Weibull shape parameter for trait 2")
    parser.add_argument("--death-rate", type=float, default=1e-19, help="Competing death hazard rate")
    parser.add_argument("--death-k", type=float, default=10, help="Competing death hazard shape")
    parser.add_argument("--standardize", action="store_true", default=True, help="Standardize liability before phenotype simulation")
    parser.add_argument("--young-gen-censoring", type=float, nargs=2, default=[20, 50], help="Age censoring range for youngest generation (min max)")
    parser.add_argument("--middle-gen-censoring", type=float, nargs=2, default=[40, 70], help="Age censoring range for middle generation (min max)")
    parser.add_argument("--old-gen-censoring", type=float, nargs=2, default=[60, 90], help="Age censoring range for oldest generation (min max)")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=level)

    pedigree = pd.read_parquet(args.pedigree)
    params = {
        "G_pheno": args.G_pheno, "censor_age": args.censor_age, "seed": args.seed,
        "young_gen_censoring": args.young_gen_censoring,
        "middle_gen_censoring": args.middle_gen_censoring,
        "old_gen_censoring": args.old_gen_censoring,
        "death_rate": args.death_rate, "death_k": args.death_k,
        "beta1": args.beta1, "rate1": args.rate1, "k1": args.k1,
        "beta2": args.beta2, "rate2": args.rate2, "k2": args.k2,
        "standardize": args.standardize,
    }
    phenotype = run_phenotype(pedigree, params)
    phenotype.to_parquet(args.output, index=False)
