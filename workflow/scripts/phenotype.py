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

import numpy as np
import pandas as pd


def simulate_phenotype(liability, beta, rate, k, seed, standardize=True):
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


def age_censor(t, left, right):
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


def death_censor(t, seed, rate=1e-19, k=10):
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


if __name__ == "__main__":
    # Read pedigree
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    params = snakemake.params

    # Filter to last G_pheno generations
    max_gen = pedigree["generation"].max()
    min_pheno_gen = max_gen - params.G_pheno + 1
    assert min_pheno_gen >= 0, f"G_pheno ({params.G_pheno}) > available generations ({max_gen + 1})"
    pedigree = pedigree[pedigree["generation"] >= min_pheno_gen].reset_index(drop=True)

    # Per-generation censoring windows.
    # Only the last 3 generations get explicit censoring windows.
    # Earlier phenotyped generations (if G_pheno > 3) use the default
    # window [0, censor_age] (unrestricted observation).
    generations = pedigree["generation"].values
    max_gen = generations.max()
    gen_windows = {
        max_gen: params.young_gen_censoring,
        max_gen - 1: params.middle_gen_censoring,
        max_gen - 2: params.old_gen_censoring,
    }
    left_censor = np.zeros(len(pedigree))
    right_censor = np.full(len(pedigree), float(params.censor_age))
    for gen, (lo, hi) in gen_windows.items():
        mask = generations == gen
        left_censor[mask] = lo
        right_censor[mask] = hi

    # Single death age per individual, shared across both traits.
    # This creates dependent censoring between traits (biologically correct:
    # death censors all outcomes simultaneously).
    rng_death = np.random.default_rng(params.seed + 1000)
    u_death = 1.0 - rng_death.uniform(size=len(pedigree))
    death_age = ((-np.log(u_death)) / params.death_rate) ** (1 / params.death_k)

    # === Trait 1 ===
    t1_raw = simulate_phenotype(
        liability=pedigree["liability1"].values,
        beta=params.beta1,
        rate=params.rate1,
        k=params.k1,
        seed=params.seed,
        standardize=params.standardize,
    )
    t1_after_age, age_censored1 = age_censor(t1_raw.copy(), left_censor, right_censor)
    death_censored1 = t1_after_age > death_age
    t_observed1 = np.where(death_censored1, death_age, t1_after_age)

    # === Trait 2 ===
    t2_raw = simulate_phenotype(
        liability=pedigree["liability2"].values,
        beta=params.beta2,
        rate=params.rate2,
        k=params.k2,
        seed=params.seed + 100,
        standardize=params.standardize,
    )
    t2_after_age, age_censored2 = age_censor(t2_raw.copy(), left_censor, right_censor)
    death_censored2 = t2_after_age > death_age
    t_observed2 = np.where(death_censored2, death_age, t2_after_age)

    # Combine into single DataFrame — use .values to avoid index alignment overhead
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

    phenotype.to_parquet(snakemake.output.phenotype, index=False)
