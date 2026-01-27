"""
Frailty model phenotype simulation.

Converts liability to binary affected status with age-at-onset
using proportional hazards frailty model with Weibull baseline.

Model:
    - Liability: L = A + C + E (from pedigree)
    - Frailty: z = exp(L)
    - Survival function: S(t|z) = exp(-lambda * t^k * z)
    - Age-at-onset via inverse CDF: t = ((-log(U)) / (lambda * z))^(1/k)
"""

import numpy as np
import pandas as pd


def simulate_phenotype(liability, beta, rate, k, seed, standardize=True):
    """Apply frailty model to convert liability to phenotype.

    Args:
        liability: quantitative phenotype
        beta: effect of liability on log-hazard
        rate: Weibull scale parameter
        k: Weibull shape parameter
        seed: Random seed
        standardize: standardize liability before phenotype


    Returns:
        time of onset
    """
    rng = np.random.default_rng(seed)
    if standardize:
        lmean = liability.mean()
        lstd = np.std(liability)
        liability = (liability - lmean)/ lstd

    frailty = np.exp(beta * liability)

    # Simulate age-at-onset via inverse CDF
    u = rng.uniform(size=len(liability))
    t = ((-np.log(u)) / (rate * frailty)) ** (1 / k)

    return t

def age_censor(t, age):
    age = int(np.floor(age))
    censored = (t>age)
    
    t[censored] = age

    return pd.DataFrame({
        't': t,
        'age_censored': censored,
    })

def death_censor(t, seed, rate=1e-19, k=10):

    rng = np.random.default_rng(seed)
    u = rng.uniform(size=len(t))
    dt = ((-np.log(u)) / rate ) ** (1 / k)
    censored = (t > dt)

    t[censored] = dt[censored]

    return pd.DataFrame({
        't': t,
        'death_censored': censored,
    })


if __name__ == "__main__":
    # Read pedigree
    pedigree = pd.read_parquet(snakemake.input.pedigree)

    # Step 1: Simulate uncensored time-to-onset
    t_raw = simulate_phenotype(
        liability=pedigree['liability'].values,
        beta=snakemake.params.beta,
        rate=snakemake.params.rate,
        k=snakemake.params.k,
        seed=snakemake.params.seed,
        standardize=snakemake.params.standardize
    )

    # Step 2: Apply age censoring first
    age_result = age_censor(t_raw.copy(), snakemake.params.censor_age)

    # Step 3: Apply death censoring sequentially to age-censored result
    death_result = death_censor(
        age_result['t'].values.copy(),
        seed=snakemake.params.seed + 1000,
        rate=snakemake.params.death_rate,
        k=snakemake.params.death_k,
    )

    # Combine into single DataFrame
    phenotype = pd.DataFrame({
        'id': pedigree['id'],
        't': t_raw,
        'age_censored': age_result['age_censored'],
        't_observed': death_result['t'],
        'death_censored': death_result['death_censored'],
    })

    # affected = onset before any censoring
    phenotype['affected'] = ~phenotype['age_censored'] & ~phenotype['death_censored']

    phenotype.to_parquet(snakemake.output.phenotype, index=False)
